"""多格式文档到 Markdown 的结构感知分块。

公共管道只有两个核心阶段：
1. convert_to_markdown_async：PDF/图片增强处理，其他格式优先使用 MarkItDown。
2. chunk_markdown：按 Markdown 标题、段落和 token 预算统一分块。
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from sqlalchemy.orm import Session

from app.core.config import (
    RAG_OCR_CONCURRENCY,
    RAG_OCR_TIMEOUT_SECONDS,
    RAG_VISION_CONCURRENCY,
    VISION_MAX_CAPTIONS_PER_DOCUMENT,
    VISION_SKIP_CAPTION_OCR_CHARS,
    VISION_TEXT_DEDUP_THRESHOLD,
)
from app.models.document import Document
from app.rag.chunk_models import DocumentChunk

try:
    import fitz  # PyMuPDF

    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


CHUNK_TOKENS = 420
OVERLAP_TOKENS = 60
PARENT_TOKENS = 1400
MIN_IMAGE_WIDTH = 80
MIN_IMAGE_HEIGHT = 80

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".gif"}
TEXT_EXTENSIONS = {
    ".txt", ".md", ".csv", ".json", ".xml", ".html", ".htm",
    ".py", ".js", ".ts", ".java", ".yaml", ".yml", ".toml",
}
MARKITDOWN_EXTENSIONS = {
    ".doc", ".docx", ".pptx", ".xlsx", ".xls", ".epub", *TEXT_EXTENSIONS,
}
SUPPORTED_DOCUMENT_EXTENSIONS = {".pdf", *IMAGE_EXTENSIONS, *MARKITDOWN_EXTENSIONS}

logger = logging.getLogger(__name__)
_markitdown_instance = None


def clean_text(text: str) -> str:
    """清理普通文本中的多余空格、空行与不可见字符"""
    if not text:
        return ""
    normalized = (
        str(text)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u3000", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\x00", "")
    )
    lines: list[str] = []
    for raw_line in normalized.split("\n"):
        line = re.sub(r"[\t\f\v ]+", " ", raw_line).strip()
        if line:
            lines.append(line)
        elif lines and lines[-1] != "":
            lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines)


def clean_markdown(markdown: str) -> str:
    """清理Markdown空白，同时保留代码块、列表与表格结构"""
    if not markdown:
        return ""
    normalized = (
        str(markdown)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u00a0", " ")
        .replace("\u3000", " ")
        .replace("\u200b", "")
        .replace("\ufeff", "")
        .replace("\x00", "")
    )
    result: list[str] = []
    in_fence = False
    for raw_line in normalized.split("\n"):
        stripped = raw_line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            line = raw_line.rstrip()
        elif in_fence:
            line = raw_line.rstrip()
        else:
            leading = raw_line[: len(raw_line) - len(raw_line.lstrip())]
            content = re.sub(r"[\t\f\v ]+", " ", stripped)
            line = leading.replace("\t", "    ") + content if content else ""
        if line:
            result.append(line)
        elif result and result[-1] != "":
            result.append("")
    while result and result[-1] == "":
        result.pop()
    return "\n".join(result)


def approx_token_len(text: str) -> int:
    if not text:
        return 0
    cjk = len(re.findall(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", text))
    other = re.sub(r"[\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff]", " ", text)
    other_tokens = sum(
        max(1, (len(token) + 3) // 4)
        for token in re.findall(r"\w+|[^\w\s]", other)
    )
    return cjk + other_tokens


def _split_sentences(text: str) -> list[str]:
    return [
        part.strip()
        for part in re.split(r"(?<=[。！？?!.])\s*|\n+", text)
        if part.strip()
    ]


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    cjk = len(re.findall(r"[\u3400-\u9fff]", text))
    multiplier = 1.0 if cjk / max(1, len(text)) >= 0.30 else 3.5
    window = max(1, int(size * multiplier))
    overlap_chars = min(int(overlap * multiplier), window - 1)
    parts: list[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + window)
        part = text[start:end].strip()
        if part:
            parts.append(part)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap_chars)
    return parts


def chunk_text(text: str, size: int = CHUNK_TOKENS, overlap: int = OVERLAP_TOKENS) -> list[str]:
    """兼容普通文本的语义边界分块"""
    text = clean_text(text)
    if not text:
        return []
    if size <= 0:
        raise ValueError("size 必须大于0")
    overlap = min(max(0, overlap), size - 1)
    sentences: list[str] = []
    for sentence in _split_sentences(text):
        sentences.extend(
            _hard_split(sentence, size, overlap)
            if approx_token_len(sentence) > size
            else [sentence]
        )
    chunks: list[str] = []
    current: list[str] = []
    tokens = 0
    for sentence in sentences:
        count = max(1, approx_token_len(sentence))
        if current and tokens + count > size:
            chunks.append("".join(current))
            kept: list[str] = []
            kept_tokens = 0
            for previous in reversed(current):
                previous_count = max(1, approx_token_len(previous))
                if kept_tokens + previous_count > overlap:
                    break
                kept.append(previous)
                kept_tokens += previous_count
            current = list(reversed(kept))
            tokens = kept_tokens
            if current and tokens + count > size:
                current, tokens = [], 0
        current.append(sentence)
        tokens += count
    if current:
        final = "".join(current)
        if not chunks or chunks[-1] != final:
            chunks.append(final)
    return chunks


def _ocr_image(image_path: str) -> str:
    try:
        import pytesseract
        from PIL import Image

        with Image.open(image_path) as image:
            kwargs = {"lang": "chi_sim+eng"}
            if RAG_OCR_TIMEOUT_SECONDS > 0:
                kwargs["timeout"] = RAG_OCR_TIMEOUT_SECONDS
            return clean_text(pytesseract.image_to_string(image, **kwargs))
    except RuntimeError as exc:
        logger.warning("图片 OCR 超时或失败 %s: %s", image_path, exc)
        return ""
    except Exception:
        return ""


def _normalize_similarity(text: str) -> str:
    return "".join(re.findall(r"[\u3400-\u9fffA-Za-z0-9]", text or "")).lower()


def _similarity(left: str, right: str) -> float:
    def grams(value: str) -> set[str]:
        value = _normalize_similarity(value)
        if not value:
            return set()
        return {value[i:i + 2] for i in range(max(1, len(value) - 1))}

    left_grams, right_grams = grams(left), grams(right)
    if not left_grams or not right_grams:
        return 0.0
    return len(left_grams & right_grams) / len(left_grams | right_grams)


def _merge_image_text(ocr_text: str, visual_text: str) -> str:
    """合并OCR与视觉语义，最终内容不暴露提取器来源"""
    ocr_text, visual_text = clean_text(ocr_text), clean_text(visual_text)
    if not ocr_text:
        return visual_text
    if not visual_text:
        return ocr_text
    visual_sentences = [
        sentence
        for sentence in _split_sentences(visual_text)
        if _similarity(sentence, ocr_text) < VISION_TEXT_DEDUP_THRESHOLD
    ]
    return clean_text("\n\n".join([ocr_text, "".join(visual_sentences)]))


@dataclass
class _ImageContext:
    captioner: object = None
    calls: int = 0
    seen: dict[str, str] = field(default_factory=dict)
    emitted: set[str] = field(default_factory=set)
    ocr_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(RAG_OCR_CONCURRENCY)
    )
    vision_semaphore: asyncio.Semaphore = field(
        default_factory=lambda: asyncio.Semaphore(RAG_VISION_CONCURRENCY)
    )
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


def _new_image_context() -> _ImageContext:
    try:
        from app.rag.vision import get_qwen_vl_captioner

        captioner = get_qwen_vl_captioner()
    except Exception as exc:
        logger.warning("Qwen-VL初始化失败，图片仅使用OCR: %s", exc)
        captioner = None
    return _ImageContext(captioner=captioner)


async def _extract_image_text_async(image_path: str, context: _ImageContext) -> str:
    image_bytes = await asyncio.to_thread(Path(image_path).read_bytes)
    digest = hashlib.sha256(image_bytes).hexdigest()
    async with context.lock:
        if digest in context.seen:
            return context.seen[digest]
    async with context.ocr_semaphore:
        ocr_text = await asyncio.to_thread(_ocr_image, image_path)
    visual_text = ""
    ocr_chars = len(_normalize_similarity(ocr_text))
    should_caption = False
    async with context.lock:
        if (
            context.captioner is not None
            and VISION_MAX_CAPTIONS_PER_DOCUMENT > 0
            and context.calls < VISION_MAX_CAPTIONS_PER_DOCUMENT
            and (VISION_SKIP_CAPTION_OCR_CHARS <= 0 or ocr_chars < VISION_SKIP_CAPTION_OCR_CHARS)
        ):
            context.calls += 1
            should_caption = True
    if should_caption:
        try:
            async with context.vision_semaphore:
                visual_text = clean_text(await context.captioner(image_path))
        except Exception as exc:
            logger.warning("图片视觉提取失败，降级为OCR: %s", exc)
    merged = _merge_image_text(ocr_text, visual_text)
    async with context.lock:
        context.seen[digest] = merged
    return merged


def _image_digest(image_path: str) -> str:
    return hashlib.sha256(Path(image_path).read_bytes()).hexdigest()


async def _deduplicate_page_images_async(page_records: list[dict]) -> set[str]:
    """在线程池并发计算PDF图片Hash，并按文档顺序只保留首次出现"""
    candidates = [
        (record, number, image_path)
        for record in page_records
        for number, image_path in record["images"]
    ]
    if not candidates:
        return set()
    digests = await asyncio.gather(
        *(asyncio.to_thread(_image_digest, image_path) for _, _, image_path in candidates)
    )
    seen: set[str] = set()
    for record in page_records:
        record["images"] = []
    for (record, number, image_path), digest in zip(candidates, digests):
        if digest in seen:
            continue
        seen.add(digest)
        record["images"].append((number, image_path, digest))
    return seen


async def _extract_images_concurrently_async(
    image_paths: list[str],
    context: _ImageContext,
) -> list[str]:
    """协程调度OCR与VL；两类任务独立限流，并保持输入顺序"""
    if not image_paths:
        return []
    digests = await asyncio.gather(
        *(asyncio.to_thread(_image_digest, image_path) for image_path in image_paths)
    )
    digest_by_path = dict(zip(image_paths, digests))
    unique_by_digest: dict[str, str] = {}
    for image_path in image_paths:
        unique_by_digest.setdefault(digest_by_path[image_path], image_path)
    unique_paths = list(unique_by_digest.values())
    unique_results = await asyncio.gather(
        *(_extract_image_text_async(image_path, context) for image_path in unique_paths)
    )
    result_by_digest = {
        digest_by_path[image_path]: result
        for image_path, result in zip(unique_paths, unique_results)
    }
    return [result_by_digest[digest_by_path[image_path]] for image_path in image_paths]


async def _close_image_context(context: _ImageContext) -> None:
    close = getattr(context.captioner, "aclose", None)
    if close is not None:
        try:
            await close()
        except Exception as exc:
            logger.debug("关闭视觉客户端失败: %s", exc)


def _safe_asset_dir(path: str) -> Path:
    source = Path(path)
    output = source.parent / "rag_assets" / source.stem
    output.mkdir(parents=True, exist_ok=True)
    return output


def _image_size(path: str) -> tuple[int, int]:
    try:
        from PIL import Image

        with Image.open(path) as image:
            return image.size
    except Exception:
        return (0, 0)


async def _convert_image_to_markdown_async(path: str) -> str:
    context = _new_image_context()
    try:
        extracted = await _extract_image_text_async(path, context)
        if not extracted:
            raise ValueError(f"图片没有提取到可用文字或视觉内容: {Path(path).name}")
        return clean_markdown(f"# 图片内容\n\n{extracted}")
    finally:
        await _close_image_context(context)


async def _convert_pdf_to_markdown_async(path: str) -> str:
    if not HAS_PYMUPDF:
        raise ImportError("增强PDF解析需要安装PyMuPDF")
    asset_dir = _safe_asset_dir(path)
    started_at = time.monotonic()
    context = _new_image_context()
    parts = [f"# {Path(path).stem}"]
    with fitz.open(path) as pdf:
        page_count = len(pdf)
        page_records: list[dict] = []
        # Keep PyMuPDF reads sequential; OCR/vision work happens later.
        for page_index, page in enumerate(pdf, 1):
            page_text = clean_text(page.get_text("text", sort=True))
            image_number = 0
            page_images: list[tuple[int, str]] = []
            for image_info in page.get_images(full=True):
                xref = image_info[0]
                extracted_image = pdf.extract_image(xref)
                image_bytes = extracted_image.get("image")
                if not image_bytes:
                    continue
                extension = extracted_image.get("ext", "png")
                image_number += 1
                image_path = asset_dir / f"page_{page_index:04d}_image_{image_number:03d}.{extension}"
                image_path.write_bytes(image_bytes)
                width, height = _image_size(str(image_path))
                if width and height and (width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT):
                    continue
                page_images.append((image_number, str(image_path)))
            page_records.append({"index": page_index, "text": page_text, "images": page_images})

        scheduled_digests = await _deduplicate_page_images_async(page_records)
        all_images = [item for record in page_records for item in record["images"]]
        all_image_texts = await _extract_images_concurrently_async(
            [image_path for _, image_path, _ in all_images], context
        )
        extracted_by_path = {
            image_path: image_text
            for (_, image_path, _), image_text in zip(all_images, all_image_texts)
        }

        # 只有无正文且嵌入图片也没有结果的页面才渲染整页，避免重复 OCR/Caption。
        scan_records: list[tuple[int, str]] = []
        for record in page_records:
            has_image_text = any(
                extracted_by_path.get(image_path)
                for _, image_path, _ in record["images"]
            )
            if not record["text"] and not has_image_text:
                scan_path = asset_dir / f"page_{record['index']:04d}_scan.png"
                pdf[record["index"] - 1].get_pixmap(
                    matrix=fitz.Matrix(2, 2), alpha=False
                ).save(str(scan_path))
                scan_records.append((record["index"], str(scan_path)))

        scan_texts = await _extract_images_concurrently_async(
            [scan_path for _, scan_path in scan_records], context
        )
        scan_by_page = {
            page_index: scan_text
            for (page_index, _), scan_text in zip(scan_records, scan_texts)
        }

        # Concurrency only changes extraction; Markdown output stays page/image ordered.
        for record in page_records:
            parts.append(f"## 第{record['index']}")
            if record["text"]:
                parts.append(record["text"])
            for number, image_path, digest in record["images"]:
                image_text = extracted_by_path.get(image_path, "")
                if image_text:
                    context.emitted.add(digest)
                    parts.append(f"### 图片{number}\n\n{image_text}")
            scan_text = scan_by_page.get(record["index"], "")
            if scan_text:
                parts.append(f"### 页面图像\n\n{scan_text}")
    markdown = clean_markdown("\n\n".join(parts))
    if not markdown:
        raise ValueError(f"PDF 没有提取到可用内容: {Path(path).name}")
    await _close_image_context(context)
    logger.info(
        "PDF转换完成: pages=%d, images=%d, ocr_concurrency=%d, vl_concurrency=%d, qwen_calls=%d, elapsed=%.2fs",
        page_count,
        len(scheduled_digests),
        RAG_OCR_CONCURRENCY,
        RAG_VISION_CONCURRENCY,
        context.calls,
        time.monotonic() - started_at,
    )
    return markdown


def _get_markitdown_instance():
    global _markitdown_instance
    if _markitdown_instance is None:
        try:
            from markitdown import MarkItDown

            _markitdown_instance = MarkItDown()
        except ImportError:
            return None
    return _markitdown_instance


def _read_text_file(path: str) -> str:
    data = Path(path).read_bytes()
    if data.startswith((b"\xff\xfe", b"\xfe\xff")) or data[:200].count(b"\x00") > 20:
        try:
            return clean_text(data.decode("utf-16"))
        except UnicodeDecodeError:
            pass
    for encoding in ("utf-8-sig", "gb18030"):
        try:
            return clean_text(data.decode(encoding))
        except UnicodeDecodeError:
            continue
    return clean_text(data.decode("utf-8", errors="replace"))


def _fallback_text_reader(path: str) -> str:
    extension = Path(path).suffix.lower()
    if extension in TEXT_EXTENSIONS:
        return _read_text_file(path)
    if extension == ".doc":
        try:
            result = subprocess.run(
                ["antiword", path],
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Legacy .doc conversion requires antiword or MarkItDown support; install antiword in the runtime"
            ) from exc
        except subprocess.CalledProcessError as exc:
            message = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeError(f"antiword failed to convert .doc: {message or exc}") from exc
        text = clean_text(result.stdout)
        if not text:
            raise ValueError("antiword returned empty text for .doc")
        return text
    raise RuntimeError(
        f"MarkItDown不可用，无法转换{extension}；请执行 uv sync 安装pyproject.toml中的依赖"
    )


def _convert_with_markitdown(path: str) -> str:
    converter = _get_markitdown_instance()
    if converter is None:
        return _fallback_text_reader(path)
    try:
        result = converter.convert(path)
        markdown = getattr(result, "text_content", None)
        if isinstance(markdown, str) and markdown.strip():
            return clean_markdown(markdown)
        raise ValueError("MarkItDown转换结果为空")
    except Exception as exc:
        logger.warning("MarkItDown转换失败 %s: %s", path, exc)
        return _fallback_text_reader(path)


async def convert_to_markdown_async(path: str) -> str:
    """将任一支持格式统一转换为Markdown字符串"""
    if not os.path.exists(path):
        raise FileNotFoundError(f"文档不存在: {path}")
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise ValueError(f"不支持的文档格式: {extension or '无扩展名'}")
    if extension == ".pdf":
        return await _convert_pdf_to_markdown_async(path)
    if extension in IMAGE_EXTENSIONS:
        return await _convert_image_to_markdown_async(path)
    return await asyncio.to_thread(_convert_with_markitdown, path)


@dataclass
class MarkdownSection:
    heading_path: Optional[str]
    content: str


@dataclass
class MarkdownChunk:
    content: str
    embedding_content: str
    heading_path: Optional[str]
    source_type: str
    location: Optional[str]
    modality: str
    page_number: Optional[int] = None
    parent_content: Optional[str] = None

    def metadata(self) -> dict:
        return {
            "heading_path": self.heading_path,
            "source_type": self.source_type,
            "location": self.location,
            "modality": self.modality,
        }


def _split_markdown_sections(markdown: str) -> list[MarkdownSection]:
    sections: list[MarkdownSection] = []
    heading_stack: list[str] = []
    buffer: list[str] = []

    def flush() -> None:
        content = clean_markdown("\n".join(buffer))
        if content:
            sections.append(MarkdownSection(" > ".join(heading_stack) or None, content))
        buffer.clear()

    in_fence = False
    for line in clean_markdown(markdown).splitlines():
        stripped = line.strip()
        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
        heading = None if in_fence else re.match(r"^(#{1,6})\s+(.+?)\s*$", stripped)
        if heading:
            flush()
            level = len(heading.group(1))
            title = heading.group(2).strip()
            heading_stack = heading_stack[: level - 1]
            heading_stack.append(title)
        else:
            buffer.append(line)
    flush()
    if not sections and clean_markdown(markdown):
        sections.append(MarkdownSection(None, clean_markdown(markdown)))
    return sections


def _chunk_markdown_blocks(content: str, size: int, overlap: int) -> list[str]:
    blocks = [block.strip() for block in re.split(r"\n\s*\n", content) if block.strip()]
    expanded: list[str] = []
    for block in blocks:
        if approx_token_len(block) > size:
            expanded.extend(_split_oversized_markdown_block(block, size, overlap))
        else:
            expanded.append(block)
    chunks: list[str] = []
    current: list[str] = []
    tokens = 0
    for block in expanded:
        count = max(1, approx_token_len(block))
        if current and tokens + count > size:
            chunks.append("\n\n".join(current))
            kept: list[str] = []
            kept_tokens = 0
            for previous in reversed(current):
                previous_count = max(1, approx_token_len(previous))
                if kept_tokens + previous_count > overlap:
                    break
                kept.append(previous)
                kept_tokens += previous_count
            current, tokens = list(reversed(kept)), kept_tokens
            if current and tokens + count > size:
                current, tokens = [], 0
        current.append(block)
        tokens += count
    if current:
        final = "\n\n".join(current)
        if not chunks or chunks[-1] != final:
            chunks.append(final)
    return chunks


def _split_oversized_markdown_block(block: str, size: int, overlap: int) -> list[str]:
    """切分超长代码块、表格或正文，避免 embedding 模型静默截断。"""
    stripped = block.strip()
    if stripped.startswith(("```", "~~~")):
        lines = stripped.splitlines()
        fence = lines[0]
        closing = lines[-1] if len(lines) > 1 and lines[-1].startswith(("```", "~~~")) else fence[:3]
        body = "\n".join(lines[1:-1] if lines[-1] == closing else lines[1:])
        pieces = chunk_text(body, size=max(32, size - 8), overlap=min(overlap, max(0, size // 4)))
        return [f"{fence}\n{piece}\n{closing}" for piece in pieces] or [stripped]

    if stripped.startswith("|"):
        rows = stripped.splitlines()
        header = rows[:2] if len(rows) > 1 and re.match(r"^\s*\|?\s*:?-+", rows[1]) else rows[:1]
        data_rows = rows[len(header):]
        pieces: list[str] = []
        current = list(header)
        for row in data_rows:
            candidate = "\n".join(current + [row])
            if len(current) > len(header) and approx_token_len(candidate) > size:
                pieces.append("\n".join(current))
                current = list(header)
            current.append(row)
        if len(current) > len(header) or not pieces:
            pieces.append("\n".join(current))
        return pieces

    return chunk_text(stripped, size=size, overlap=overlap)


def chunk_markdown(
    markdown: str,
    source_type: str,
    chunk_tokens: int = CHUNK_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
    parent_tokens: int = PARENT_TOKENS,
) -> list[MarkdownChunk]:
    """按Markdown标题和块边界切分，图片内容与普通文本走同一管道"""
    chunks: list[MarkdownChunk] = []
    for section in _split_markdown_sections(markdown):
        for content in _chunk_markdown_blocks(section.content, chunk_tokens, overlap_tokens):
            heading = section.heading_path
            page_match = re.search(r"第(\d+)", heading or "")
            page_number = int(page_match.group(1)) if page_match else None
            modality = "image" if source_type in {ext.lstrip('.') for ext in IMAGE_EXTENSIONS} or "图片" in (heading or "") else "text"
            location = f"page:{page_number}" if page_number else heading
            embedding_content = f"标题：{heading}\n\n{content}" if heading else content
            chunks.append(
                MarkdownChunk(
                    content=content,
                    embedding_content=embedding_content,
                    heading_path=heading,
                    source_type=source_type,
                    location=location,
                    modality=modality,
                    page_number=page_number,
                )
            )

    window_radius = 1
    for index, chunk in enumerate(chunks):
        start = max(0, index - window_radius)
        end = min(len(chunks), index + window_radius + 1)
        chunk.parent_content = "\n\n".join(chunks[i].content for i in range(start, end))
    return chunks


async def chunk_document_async(
    document_id: int,
    db: Session,
    chunk_size: int = CHUNK_TOKENS,
    overlap: int = OVERLAP_TOKENS,
) -> list[dict]:
    """文档转Markdown、分块并幂等写入PostgreSQL"""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise ValueError(f"文档不存在: id={document_id}")
    file_path = doc.storage_key or doc.file_path
    if not file_path:
        raise ValueError(f"文档没有可用存储路径: id={document_id}")
    source_type = Path(file_path).suffix.lower().lstrip(".")
    markdown = await convert_to_markdown_async(file_path)
    chunks = chunk_markdown(markdown, source_type, chunk_size, overlap)
    if not chunks:
        raise ValueError(f"文档没有可索引内容: {doc.original_file_name or doc.file_name}")

    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    db.flush()
    records: list[DocumentChunk] = []
    for index, chunk in enumerate(chunks):
        record = DocumentChunk(
            document_id=document_id,
            content=chunk.content,
            embedding_content=chunk.embedding_content,
            chunk_index=index,
            modality=chunk.modality,
            page_start=chunk.page_number,
            page_end=chunk.page_number,
            metadata_json=json.dumps(chunk.metadata(), ensure_ascii=False),
        )
        db.add(record)
        records.append(record)
    db.commit()
    for record in records:
        db.refresh(record)
    return [
        {
            "id": record.id,
            "document_id": record.document_id,
            "chunk_index": record.chunk_index,
            "modality": record.modality,
            "page_start": record.page_start,
            "page_end": record.page_end,
            "metadata": json.loads(record.metadata_json or "{}"),
            "content_preview": record.content[:160] + ("..." if len(record.content) > 160 else ""),
        }
        for record in records
    ]


__all__ = [
    "SUPPORTED_DOCUMENT_EXTENSIONS",
    "chunk_document_async",
    "chunk_markdown",
    "chunk_text",
    "clean_markdown",
    "clean_text",
    "convert_to_markdown_async",
]
