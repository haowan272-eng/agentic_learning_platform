"""Qwen-VL/vLLM 图片语义描述器。

vLLM 提供 OpenAI-compatible API，因此索引端只依赖项目已有的 openai SDK。
没有配置 VISION_MODEL 时返回 None，主流程自动降级为 OCR + 图片附近正文。
"""
from __future__ import annotations

import base64
import asyncio
import hashlib
import io
import json
import mimetypes
import os
from pathlib import Path
from typing import Optional

from app.core.config import (
    VISION_API_KEY,
    VISION_BASE_URL,
    VISION_CACHE_CAPTIONS,
    VISION_MAX_IMAGE_EDGE,
    VISION_MAX_RETRIES,
    VISION_MAX_TOKENS,
    VISION_MODEL,
    VISION_TIMEOUT_SECONDS,
)


CAPTION_PROMPT_VERSION = "qwen-vl-rag-v1"
CAPTION_PROMPT = """请分析这张从旅游文档或PDF中提取的图片，为 RAG 检索生成中文描述。请覆盖以下信息：
1. 图片类型，例如照片、地图、路线图、行程表、票价表或统计图；
2. 图片表达的核心内容；
3. 可确认的地点、景点、路线、时间、价格、数字和关键词；
4. 图中元素之间的关系，例如先后顺序、路线连接、数值变化；
5. 图片中的关键文字。
要求：直接输出紧凑的中文描述；不要使用Markdown表格；不要猜测看不清或不存在的信息"""


class QwenVLCaptioner:
    """通过 vLLM 的 OpenAI-compatible 接口调用 Qwen-VL。"""

    def __init__(
        self,
        api_key: str,
        model: str,
        base_url: str,
        timeout_seconds: float = 90.0,
        max_retries: int = 2,
        max_tokens: int = 400,
        max_image_edge: int = 1600,
        cache_captions: bool = True,
    ):
        if not model.strip():
            raise ValueError("VISION_MODEL 不能为空")
        if not base_url.strip():
            raise ValueError("VISION_BASE_URL 不能为空")

        from openai import AsyncOpenAI

        self.model = model.strip()
        self.max_tokens = max(64, max_tokens)
        self.max_image_edge = max(256, max_image_edge)
        self.cache_captions = cache_captions
        self.client = AsyncOpenAI(
            api_key=api_key.strip() or "EMPTY",
            base_url=base_url.rstrip("/"),
            timeout=timeout_seconds,
            max_retries=max(0, max_retries),
        )

    def _cache_key(self, image_bytes: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(image_bytes)
        digest.update(self.model.encode("utf-8"))
        digest.update(CAPTION_PROMPT_VERSION.encode("utf-8"))
        digest.update(str(self.max_image_edge).encode("ascii"))
        return digest.hexdigest()

    @staticmethod
    def _cache_path(image_path: Path) -> Path:
        return image_path.with_suffix(image_path.suffix + ".caption.json")

    def _read_cache(self, image_path: Path, cache_key: str) -> Optional[str]:
        if not self.cache_captions:
            return None
        try:
            data = json.loads(self._cache_path(image_path).read_text(encoding="utf-8"))
            if data.get("cache_key") == cache_key and isinstance(data.get("caption"), str):
                caption = data["caption"].strip()
                return caption or None
        except (OSError, ValueError, TypeError):
            pass
        return None

    def _write_cache(self, image_path: Path, cache_key: str, caption: str) -> None:
        if not self.cache_captions:
            return
        cache_path = self._cache_path(image_path)
        temp_path = cache_path.with_suffix(cache_path.suffix + f".{os.getpid()}.tmp")
        try:
            temp_path.write_text(
                json.dumps(
                    {
                        "cache_key": cache_key,
                        "model": self.model,
                        "prompt_version": CAPTION_PROMPT_VERSION,
                        "caption": caption,
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            temp_path.replace(cache_path)
        except OSError:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass

    def _image_data_url(self, path: Path, image_bytes: bytes) -> str:
        """限制图片分辨率，避免 PDF 大图导致显存、传输和推理成本失控"""
        mime_type = mimetypes.guess_type(path.name)[0] or "image/png"
        prepared = image_bytes
        try:
            from PIL import Image

            with Image.open(io.BytesIO(image_bytes)) as image:
                width, height = image.size
                if max(width, height) > self.max_image_edge:
                    image.thumbnail((self.max_image_edge, self.max_image_edge), Image.Resampling.LANCZOS)
                    if image.mode not in ("RGB", "RGBA"):
                        image = image.convert("RGB")
                    output = io.BytesIO()
                    image.save(output, format="PNG", optimize=True)
                    prepared = output.getvalue()
                    mime_type = "image/png"
        except Exception:
            # If Pillow is unavailable or decoding fails, send original bytes to the vision service.
            pass
        encoded = base64.b64encode(prepared).decode("ascii")
        return f"data:{mime_type};base64,{encoded}"

    async def __call__(self, image_path: str) -> str:
        path = Path(image_path)
        image_bytes = await asyncio.to_thread(path.read_bytes)
        cache_key = self._cache_key(image_bytes)
        cached = await asyncio.to_thread(self._read_cache, path, cache_key)
        if cached:
            return cached

        image_data_url = await asyncio.to_thread(self._image_data_url, path, image_bytes)
        from app.rag.distributed_limit import vision_global_slot

        async with vision_global_slot():
            response = await self.client.chat.completions.create(
                model=self.model,
                temperature=0,
                max_tokens=self.max_tokens,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": CAPTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {"url": image_data_url},
                            },
                        ],
                    }
                ],
            )
        content = response.choices[0].message.content
        if not isinstance(content, str) or not content.strip():
            raise ValueError("Qwen-VL 返回了空图片描述")
        caption = content.strip()[:4000]
        await asyncio.to_thread(self._write_cache, path, cache_key, caption)
        return caption

    async def aclose(self) -> None:
        await self.client.close()


def get_qwen_vl_captioner() -> Optional[QwenVLCaptioner]:
    """根据配置创建 Qwen-VL Captioner；VISION_MODEL 为空表示关闭"""
    if not VISION_MODEL.strip():
        return None
    return QwenVLCaptioner(
        api_key=VISION_API_KEY,
        model=VISION_MODEL,
        base_url=VISION_BASE_URL,
        timeout_seconds=VISION_TIMEOUT_SECONDS,
        max_retries=VISION_MAX_RETRIES,
        max_tokens=VISION_MAX_TOKENS,
        max_image_edge=VISION_MAX_IMAGE_EDGE,
        cache_captions=VISION_CACHE_CAPTIONS,
    )


__all__ = [
    "CAPTION_PROMPT",
    "QwenVLCaptioner",
    "get_qwen_vl_captioner",
]
