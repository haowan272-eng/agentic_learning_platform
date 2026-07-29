"""统一Markdown转换与分块测试，不调用外部视觉服务"""
from app.rag.chunker import (
    SUPPORTED_DOCUMENT_EXTENSIONS,
    _merge_image_text,
    chunk_markdown,
    chunk_text,
    clean_markdown,
    clean_text,
    convert_to_markdown_async,
)


def test_clean_text_removes_extra_spaces_and_blank_lines():
    raw = "\ufeff  第一段\t内容  \r\n\r\n\r\n  第二段\u3000内容  \n\n"
    assert clean_text(raw) == "第一段内容\n\n第二段内容"


def test_clean_markdown_preserves_structure():
    raw = "# 标题\n\n\n-  项目一\n\n```python\n  x = 1\n```"
    cleaned = clean_markdown(raw)
    assert cleaned.startswith("# 标题")
    assert "- 项目一" in cleaned
    assert "  x = 1" in cleaned


def test_chunk_text_applies_cleaning_before_split():
    chunks = chunk_text("  北京   故宫。\n\n\n  上海\t外滩。 ", size=100, overlap=0)
    assert chunks == ["北京 故宫。上海 外滩"]


def test_supported_document_formats_cover_text_office_and_images():
    expected = {".pdf", ".doc", ".docx", ".pptx", ".xlsx", ".txt", ".md", ".png", ".jpg", ".gif"}
    assert expected <= SUPPORTED_DOCUMENT_EXTENSIONS


def test_txt_fallback_converts_to_markdown(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.chunker._get_markitdown_instance", lambda: None)
    path = tmp_path / "guide.txt"
    path.write_text("  北京   故宫。\n\n上海 外滩", encoding="utf-8")
    import asyncio

    markdown = asyncio.run(convert_to_markdown_async(str(path)))
    assert "北京 故宫" in markdown


def test_markdown_heading_structure_controls_chunks():
    markdown = "# 北京攻略\n\n## 交通\n\n地铁出行。\n\n## 住宿\n\n酒店信息。"
    chunks = chunk_markdown(markdown, "md", chunk_tokens=100, overlap_tokens=0)
    assert len(chunks) == 2
    assert chunks[0].heading_path == "北京攻略 > 交通"
    assert chunks[1].heading_path == "北京攻略 > 住宿"


def test_image_markdown_is_atomic_and_searchable():
    markdown = "# 图片内容\n\n北京中心城区路线图。故宫、天安门、王府井。"
    chunks = chunk_markdown(markdown, "png", chunk_tokens=100, overlap_tokens=0)
    assert len(chunks) == 1
    assert chunks[0].modality == "image"
    assert "故宫" in chunks[0].embedding_content


def test_ocr_and_visual_text_merge_without_source_labels():
    merged = _merge_image_text(
        "故宫、天安门、王府井",
        "故宫、天安门、王府井。路线按照由西向东的顺序连接",
    )
    assert merged.count("故宫") == 1
    assert "由西向东" in merged
    assert "OCR" not in merged and "Qwen" not in merged


def test_markdown_table_stays_intact():
    table = "| 景点 | 时间 |\n| --- | --- |\n| 故宫 | 上午 |"
    chunks = chunk_markdown(f"# 行程\n\n{table}", "docx", chunk_tokens=100, overlap_tokens=0)
    assert len(chunks) == 1
    assert "| 故宫 | 上午 |" in chunks[0].content
