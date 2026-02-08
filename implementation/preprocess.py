from __future__ import annotations

from pathlib import Path
from unstructured.partition.pdf import partition_pdf
import os
import re

def extract_markdown_from_pdf_unstructured(
    pdf_path: str | Path,
    *,
    strategy: str = "auto",   # auto | fast | hi_res | ocr_only
) -> str:
    """
    Extract text from a PDF using Unstructured and return Markdown-formatted text.
    Suitable for chunking & embedding (RAG).
    """
    pdf_path = Path(pdf_path)

    elements = partition_pdf(
        filename=str(pdf_path),
        strategy=strategy,
        infer_table_structure=True,
        extract_images=False,
    )

    md_lines: list[str] = []

    for e in elements:
        text = getattr(e, "text", None)
        if not text:
            continue

        text = text.strip()
        if not text:
            continue

        category = e.category

        # ---------- Titles ----------
        if category == "Title":
            md_lines.append(f"# {text}")

        # ---------- Tables ----------
        elif category == "Table":
            md_lines.append("\n```text")
            md_lines.append(text)
            md_lines.append("```\n")

        # ---------- Lists ----------
        elif category == "ListItem":
            md_lines.append(f"- {text}")

        # ---------- Normal paragraphs ----------
        else:
            md_lines.append(text)

    # Clean spacing
    markdown = "\n\n".join(md_lines)
    return markdown.strip()

def merge_label_value(lines: list[str]) -> list[str]:
    merged = []
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if (
            i + 1 < len(lines)
            and len(line) < 40
            and lines[i + 1].lstrip().startswith((":","- :"))
        ):
            value = lines[i + 1].replace("- :", ":").strip()
            merged.append(f"{line}{value}")
            i += 2
        else:
            merged.append(line)
            i += 1

    return merged

def dedupe_consecutive(lines: list[str]) -> list[str]:
    out = []
    prev = None
    for line in lines:
        if line != prev:
            out.append(line)
        prev = line
    return out

def clean_lines(md: str) -> list[str]:
    lines = [l.strip() for l in md.splitlines() if l.strip()]
    # Remove "Page X" or "Page X of Y" lines
    return [l for l in lines if not re.match(r"^page \d+", l, re.IGNORECASE)]

def normalize_markdown(md: str) -> str:
    lines = clean_lines(md)
    lines = dedupe_consecutive(lines)
    lines = merge_label_value(lines)
    return "\n\n".join(lines)

def main():
    raw_dir = Path("knowledge-base/raw")
    preprocessed_dir = Path("knowledge-base/preprocessed")
    preprocessed_dir.mkdir(parents=True, exist_ok=True)

    pdf_files = list(raw_dir.glob("*.pdf"))
    print(f"Found {len(pdf_files)} PDF files in {raw_dir}")

    for pdf_file in pdf_files:
        print(f"Processing {pdf_file.name}...")
        try:
            markdown_text = extract_markdown_from_pdf_unstructured(pdf_file)
            normalized_text = normalize_markdown(markdown_text)
            
            output_file = preprocessed_dir / (pdf_file.stem + ".md")
            output_file.write_text(normalized_text, encoding="utf-8")
            print(f"Saved to {output_file}")
        except Exception as e:
            print(f"Error processing {pdf_file.name}: {e}")

if __name__ == "__main__":
    main()
