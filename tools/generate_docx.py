import sys
from pathlib import Path
from docx import Document
from docx.shared import Pt, Inches


def is_section_heading(line: str) -> bool:
    s = line.strip()
    if not s:
        return False
    # ABSTRACT or other all-caps single-word headings
    if s.isupper() and len(s.split()) <= 5:
        return True
    # Numbered headings like "1. INTRODUCTION"
    if s[0].isdigit() and s[1:3] == '. ':
        return True
    return False


def build_docx(txt_path: Path, docx_path: Path) -> None:
    text = txt_path.read_text(encoding='utf-8')
    lines = text.splitlines()

    doc = Document()

    # Page setup: 1 inch margins
    section = doc.sections[0]
    section.top_margin = Inches(1)
    section.bottom_margin = Inches(1)
    section.left_margin = Inches(1)
    section.right_margin = Inches(1)

    # Base style
    normal_style = doc.styles['Normal']
    normal_font = normal_style.font
    normal_font.name = 'Times New Roman'
    normal_font.size = Pt(12)

    # Title: first non-empty line
    idx = 0
    while idx < len(lines) and not lines[idx].strip():
        idx += 1
    if idx < len(lines):
        title_text = lines[idx].strip()
        title_p = doc.add_paragraph()
        run = title_p.add_run(title_text)
        run.bold = True
        run.font.size = Pt(16)
    idx += 1

    # Add a blank line after title
    doc.add_paragraph("")

    # Process remaining lines
    for i in range(idx, len(lines)):
        line = lines[i]
        s = line.strip()
        if not s:
            # blank line -> paragraph break
            doc.add_paragraph("")
            continue
        if is_section_heading(s):
            p = doc.add_paragraph()
            r = p.add_run(s)
            r.bold = True
            r.font.size = Pt(14)
        else:
            p = doc.add_paragraph(s)
            # Slight spacing after paragraphs for readability
            p.paragraph_format.space_after = Pt(6)

    doc.save(str(docx_path))


def main():
    if len(sys.argv) < 3:
        print("Usage: generate_docx.py <input.txt> <output.docx>")
        sys.exit(1)
    txt_path = Path(sys.argv[1])
    docx_path = Path(sys.argv[2])
    if not txt_path.exists():
        print(f"Input file not found: {txt_path}")
        sys.exit(2)
    docx_path.parent.mkdir(parents=True, exist_ok=True)
    build_docx(txt_path, docx_path)
    print(f"Wrote DOCX: {docx_path}")


if __name__ == '__main__':
    main()
