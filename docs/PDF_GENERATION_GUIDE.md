# PDF Generation Guide for KBS Documentation

## 🎯 Quick PDF Download Options

### Option 1: GitPrint (Easiest)
Click the "Download PDF" badge at the top of [KBS_FINAL_DOCUMENTATION.md](../KBS_FINAL_DOCUMENTATION.md)

This uses GitPrint.com to automatically convert the markdown to PDF.

### Option 2: Browser Print to PDF
1. Open [KBS_FINAL_DOCUMENTATION.md](../KBS_FINAL_DOCUMENTATION.md) on GitHub
2. Press `Ctrl+P` (Windows/Linux) or `Cmd+P` (Mac)
3. Select "Save as PDF" as the destination
4. Click "Save"

### Option 3: Markdown to PDF Converter (Local)

#### Using `pandoc` (Recommended):
```bash
# Install pandoc
# Ubuntu/Debian:
sudo apt-get install pandoc texlive-latex-base texlive-fonts-recommended

# macOS:
brew install pandoc basictex

# Windows: Download from https://pandoc.org/installing.html

# Generate PDF
pandoc KBS_FINAL_DOCUMENTATION.md -o KBS_FINAL_DOCUMENTATION.pdf \
  --pdf-engine=xelatex \
  --toc \
  --number-sections \
  -V geometry:margin=1in \
  -V fontsize=11pt
```

#### Using Python `markdown-pdf`:
```bash
# Install
pip install markdown-pdf

# Generate PDF
markdown-pdf KBS_FINAL_DOCUMENTATION.md -o KBS_FINAL_DOCUMENTATION.pdf
```

#### Using `grip` (GitHub-style rendering):
```bash
# Install
pip install grip

# Export to HTML first
grip KBS_FINAL_DOCUMENTATION.md --export KBS_FINAL_DOCUMENTATION.html

# Then convert HTML to PDF using browser or wkhtmltopdf
wkhtmltopdf KBS_FINAL_DOCUMENTATION.html KBS_FINAL_DOCUMENTATION.pdf
```

### Option 4: VS Code Extension
1. Install "Markdown PDF" extension in VS Code
2. Open `KBS_FINAL_DOCUMENTATION.md`
3. Press `Ctrl+Shift+P` and type "Markdown PDF: Export (pdf)"
4. PDF will be saved in the same directory

### Option 5: Online Markdown to PDF Converters
- **Dillinger**: https://dillinger.io/ (paste markdown, download PDF)
- **Markdown to PDF**: https://www.markdowntopdf.com/
- **CloudConvert**: https://cloudconvert.com/md-to-pdf

## 📋 PDF Formatting Tips

### Best Quality PDF:
1. Use **pandoc** with LaTeX engine (Option 3)
2. Includes table of contents, page numbers, proper formatting
3. Professional appearance

### Quick and Easy:
1. Use **GitPrint badge** (Option 1)
2. No installation required
3. Works directly from GitHub

### Custom Styling:
Create a custom CSS file for grip or pandoc:

```css
/* custom-style.css */
body {
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    max-width: 900px;
    margin: 0 auto;
    padding: 20px;
}

h1 {
    color: #2c3e50;
    border-bottom: 3px solid #3498db;
    padding-bottom: 10px;
}

code {
    background-color: #f4f4f4;
    padding: 2px 6px;
    border-radius: 3px;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 20px 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}
```

Then use with pandoc:
```bash
pandoc KBS_FINAL_DOCUMENTATION.md -o KBS_FINAL_DOCUMENTATION.pdf \
  --css custom-style.css \
  --pdf-engine=wkhtmltopdf
```

## 🚀 Automated PDF Generation (GitHub Actions)

To automatically generate PDF on every commit, create `.github/workflows/generate-pdf.yml`:

```yaml
name: Generate PDF Documentation

on:
  push:
    paths:
      - 'KBS_FINAL_DOCUMENTATION.md'

jobs:
  convert_via_pandoc:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Create PDF
        uses: docker://pandoc/latex:latest
        with:
          args: >
            KBS_FINAL_DOCUMENTATION.md
            -o KBS_FINAL_DOCUMENTATION.pdf
            --pdf-engine=xelatex
            --toc
            --number-sections
            -V geometry:margin=1in
      
      - name: Upload PDF artifact
        uses: actions/upload-artifact@v3
        with:
          name: documentation-pdf
          path: KBS_FINAL_DOCUMENTATION.pdf
      
      - name: Commit PDF
        run: |
          git config --local user.email "action@github.com"
          git config --local user.name "GitHub Action"
          git add KBS_FINAL_DOCUMENTATION.pdf
          git commit -m "Auto-generate PDF documentation" || echo "No changes"
          git push
```

## 📄 Expected PDF Output

The generated PDF will include:
- ✅ Title page with badges and metadata
- ✅ Table of contents (if using pandoc)
- ✅ All 9 main sections
- ✅ Formatted code blocks and tables
- ✅ Proper heading hierarchy
- ✅ Page numbers and cross-references
- ✅ ~40-50 pages (depending on formatting)

## 🔗 Additional Resources

- **Pandoc User Guide**: https://pandoc.org/MANUAL.html
- **Markdown Syntax**: https://www.markdownguide.org/
- **LaTeX Installation**: https://www.latex-project.org/get/

---

**Last Updated**: April 13, 2026
