import sys, markdown
from xhtml2pdf import pisa
src, dst = sys.argv[1], sys.argv[2]
figdir = sys.argv[3] if len(sys.argv) > 3 else "fig"
html = markdown.markdown(open(src, encoding="utf-8").read().replace("FIGDIR", figdir), extensions=["tables", "fenced_code"])
AMPLADES = {"step": 10, "what it does": 17, "how": 45, "why": 28, "stage": 11, "starting problem": 20, "what we built or tried": 31, "what we learned": 24, "kept": 14, "idea": 40, "versions tried (approx.)": 16, "what we kept": 37,
            "dataset": 19, "matches": 20, "role": 15, "labelled by": 24, "source · licence": 22, "situation": 55, "label": 45,
            "change to the external data": 36, "what it fixes": 28, "effect": 36, "limitation": 30, "what we did": 40, "what remains": 30}
import re as _re
html = _re.sub(r"<th>([^<]*)</th>", lambda m: f'<th width="{AMPLADES[m.group(1).lower()]}%">{m.group(1)}</th>' if m.group(1).lower() in AMPLADES else m.group(0), html)
import os as _os
_FD = _os.environ.get("TD_FONT_DIR", "/usr/share/fonts/truetype/dejavu")   # DejaVu covers → − ≤ ≠ (Helvetica does not)
_FACES = "".join(f"@font-face {{ font-family: DejaVu; src: url('{_FD}/{f}.ttf'); {w} }}\n" for f, w in (
    ("DejaVuSans", ""), ("DejaVuSans-Bold", "font-weight: bold;"), ("DejaVuSans", "font-style: italic;"),
    ("DejaVuSans-Bold", "font-weight: bold; font-style: italic;")))
css = _FACES + """@page { size: A4; margin: 1.6cm 1.7cm; }
body { font-family: DejaVu, Helvetica; font-size: 8.9pt; line-height: 1.4; color: #111; }
h1 { font-size: 17pt; margin: 0 0 2pt 0; color: #0b2e59; }
h2 { font-size: 11.5pt; margin: 12pt 0 5pt 0; color: #0b2e59; border-bottom: 0.6pt solid #9aa; padding-bottom: 1pt; }
p { margin: 6pt 0 5pt 0; } li { margin: 0 0 3pt 0; } ul, ol { margin: 4pt 0 6pt 0; }
table { width: 100%; margin: 6pt 0 8pt 0; }
strong, b, em, i, th, td, li, p, h2 { font-family: DejaVu; }
th, td { border: 0.5pt solid #889; padding: 4pt 5pt 3pt 5pt; font-size: 8.2pt; line-height: 1.35; vertical-align: top; }
th { background-color: #e6ecf3; }
code { font-family: Courier; font-size: 8.2pt; }
.big { font-size: 15pt; font-weight: bold; color: #1b8a8f; }
img { margin: 2pt 0 2pt 0; }
pre { font-family: Courier; font-size: 7.8pt; background-color: #f2f4f7; padding: 4pt; border: 0.5pt solid #ccd; }"""
doc = f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>{html}</body></html>"
with open(dst, "wb") as fh:
    r = pisa.CreatePDF(doc, dest=fh, encoding="utf-8")
print("errores:", r.err)
