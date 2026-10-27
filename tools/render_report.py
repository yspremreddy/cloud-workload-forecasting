from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import mistune

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    parser = argparse.ArgumentParser(description="Render the Markdown report to a temporary review PDF.")
    parser.add_argument("--output", type=Path, required=True, help="PDF path, for example .venv/final_report_preview.pdf")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    markdown = (ROOT / "reports" / "final_report.md").read_text(encoding="utf-8")
    body = mistune.create_markdown(plugins=["table", "strikethrough"])(markdown)
    base_uri = ROOT.as_uri() + "/"
    document = f"""<!doctype html>
<html><head><meta charset="utf-8"><base href="{base_uri}"><style>
@page {{ size: Letter; margin: 0.62in 0.68in 0.65in; }}
* {{ box-sizing: border-box; }}
body {{ font: 10.2pt/1.27 Arial, Helvetica, sans-serif; color: #172536; }}
h1 {{ font-size: 21pt; line-height: 1.1; margin: 0 0 15pt; color: #123456; }}
h2 {{ font-size: 14pt; margin: 15pt 0 6pt; padding-bottom: 3pt; border-bottom: 1px solid #b9c7d5; color: #174b72; break-after: avoid; }}
h3 {{ font-size: 11.3pt; margin: 11pt 0 4pt; break-after: avoid; }}
p {{ margin: 4pt 0 7pt; orphans: 3; widows: 3; }}
table {{ border-collapse: collapse; width: 100%; font-size: 8.8pt; margin: 7pt 0 10pt; break-inside: avoid; }}
th, td {{ border: 1px solid #c4ced8; padding: 4px 6px; vertical-align: top; }}
th {{ background: #eaf1f7; text-align: left; }}
tr:nth-child(even) td {{ background: #f8fafc; }}
img {{ display:block; max-width: 100%; max-height: 4.7in; object-fit: contain; margin: 8pt auto; break-inside: avoid; }}
li {{ margin: 2pt 0; }}
code {{ font: 8.7pt Consolas, monospace; overflow-wrap: anywhere; }}
blockquote {{ border-left: 3px solid #a8bfd4; padding-left: 9pt; margin-left: 0; color:#334; }}
</style></head><body>{body}</body></html>"""
    candidates = [os.environ.get("CHROME_PATH"), shutil.which("chrome"), shutil.which("msedge")]
    if os.name == "nt":
        candidates.extend([
            r"C:\Program Files\Google\Chrome\Application\chrome.exe",
            r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        ])
    browser = next((Path(p) for p in candidates if p and Path(p).is_file()), None)
    if browser is None:
        raise SystemExit("Chrome/Edge not found. Set CHROME_PATH to its executable.")
    with tempfile.TemporaryDirectory(prefix="report-render-", dir=output.parent) as tmp:
        html_path = Path(tmp) / "report.html"
        html_path.write_text(document, encoding="utf-8")
        subprocess.run([
            str(browser), "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-pdf-header-footer", f"--print-to-pdf={output}", html_path.as_uri(),
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    print(f"Rendered {ROOT / 'reports' / 'final_report.md'} to {output}")

if __name__ == "__main__":
    main()
