"""Render the technical report Markdown as a self-contained HTML page.

    uv run --with markdown python docs/build_report_html.py \
        docs/converse-stt-2.0-technical-report.md docs/converse-stt-2.0-technical-report.html

Only dependency beyond the standard library is the `markdown` package; the output embeds its
own CSS (same palette as the leaderboard template) and loads nothing from the network.
"""

import html
import re
import sys
from pathlib import Path

import markdown

src = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])

# --- split off title block -------------------------------------------------
lines = src.splitlines()
title = lines[0].lstrip("# ").strip()
byline = lines[2].strip().strip("*")
# lede = first paragraph after byline
i = 4
lede_lines = []
while lines[i].strip():
    lede_lines.append(lines[i])
    i += 1
lede = " ".join(lede_lines)
# key-stats table
while not lines[i].startswith("|"):
    i += 1
stats = []
while i < len(lines) and lines[i].startswith("|"):
    cells = [c.strip() for c in lines[i].strip("|").split("|")]
    if cells and cells[0] and not set(cells[0]) <= set("-: "):
        stats.append((cells[0], cells[1]))
    i += 1
body_md = "\n".join(lines[i:])

# --- math block -> html ----------------------------------------------------
formula = (
    '<div class="formula">'
    '<span class="eq"><span class="lhs">WER</span> = <span class="frac"><span>S + D + I</span><span>H + S + D</span></span></span>'
    '<span class="eq"><span class="lhs">CER</span> = <span class="frac"><span>S<sub>c</sub> + D<sub>c</sub> + I<sub>c</sub></span><span>H<sub>c</sub> + S<sub>c</sub> + D<sub>c</sub></span></span></span>'
    "</div>"
)
body_md = re.sub(r"\\\[.*?\\\]", formula, body_md, flags=re.S)

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
    extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
)
body_html = md.convert(body_md)
toc_html = md.toc

# wrap tables for horizontal scroll
body_html = body_html.replace("<table>", '<div class="tablewrap"><table>').replace("</table>", "</table></div>")

stats_html = "".join(
    f'<div class="stat"><strong>{html.escape(v)}</strong><span>{html.escape(k)}</span></div>' for k, v in stats
)

page = f"""<!DOCTYPE html>
<!-- Converse-STT 2.0 technical report. Generated from docs/converse-stt-2.0-technical-report.md. -->
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)} · Ocular AI</title>
<style>
  :root {{
    --bg: #fafaf8; --fg: #1c1b19; --muted: #6f6a63; --line: #e4e0d8; --soft: #f1eee8; --mark: #8a3d2a; --fill: #e7b5a4;
    color-scheme: light;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; }}
  body {{ background: var(--bg); color: var(--fg); font-family: ui-sans-serif, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; font-size: 16px; line-height: 1.55; }}
  a {{ color: inherit; text-decoration-color: var(--line); text-underline-offset: 3px; }}
  a:hover {{ text-decoration-color: var(--mark); }}
  .wrap {{ width: min(1180px, calc(100% - 40px)); margin: 0 auto; }}
  header {{ padding: 72px 0 28px; text-align: center; }}
  .eyebrow {{ margin: 0 0 14px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); }}
  h1 {{ margin: 0 auto; max-width: 900px; font-size: clamp(32px, 4.6vw, 54px); font-weight: 500; letter-spacing: -0.03em; line-height: 1.06; }}
  .byline {{ margin: 18px 0 0; color: var(--muted); font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 13px; }}
  .lede {{ max-width: 760px; margin: 22px auto 0; color: var(--muted); font-size: 18px; text-align: left; }}
  .stats {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px 24px; margin: 44px auto 0; max-width: 980px; border-block: 1px solid var(--line); padding: 22px 0; text-align: left; }}
  .stat strong {{ display: block; font-size: 22px; font-weight: 500; letter-spacing: -0.02em; font-variant-numeric: tabular-nums; line-height: 1.2; }}
  .stat span {{ color: var(--muted); font-size: 13px; }}
  main {{ padding: 24px 0 96px; }}
  .layout {{ display: grid; grid-template-columns: 240px minmax(0, 1fr); gap: 56px; align-items: start; }}
  nav.toc {{ position: sticky; top: 24px; font-size: 13px; max-height: calc(100vh - 48px); overflow: auto; padding-right: 8px; }}
  nav.toc .toctitle {{ margin: 0 0 10px; font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 11px; letter-spacing: 0.14em; text-transform: uppercase; color: var(--muted); }}
  nav.toc ul {{ list-style: none; margin: 0; padding: 0; }}
  nav.toc > ul > li {{ margin: 0 0 8px; }}
  nav.toc ul ul {{ margin: 4px 0 0 12px; border-left: 1px solid var(--line); padding-left: 10px; }}
  nav.toc ul ul li {{ margin: 0 0 4px; color: var(--muted); }}
  nav.toc a {{ text-decoration: none; }}
  nav.toc a:hover {{ color: var(--mark); }}
  article {{ min-width: 0; max-width: 860px; }}
  article h2 {{ margin: 64px 0 18px; font-size: 30px; font-weight: 500; letter-spacing: -0.03em; line-height: 1.1; padding-top: 18px; border-top: 1px solid var(--line); }}
  article h2:first-child {{ margin-top: 0; border-top: 0; padding-top: 0; }}
  article h3 {{ margin: 36px 0 10px; font-size: 19px; font-weight: 560; letter-spacing: -0.01em; }}
  article p {{ margin: 0 0 14px; }}
  article ul, article ol {{ margin: 0 0 16px; padding-left: 22px; }}
  article li {{ margin: 0 0 6px; }}
  article li > p {{ margin: 0 0 6px; }}
  article hr {{ display: none; }}
  article strong {{ font-weight: 600; }}
  article code {{ font-family: ui-monospace, "SF Mono", Menlo, monospace; font-size: 0.88em; background: var(--soft); padding: 1px 5px; border-radius: 5px; }}
  article pre {{ margin: 0 0 18px; padding: 14px 16px; background: var(--soft); border-radius: 12px; overflow: auto; font-size: 13px; line-height: 1.5; }}
  article pre code {{ background: transparent; padding: 0; font-size: inherit; }}
  article blockquote {{ margin: 0 0 16px; padding: 10px 18px; border-left: 3px solid var(--fill); color: #3a3733; background: #f6f3ee; border-radius: 0 10px 10px 0; }}
  article blockquote p {{ margin: 0; }}
  .tablewrap {{ margin: 0 0 20px; overflow-x: auto; border: 1px solid var(--line); border-radius: 12px; background: #fff; }}
  article table {{ border-collapse: collapse; width: 100%; font-size: 13.5px; font-variant-numeric: tabular-nums; }}
  article th, article td {{ padding: 8px 10px; text-align: left; vertical-align: top; border-top: 1px solid var(--line); }}
  article thead th {{ border-top: 0; background: var(--soft); font-weight: 560; font-size: 12px; letter-spacing: 0.03em; color: var(--muted); position: sticky; top: 0; white-space: nowrap; }}
  article tbody tr:hover td {{ background: #faf7f2; }}
  article td code, article th code {{ white-space: nowrap; }}
  .formula {{ display: flex; gap: 48px; flex-wrap: wrap; justify-content: center; margin: 10px 0 20px; padding: 18px; background: var(--soft); border-radius: 12px; font-size: 17px; }}
  .formula .eq {{ display: inline-flex; align-items: center; gap: 10px; }}
  .formula .lhs {{ font-variant: small-caps; letter-spacing: 0.04em; }}
  .frac {{ display: inline-flex; flex-direction: column; align-items: center; vertical-align: middle; font-size: 15px; }}
  .frac span:first-child {{ border-bottom: 1px solid var(--fg); padding: 0 6px 2px; }}
  .frac span:last-child {{ padding: 2px 6px 0; }}
  footer {{ border-top: 1px solid var(--line); padding: 28px 0 60px; color: var(--muted); font-size: 13px; }}
  @media (max-width: 960px) {{
    .layout {{ grid-template-columns: 1fr; gap: 24px; }}
    nav.toc {{ position: static; max-height: none; border-bottom: 1px solid var(--line); padding-bottom: 16px; }}
    .stats {{ grid-template-columns: repeat(2, 1fr); }}
    header {{ padding-top: 40px; }}
  }}
  @media print {{
    nav.toc {{ display: none; }}
    .layout {{ grid-template-columns: 1fr; }}
    article h2 {{ break-before: page; }}
    article h2:first-child {{ break-before: auto; }}
    .tablewrap {{ border: 0; }}
  }}
</style>
</head>
<body>
<header>
  <div class="wrap">
    <p class="eyebrow">Ocular AI Converse · Technical report</p>
    <h1>{html.escape(title)}</h1>
    <p class="byline">{html.escape(byline)}</p>
    <p class="lede">{html.escape(lede)}</p>
    <div class="stats">{stats_html}</div>
  </div>
</header>
<main>
  <div class="wrap layout">
    <nav class="toc" aria-label="Contents">
      <p class="toctitle">Contents</p>
      {toc_html}
    </nav>
    <article>
{body_html}
    </article>
  </div>
</main>
<footer>
  <div class="wrap">Converse-STT 2.0 · Ocular AI · harness and report code under the MIT License · recordings CC BY 4.0 · research@useocular.com</div>
</footer>
</body>
</html>
"""
out.write_text(page, encoding="utf-8")
print(out, len(page))
