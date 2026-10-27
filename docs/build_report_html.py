"""Render the technical report Markdown as a self-contained HTML page.

    uv run --with markdown python docs/build_report_html.py \
        docs/converse-stt-2.0-technical-report.md docs/converse-stt-2.0-technical-report.html

Only dependency beyond the standard library is the `markdown` package. The output embeds its own
CSS, styled after the Converse report pages on useocular.com and sharing the leaderboard template's
design tokens; the only network request is the optional JetBrains Mono web font.
"""

import html
import re
import sys
from pathlib import Path

import markdown

src = Path(sys.argv[1]).read_text(encoding="utf-8")
out = Path(sys.argv[2])

HARNESS_HREF = "https://github.com/OcularEngineering/converse-stt"
LEADERBOARD_HREF = "../reports/converse-stt.html"
DATASET_HREF = (
    "https://huggingface.co/datasets/OcularAIInc/"
    "AMERICAN-ENGLISH-TRANSCRIBED-HIFI-FULL-DUPLEX-TWO-SPEAKER-CONVERSATIONAL-DATASET-SAMPLE"
)
HEAT_AXIS = 12.0  # per-track WER at which the heat shade saturates (same as the leaderboard page)

ICON_COPY = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true"><rect x="9" y="9" width="11" height="11" rx="2"/>'
    '<path d="M5 15V5a2 2 0 0 1 2-2h10"/></svg>'
)
ICON_GH = (
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 .5C5.65.5.5 5.65.5 12c0 5.08 3.29 '
    "9.39 7.86 10.91.58.1.79-.25.79-.56v-2.17c-3.2.7-3.87-1.36-3.87-1.36-.52-1.33-1.28-1.68-1.28-1.68-1.04-.71.08-.7"
    ".08-.7 1.15.08 1.76 1.19 1.76 1.19 1.03 1.76 2.7 1.25 3.36.96.1-.75.4-1.25.73-1.54-2.55-.29-5.24-1.28-5.24-5.69 "
    "0-1.26.45-2.28 1.19-3.09-.12-.29-.52-1.46.11-3.04 0 0 .97-.31 3.17 1.18a11 11 0 0 1 5.77 0c2.2-1.49 3.16-1.18 "
    "3.16-1.18.63 1.58.24 2.75.12 3.04.74.81 1.19 1.83 1.19 3.09 0 4.42-2.7 5.39-5.26 5.68.41.36.78 1.06.78 2.14v3.17c0 "
    '.31.21.67.8.56A11.52 11.52 0 0 0 23.5 12C23.5 5.65 18.35.5 12 .5Z"/></svg>'
)
ICON_HF = (
    '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M12 2.5a9.5 9.5 0 1 0 0 19 9.5 9.5 0 0 '
    "0 0-19Zm-3.4 7.1a1.1 1.1 0 1 1 0 2.2 1.1 1.1 0 0 1 0-2.2Zm6.8 0a1.1 1.1 0 1 1 0 2.2 1.1 1.1 0 0 1 0-2.2ZM12 "
    '17.6c-2.6 0-4.5-1.7-4.5-3.3 0-.5.4-.8.9-.6 1 .5 2.3.8 3.6.8s2.6-.3 3.6-.8c.5-.2.9.1.9.6 0 1.6-1.9 3.3-4.5 3.3Z"/></svg>'
)


def inline(md_text: str) -> str:
    """Render a one-line Markdown fragment without the surrounding <p>."""
    rendered = markdown.markdown(md_text).strip()
    return re.sub(r"^<p>|</p>$", "", rendered)


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
    '<figure class="formula">'
    '<p class="formula-eyebrow">Word and character error rate</p>'
    '<div class="formula-math">'
    '<span class="eq"><span class="lhs">WER</span><span class="rel">=</span>'
    '<span class="frac"><span class="top"><i>S</i> + <i>D</i> + <i>I</i></span>'
    '<span class="bottom"><i>H</i> + <i>S</i> + <i>D</i></span></span></span>'
    '<span class="eq"><span class="lhs">CER</span><span class="rel">=</span>'
    '<span class="frac"><span class="top"><i>S<sub>c</sub></i> + <i>D<sub>c</sub></i> + <i>I<sub>c</sub></i></span>'
    '<span class="bottom"><i>H<sub>c</sub></i> + <i>S<sub>c</sub></i> + <i>D<sub>c</sub></i></span></span></span>'
    "</div>"
    '<dl class="formula-terms">'
    "<dt><i>H</i></dt><dd>Reference words the hypothesis matched</dd>"
    "<dt><i>S</i></dt><dd>Reference words replaced by a different word</dd>"
    "<dt><i>D</i></dt><dd>Reference words missing from the hypothesis</dd>"
    "<dt><i>I</i></dt><dd>Hypothesis words with no reference word</dd>"
    "<dt><i>H</i> + <i>S</i> + <i>D</i></dt><dd>The reference length</dd>"
    "</dl>"
    "</figure>"
)
body_md = re.sub(r"\\\[.*?\\\]", formula, body_md, flags=re.S)

md = markdown.Markdown(
    extensions=["tables", "fenced_code", "toc", "attr_list", "md_in_html"],
    extension_configs={"toc": {"toc_depth": "2-3", "permalink": False}},
)
body_html = md.convert(body_md)
toc_html = re.sub(r'^<div class="toc">\s*|\s*</div>$', "", md.toc.strip())
toc_tokens = md.toc_tokens


# --- tables: numeric columns, sticky name column, heat shading --------------
TAG_RE = re.compile(r"<[^>]+>")
NUMERIC_RE = re.compile(r"^[\s−\-+≈<>≤≥$]*\d")
DASH_CELLS = {"", "—", "–", "-", "n/a"}
TRACK_HEADER_RE = re.compile(r"/ch[12]$")


def plain(cell_html: str) -> str:
    return html.unescape(TAG_RE.sub("", cell_html)).strip()


def is_numeric(text: str) -> bool:
    return text in DASH_CELLS or bool(NUMERIC_RE.match(text))


def heat_cell(cell_html: str) -> str:
    text = plain(cell_html)
    match = re.match(r"^\d+(?:\.\d+)?", text)
    if not match:
        return cell_html
    value = float(match.group())
    if value >= HEAT_AXIS:
        return f'<span class="heat hot">{cell_html}</span>'
    alpha = 0.08 + min(1.0, value / HEAT_AXIS) * 0.92
    return f'<span class="heat" style="background:rgba(232,174,145,{alpha:.3f})">{cell_html}</span>'


def render_table(table_html: str, eyebrow: str) -> str:
    headers = re.findall(r"<th[^>]*>(.*?)</th>", table_html, flags=re.S)
    body_rows = [
        re.findall(r"<td[^>]*>(.*?)</td>", row, flags=re.S)
        for row in re.findall(r"<tr>(.*?)</tr>", table_html, flags=re.S)[1:]
    ]
    ncol = len(headers)
    columns = [[row[c] for row in body_rows if c < len(row)] for c in range(ncol)]
    numeric = []
    for c, cells in enumerate(columns):
        texts = [plain(cell) for cell in cells]
        filled = [t for t in texts if t not in DASH_CELLS]
        numeric.append(bool(filled) and all(is_numeric(t) for t in filled))
    heat_cols = {c for c, h in enumerate(headers) if TRACK_HEADER_RE.search(plain(h))}
    name_col = next((c for c in range(ncol) if not numeric[c]), 0)
    # the name column stays put while the rest scrolls; a leading rank column rides along with it
    sticky = ncol >= 5 and name_col <= 1

    def cls(c: int) -> str:
        parts = []
        if c == name_col:
            parts.append("name")
            if sticky:
                parts.append("sticky" if name_col == 0 else "sticky sticky-2")
        elif c == 0 and name_col == 1:
            parts.append("rank")
            if sticky:
                parts.append("sticky")
        elif numeric[c]:
            parts.append("num")
        if c in heat_cols:
            parts.append("heat-col")
        return f' class="{" ".join(parts)}"' if parts else ""

    head = "".join(f"<th{cls(c)}>{h}</th>" for c, h in enumerate(headers))
    rows = []
    for row in body_rows:
        cells = []
        for c, cell in enumerate(row):
            content = heat_cell(cell) if c in heat_cols else cell
            cells.append(f"<td{cls(c)}>{content}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    if heat_cols:
        min_width = f"{220 + 72 * len(heat_cols) + 88 * (ncol - 1 - len(heat_cols))}px"
    else:
        min_width = (
            "960px" if ncol >= 10 else "880px" if ncol >= 8 else "720px" if ncol >= 6 else "560px" if ncol >= 4 else "0"
        )
    table = f'<table style="min-width:{min_width}"><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table>'
    caption = f'<figcaption class="fig-eyebrow">{eyebrow}</figcaption>' if eyebrow else ""
    return (
        f'<figure class="breakout{" heatmap" if heat_cols else ""}">{caption}'
        f'<div class="scroll-frame"><div class="scroller">{table}</div></div></figure>'
    )


HEADING_RE = re.compile(r"<h([23])[^>]*>(.*?)</h\1>", flags=re.S)
parts = re.split(r"(<table>.*?</table>)", body_html, flags=re.S)
rendered_parts = []
current_heading = ""
for part in parts:
    if part.startswith("<table>"):
        rendered_parts.append(render_table(part, current_heading))
        continue
    headings = HEADING_RE.findall(part)
    if headings:
        current_heading = plain(headings[-1][1])
    rendered_parts.append(part)
body_html = "".join(rendered_parts)

# --- other rendering hooks --------------------------------------------------
# code blocks get the card treatment and a copy button
body_html = re.sub(
    r"<pre>(<code[^>]*>.*?</code>)</pre>",
    lambda m: (
        '<figure class="codeblock"><pre>' + m.group(1) + "</pre>"
        f'<button class="copy" type="button" aria-label="Copy code" title="Copy code">{ICON_COPY}</button></figure>'
    ),
    body_html,
    flags=re.S,
)
# a bold label paragraph directly before an excerpt moves into the quote, as on the live report
body_html = re.sub(
    r"<p>(<strong>.*?</strong>[^<\n]*?):?</p>\s*<blockquote>",
    r'<blockquote><p class="quote-label">\1</p>',
    body_html,
    flags=re.S,
)
# dagger footnotes under tables
body_html = re.sub(r"<p>(?=[†‡])", '<p class="footnote">', body_html)
# section rules are implied by the heading spacing
body_html = re.sub(r"\s*<hr />\s*", "\n", body_html)

# --- hero pieces -----------------------------------------------------------
byline_parts = [inline(part.strip()) for part in byline.split(" · ")]
byline_html = '<span class="sep" aria-hidden="true">·</span>'.join(
    f'<span class="{"lead" if n == 0 else ""}">{part}</span>' for n, part in enumerate(byline_parts)
)


def stat_tile(label: str, value: str) -> str:
    note = ""
    match = re.match(r"^(.*?)\s*\((.*)\)\s*$", value)
    if match:
        value, note = match.group(1), match.group(2)
    elif re.match(r"^[\d.,%\s/]+\s(?:of|per)\s", value):
        value, note = re.split(r"\s(?=(?:of|per)\s)", value, maxsplit=1)
    elif len(value) > 22 and ", " in value:
        value, note = value.split(", ", 1)
    size = " long" if len(value) > 14 else ""
    note_html = f"<small>{html.escape(note)}</small>" if note else ""
    return f'<div class="stat{size}"><span>{html.escape(label)}</span><strong>{html.escape(value)}</strong>{note_html}</div>'


stats_html = "".join(stat_tile(k, v) for k, v in stats)


def contents_nav(tokens: list[dict]) -> str:
    items = []
    for n, token in enumerate(tokens, 1):
        items.append(
            f'<li><a href="#{token["id"]}"><span class="n">{n:02d}</span><span>{html.escape(token["name"])}</span></a></li>'
        )
    return "".join(items)


contents_html = contents_nav(toc_tokens)

page = f"""<!DOCTYPE html>
<!-- Converse-STT 2.0 technical report. Generated from docs/converse-stt-2.0-technical-report.md by
     docs/build_report_html.py (Ocular AI's open-source Converse-STT harness, MIT License):
     https://github.com/OcularEngineering/converse-stt -->
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>{html.escape(title)} · Ocular AI</title>
<meta name="description" content="{html.escape(lede)}" />
<link rel="preconnect" href="https://fonts.googleapis.com" />
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin />
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet" />
<style>
  :root {{
    --background: #ffffff;
    --foreground: #1d1d17;
    --foreground-body: #1d1d17e6;
    --muted: #24241d0d;
    --muted-foreground: #1d1d1780;
    --border: #24241d1a;
    --card: #24241d08;
    --card-sticky: #f8f8f8; /* --card over white, opaque, for sticky table columns */
    --saturated: #8a3d2a;
    --sans: "OpenAI Sans", -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI", Helvetica, Arial, sans-serif;
    --mono: "JetBrains Mono", ui-monospace, "SF Mono", Menlo, monospace;
    --text-width: 680px;
    color-scheme: light;
  }}
  * {{ box-sizing: border-box; }}
  html {{ -webkit-text-size-adjust: 100%; scroll-behavior: smooth; }}
  html, body {{ margin: 0; }}
  body {{ background: var(--background); color: var(--foreground); font-family: var(--sans); font-size: 16px; line-height: 1.5; -webkit-font-smoothing: antialiased; }}
  button {{ font: inherit; color: inherit; cursor: pointer; }}
  a {{ color: inherit; text-decoration: none; }}
  p {{ margin: 0; }}
  h1, h2, h3, h4 {{ margin: 0; font-weight: 500; }}
  code {{ font-family: var(--mono); }}
  .sr-only {{ position: absolute; width: 1px; height: 1px; overflow: hidden; clip: rect(0 0 0 0); white-space: nowrap; }}
  .container {{ margin: 0 auto; width: 100%; max-width: 1440px; padding: 0 20px; }}
  @media (min-width: 768px) {{ .container {{ padding: 0 64px; }} }}
  @media (min-width: 1024px) {{ .container {{ padding: 0 80px; }} }}

  /* Shared pieces */
  .eyebrow {{ font-family: var(--mono); font-size: 12px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-foreground); }}
  .pill {{ display: inline-flex; align-items: center; justify-content: center; gap: 8px; height: 44px; padding: 0 20px; border-radius: 999px; border: 0; background: var(--muted); color: var(--foreground); font-size: 14px; white-space: nowrap; transition: background-color 0.15s ease, color 0.15s ease, transform 0.1s ease; }}
  .pill:hover {{ background: #24241d1a; }}
  .pill:active {{ transform: scale(0.98); }}
  .pill.primary {{ background: var(--foreground); color: var(--background); }}
  .pill.primary:hover {{ background: #1d1d17e6; }}
  .pill svg {{ width: 16px; height: 16px; flex: none; }}

  /* Hero */
  .hero {{ padding: 64px 0 32px; text-align: center; }}
  .hero .eyebrow {{ margin-bottom: 20px; }}
  .hero h1 {{ margin: 0 auto; max-width: 56rem; font-size: 2rem; line-height: 1.02; letter-spacing: -0.03em; text-wrap: balance; }}
  .byline {{ display: flex; flex-wrap: wrap; align-items: center; justify-content: center; gap: 4px 12px; margin-top: 20px; font-size: 14px; color: var(--muted-foreground); }}
  .byline .lead {{ color: var(--foreground); font-weight: 500; }}
  .byline .sep {{ color: #1d1d1766; }}
  .byline code {{ font-size: 13px; }}
  .hero .lede {{ margin: 24px auto 0; max-width: 46rem; font-size: 16px; line-height: 1.65; color: var(--muted-foreground); text-wrap: pretty; }}
  .hero .actions {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 12px; margin-top: 32px; }}
  .stats {{ display: grid; grid-template-columns: repeat(2, 1fr); gap: 28px 24px; margin: 56px auto 0; padding: 28px 0; border-top: 1px solid var(--border); border-bottom: 1px solid var(--border); text-align: left; }}
  .stat {{ min-width: 0; }}
  .stat span {{ display: block; margin-bottom: 12px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-foreground); }}
  .stat strong {{ display: block; font-size: 1.75rem; font-weight: 500; letter-spacing: -0.03em; line-height: 1.1; font-variant-numeric: tabular-nums; text-wrap: balance; }}
  .stat.long strong {{ font-size: 1.25rem; line-height: 1.25; padding-top: 0.25rem; }}
  .stat small {{ display: block; margin-top: 8px; font-size: 13px; line-height: 1.5; color: var(--muted-foreground); text-wrap: pretty; }}
  @media (min-width: 768px) {{
    .hero {{ padding: 96px 0 48px; }}
    .hero h1 {{ font-size: 3rem; }}
    .hero .lede {{ font-size: 17px; }}
    .stats {{ grid-template-columns: repeat(4, 1fr); gap: 36px 32px; margin-top: 72px; padding: 36px 0; }}
    .stat strong {{ font-size: 2.25rem; }}
    .stat.long strong {{ font-size: 1.5rem; }}
  }}
  @media (min-width: 1280px) {{
    .hero h1 {{ font-size: 3.5rem; }}
  }}

  /* Layout: side contents + article */
  main {{ padding: 32px 0 96px; }}
  .layout {{ display: grid; grid-template-columns: minmax(0, 1fr); gap: 32px; align-items: start; }}
  @media (min-width: 1024px) {{
    main {{ padding-top: 48px; }}
    .layout {{ grid-template-columns: 220px minmax(0, 1fr); gap: 64px; }}
  }}
  .toc {{ font-size: 13px; }}
  .toc summary {{ list-style: none; cursor: pointer; }}
  .toc summary::-webkit-details-marker {{ display: none; }}
  .toc .toctitle {{ display: flex; align-items: center; justify-content: space-between; gap: 12px; margin: 0 0 12px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-foreground); }}
  .toc .toctitle svg {{ width: 14px; height: 14px; transition: transform 0.15s ease; }}
  .toc details[open] .toctitle svg {{ transform: rotate(180deg); }}
  .toc ul {{ list-style: none; margin: 0; padding: 0; }}
  .toc > details > ul > li, .toc > ul > li {{ margin: 0; }}
  .toc a {{ display: block; padding: 5px 0; color: var(--muted-foreground); line-height: 1.4; transition: color 0.15s ease; }}
  .toc a:hover {{ color: var(--foreground); }}
  .toc a.is-active {{ color: var(--foreground); font-weight: 500; }}
  .toc ul ul {{ margin: 0 0 4px 10px; padding-left: 12px; border-left: 1px solid var(--border); }}
  .toc ul ul a {{ padding: 3px 0; font-size: 12.5px; }}
  .toc details {{ border-bottom: 1px solid var(--border); padding-bottom: 12px; }}
  @media (min-width: 1024px) {{
    .toc {{ position: sticky; top: 24px; max-height: calc(100vh - 48px); overflow-y: auto; padding-right: 12px; scrollbar-width: thin; }}
    .toc details {{ border: 0; padding: 0; }}
    .toc .toctitle svg {{ opacity: 0.6; }}
  }}

  /* Article */
  .article {{ min-width: 0; }}
  .article > * {{ max-width: var(--text-width); }}
  .article > .breakout, .article > .codeblock, .article > .formula {{ max-width: none; }}
  .article h2 {{ margin: 64px 0 16px; font-size: 1.5rem; line-height: 1.2; letter-spacing: -0.02em; scroll-margin-top: 24px; }}
  .article > h2:first-child {{ margin-top: 0; }}
  .article h3 {{ margin: 40px 0 12px; font-size: 1.2rem; line-height: 1.3; letter-spacing: -0.015em; scroll-margin-top: 24px; }}
  .article p {{ margin: 0 0 20px; font-size: 17px; line-height: 1.7; color: var(--foreground-body); text-wrap: pretty; }}
  .article a {{ text-decoration: underline; text-underline-offset: 2px; text-decoration-color: #1d1d1766; }}
  .article a:hover {{ text-decoration-color: currentColor; }}
  .article ul, .article ol {{ margin: 0 0 20px; padding-left: 24px; font-size: 17px; line-height: 1.7; color: var(--foreground-body); }}
  .article li {{ margin: 0 0 10px; padding-left: 4px; }}
  .article li > p {{ margin: 0 0 10px; }}
  .article li::marker {{ color: var(--muted-foreground); font-variant-numeric: tabular-nums; }}
  .article strong {{ font-weight: 600; color: var(--foreground); }}
  .article code {{ font-size: 0.86em; background: var(--muted); padding: 2px 6px; border-radius: 4px; color: var(--foreground); overflow-wrap: anywhere; }}
  .article p.footnote {{ font-size: 13px; line-height: 1.6; color: var(--muted-foreground); margin-top: -8px; }}

  /* Excerpts and callouts */
  .article blockquote {{ margin: 0 0 24px; padding: 2px 0 2px 20px; border-left: 2px solid #1d1d1733; }}
  .article blockquote p {{ font-size: 16px; line-height: 1.75; margin: 0 0 12px; }}
  .article blockquote p:last-child {{ margin-bottom: 0; }}
  .article blockquote .quote-label {{ font-size: 14px; color: var(--muted-foreground); }}
  .article blockquote .quote-label strong {{ font-weight: 500; color: var(--muted-foreground); }}

  /* Code */
  .codeblock {{ position: relative; margin: 32px 0; }}
  .codeblock pre {{ margin: 0; padding: 24px 64px 24px 24px; border-radius: 16px; background: var(--card); font-family: var(--mono); font-size: 13px; line-height: 1.65; color: var(--foreground); overflow-x: auto; }}
  .codeblock pre code {{ background: transparent; padding: 0; font-size: inherit; border-radius: 0; overflow-wrap: normal; }}
  .codeblock pre code.language-bibtex {{ white-space: pre-wrap; overflow-wrap: anywhere; }}
  .copy {{ position: absolute; top: 16px; right: 16px; display: inline-flex; align-items: center; justify-content: center; width: 36px; height: 36px; border-radius: 999px; border: 1px solid #1d1d171a; background: #ffffffcc; color: var(--muted-foreground); transition: background-color 0.15s ease, color 0.15s ease; }}
  .copy:hover {{ background: var(--background); color: var(--foreground); }}
  .copy svg {{ width: 16px; height: 16px; }}
  .copy.copied {{ color: #2f7a3e; }}

  /* Tables */
  .breakout {{ margin: 32px 0 36px; padding: 20px; border-radius: 16px; background: var(--card); }}
  .fig-eyebrow {{ margin: 0 0 16px; font-family: var(--mono); font-size: 11px; letter-spacing: 0.1em; text-transform: uppercase; color: var(--muted-foreground); }}
  .scroll-frame {{ position: relative; }}
  .scroll-frame::after {{ content: ""; position: absolute; top: 0; right: 0; bottom: 0; width: 40px; border-radius: 0 8px 8px 0; background: linear-gradient(to right, rgba(248,248,248,0), #f8f8f8 85%); opacity: 0; pointer-events: none; transition: opacity 0.2s ease; }}
  .scroll-frame.can-scroll:not(.at-end)::after {{ opacity: 1; }}
  .scroller {{ overflow-x: auto; -webkit-overflow-scrolling: touch; scrollbar-width: thin; }}
  .breakout table {{ width: 100%; border-collapse: separate; border-spacing: 0; }}
  .breakout th {{ padding: 0 20px 14px 0; text-align: left; vertical-align: bottom; font-size: 13px; font-weight: 400; line-height: 1.3; color: var(--muted-foreground); border-bottom: 1px solid #24241d26; }}
  .breakout th:last-child, .breakout td:last-child {{ padding-right: 0; }}
  .breakout td {{ padding: 11px 20px 11px 0; vertical-align: top; font-size: 13px; line-height: 1.55; color: var(--muted-foreground); border-bottom: 1px solid #24241d14; }}
  .breakout tbody tr:last-child td {{ border-bottom: 0; }}
  .breakout th.name {{ color: var(--foreground); font-weight: 600; font-size: 14px; letter-spacing: -0.01em; }}
  .breakout td.name {{ color: var(--foreground); font-weight: 500; letter-spacing: -0.01em; }}
  .breakout .sticky {{ position: sticky; left: 0; z-index: 2; background: var(--card-sticky); }}
  .breakout .rank {{ width: 24px; padding-right: 16px; font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; color: var(--muted-foreground); }}
  .breakout .sticky-2 {{ left: 40px; }}
  .breakout .num {{ text-align: right; font-family: var(--mono); font-variant-numeric: tabular-nums; font-size: 12.5px; white-space: nowrap; }}
  .breakout td code {{ font-size: 12px; padding: 1px 5px; }}
  .breakout td strong {{ font-weight: 600; }}
  .breakout tbody tr:hover td {{ background-color: #24241d08; }}
  .breakout tbody tr:hover td.sticky {{ background-color: #f3f3f0; }}
  .heatmap th.heat-col {{ text-align: center; font-family: var(--mono); font-size: 11px; }}
  .heatmap td.heat-col {{ padding: 3px 3px; text-align: center; }}
  .heatmap .heat {{ display: block; padding: 7px 6px; border-radius: 4px; font-family: var(--mono); font-size: 12px; font-variant-numeric: tabular-nums; color: var(--foreground); }}
  .heatmap .heat.hot {{ background: var(--saturated); color: #f6ece7; }}
  .heatmap td {{ border-bottom: 0; padding-top: 3px; padding-bottom: 3px; }}
  .heatmap td.name {{ padding-top: 8px; }}
  .heatmap tbody tr:hover td {{ background-color: transparent; }}
  @media (min-width: 768px) {{ .breakout {{ padding: 32px 40px 36px; }} .fig-eyebrow {{ margin-bottom: 24px; }} }}

  /* Formula */
  .formula {{ margin: 32px 0; padding: 36px 24px; border-radius: 16px; background: var(--card); text-align: center; }}
  .formula-eyebrow {{ margin: 0 0 24px !important; font-size: 14px !important; font-weight: 600; letter-spacing: -0.005em; color: var(--foreground) !important; }}
  .formula-math {{ display: flex; flex-wrap: wrap; justify-content: center; gap: 24px 56px; font-family: "KaTeX_Main", "Latin Modern Roman", "STIX Two Math", "Cambria Math", "Times New Roman", serif; font-size: 1.6rem; color: var(--foreground); overflow-x: auto; }}
  .formula .eq {{ display: inline-flex; align-items: center; gap: 12px; }}
  .formula .lhs {{ font-variant: normal; }}
  .formula .rel {{ padding: 0 2px; }}
  .formula i {{ font-style: italic; font-family: "KaTeX_Math", "Latin Modern Math", "STIX Two Math", "Cambria Math", "Times New Roman", serif; }}
  .formula sub {{ font-size: 0.6em; line-height: 0; }}
  .frac {{ display: inline-flex; flex-direction: column; align-items: center; font-size: 0.85em; line-height: 1.3; }}
  .frac .top {{ padding: 0 8px 3px; border-bottom: 1px solid var(--foreground); }}
  .frac .bottom {{ padding: 3px 8px 0; }}
  .formula-terms {{ display: grid; grid-template-columns: auto 1fr; gap: 8px 16px; max-width: 28rem; margin: 32px auto 0; padding-top: 24px; border-top: 1px solid var(--border); text-align: left; font-size: 14px; }}
  .formula-terms dt {{ text-align: right; font-family: "KaTeX_Main", "Latin Modern Roman", "STIX Two Math", "Cambria Math", "Times New Roman", serif; font-size: 1.05rem; color: var(--foreground); }}
  .formula-terms dd {{ margin: 0; color: var(--muted-foreground); }}
  @media (min-width: 768px) {{ .formula {{ padding: 40px 32px; }} .formula-math {{ font-size: 1.9rem; }} }}

  /* In this report */
  .contents {{ margin: 64px 0 0; padding: 24px; border: 1px solid var(--border); border-radius: 16px; background: #24241d05; }}
  .contents h2 {{ margin: 0 0 20px; font-size: 1.125rem; letter-spacing: -0.01em; }}
  .contents ol {{ display: grid; gap: 4px; margin: 0; padding: 0; list-style: none; }}
  .contents a {{ display: flex; gap: 12px; align-items: flex-start; padding: 10px 8px; border-radius: 8px; font-size: 14px; color: #1d1d17cc; text-decoration: none; transition: background-color 0.15s ease, color 0.15s ease; }}
  .contents a:hover {{ background: var(--muted); color: var(--foreground); }}
  .contents .n {{ padding-top: 2px; font-family: var(--mono); font-size: 11px; color: var(--muted-foreground); font-variant-numeric: tabular-nums; }}
  @media (min-width: 640px) {{ .contents {{ padding: 32px; }} .contents ol {{ grid-template-columns: repeat(2, 1fr); }} }}

  footer {{ margin-top: 48px; padding: 32px 0 48px; border-top: 1px solid var(--border); font-size: 13px; line-height: 1.6; color: var(--muted-foreground); }}
  footer .container {{ display: flex; flex-wrap: wrap; justify-content: space-between; gap: 12px 32px; }}
  footer a {{ text-decoration: underline; text-underline-offset: 2px; }}

  @media print {{
    .toc, .hero .actions, .copy, .contents {{ display: none; }}
    .layout {{ grid-template-columns: 1fr; }}
    .article h2 {{ break-before: page; }}
    .article > h2:first-child {{ break-before: auto; }}
    .breakout, .codeblock pre, .formula {{ background: transparent; border: 1px solid var(--border); }}
    .breakout .sticky {{ position: static; }}
  }}
</style>
</head>
<body>
<header class="hero">
  <div class="container">
    <p class="eyebrow">Ocular Converse</p>
    <h1>{html.escape(title)}</h1>
    <p class="byline">{byline_html}</p>
    <p class="lede">{html.escape(lede)}</p>
    <div class="actions">
      <a class="pill primary" href="{LEADERBOARD_HREF}">Leaderboard</a>
      <a class="pill" href="{HARNESS_HREF}" target="_blank" rel="noopener noreferrer">{ICON_GH}Harness</a>
      <a class="pill" href="{DATASET_HREF}" target="_blank" rel="noopener noreferrer">{ICON_HF}Dataset</a>
    </div>
    <div class="stats">{stats_html}</div>
  </div>
</header>
<main>
  <div class="container layout">
    <nav class="toc" aria-label="Contents">
      <details open>
        <summary><p class="toctitle">Contents <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></p></summary>
        {toc_html}
      </details>
    </nav>
    <article class="article">
{body_html}
      <nav class="contents" aria-label="Article contents">
        <h2>In this report</h2>
        <ol>{contents_html}</ol>
      </nav>
    </article>
  </div>
</main>
<footer>
  <div class="container">
    <p>Converse-STT is a benchmark by <a href="https://www.useocular.com">Ocular AI</a>. Every number in this report was produced by the open-source <a href="{HARNESS_HREF}">ocular-evals</a> harness (MIT License); recordings and references CC BY 4.0.</p>
    <p><a href="{LEADERBOARD_HREF}">Leaderboard</a> · <a href="mailto:research@useocular.com">research@useocular.com</a></p>
  </div>
</footer>
<script>
  // Copy buttons on code blocks.
  document.querySelectorAll(".codeblock .copy").forEach((button) => {{
    const original = button.innerHTML;
    button.addEventListener("click", async () => {{
      const text = button.parentElement.querySelector("pre").textContent;
      try {{
        await navigator.clipboard.writeText(text);
        button.classList.add("copied");
        button.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true"><path d="M5 12l5 5L20 7"/></svg>';
        setTimeout(() => {{ button.classList.remove("copied"); button.innerHTML = original; }}, 2000);
      }} catch (error) {{ /* clipboard unavailable */ }}
    }});
  }});

  // Fade the right edge of tables that are wider than their card until they are scrolled to the end.
  document.querySelectorAll(".scroll-frame").forEach((frame) => {{
    const scroller = frame.querySelector(".scroller");
    const update = () => {{
      frame.classList.toggle("can-scroll", scroller.scrollWidth > scroller.clientWidth + 1);
      frame.classList.toggle("at-end", scroller.scrollLeft + scroller.clientWidth >= scroller.scrollWidth - 1);
    }};
    scroller.addEventListener("scroll", update, {{ passive: true }});
    window.addEventListener("resize", update);
    update();
  }});

  // On narrow screens the contents list sits above the article, so it starts collapsed.
  const tocDetails = document.querySelector(".toc details");
  const narrow = window.matchMedia("(max-width: 1023px)");
  let wasNarrow = null;
  const syncToc = () => {{
    if (narrow.matches === wasNarrow) return;
    wasNarrow = narrow.matches;
    tocDetails.open = !narrow.matches;
  }};
  syncToc();
  narrow.addEventListener("change", syncToc);
  window.addEventListener("resize", syncToc);

  // Highlight the current section in the side contents.
  const tocLinks = Array.from(document.querySelectorAll(".toc a"));
  const byId = new Map(tocLinks.map((link) => [decodeURIComponent(link.getAttribute("href").slice(1)), link]));
  const headings = Array.from(document.querySelectorAll(".article h2[id], .article h3[id]")).filter((h) => byId.has(h.id));
  let active = null;
  function updateActive() {{
    let current = null;
    for (const heading of headings) {{
      if (heading.getBoundingClientRect().top <= 96) current = heading;
      else break;
    }}
    if (current === active) return;
    if (active) byId.get(active.id).classList.remove("is-active");
    active = current;
    if (active) {{
      const link = byId.get(active.id);
      link.classList.add("is-active");
      const nav = link.closest(".toc");
      if (nav && getComputedStyle(nav).position === "sticky") {{
        const top = link.offsetTop - nav.clientHeight / 2;
        if (Math.abs(nav.scrollTop - top) > nav.clientHeight / 3) nav.scrollTo({{ top, behavior: "smooth" }});
      }}
    }}
  }}
  window.addEventListener("scroll", updateActive, {{ passive: true }});
  window.addEventListener("resize", updateActive);
  updateActive();
</script>
</body>
</html>
"""
out.write_text(page, encoding="utf-8")
print(out, len(page))
