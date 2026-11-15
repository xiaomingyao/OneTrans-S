#!/usr/bin/env python3
# coding=utf-8
"""迁移指导 md -> 自包含 HTML（图片 base64、表格横滚）"""
import base64
import html as _html
import os
import re
import sys

import markdown

# 代码高亮：Pygments 直接产出「内联样式」的 HTML，避免为每条样式规则再塞一份 CSS
from pygments import highlight as _pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, PythonLexer, TextLexer, get_lexer_by_name

_LEXERS = {"python": PythonLexer, "py": PythonLexer, "bash": BashLexer, "sh": BashLexer, "shell": BashLexer}
_FMT = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)


_ADD, _DEL = "#a6e22e", "#f92672"


def _diff_aware(code: str, lexer) -> str:
    """代码块里若多为 +/- 前缀（改前/改后对照），把标记单独着色，其余仍按语言高亮。"""
    lines = code.split("\n")
    body_lines = [l for l in lines if l.strip()]
    marks = [l for l in body_lines if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    if len(marks) < 2 or len(marks) * 4 < len(body_lines):
        return _pyg_highlight(code, lexer, _FMT)

    out = []
    for line in lines:
        if line.strip() and line[:1] in "+-" and not line.startswith(("+++", "---")):
            color = _ADD if line[0] == "+" else _DEL
            out.append(f'<span style="color:{color};font-weight:700">{line[0]}</span>'
                       + _pyg_highlight(line[1:], lexer, _FMT).rstrip("\n"))
        elif line.strip():
            out.append(_pyg_highlight(line, lexer, _FMT).rstrip("\n"))
        else:
            out.append("")
    return "\n".join(out)


def highlight_code(html_body: str) -> str:
    """把 <pre><code class="language-xxx">…</code></pre> 替换成带语法高亮的块。"""

    def _one(m: "re.Match") -> str:
        lang = (m.group(1) or "").strip().lower()
        code = _html.unescape(m.group(2))
        lexer_cls = _LEXERS.get(lang)
        if lexer_cls is None:
            if lang:
                try:
                    lexer = get_lexer_by_name(lang)
                except Exception:
                    lexer = TextLexer(stripnl=False)
            else:
                lexer = TextLexer(stripnl=False)
        else:
            lexer = lexer_cls(stripnl=False)
        return f'<pre class="hl">{_diff_aware(code, lexer)}</pre>'

    return re.sub(r'<pre><code class="language-([a-z0-9+#-]*)">(.*?)</code></pre>',
                  _one, html_body, flags=re.S)

ROOT = "/root/bytedance-onetrans"
SRC = sys.argv[1] if len(sys.argv) > 1 else "docs/MIGRATION_GUIDE_bytedance_onetrans_v1_20261114.md"
OUT = sys.argv[2] if len(sys.argv) > 2 else "docs/MIGRATION_GUIDE_bytedance_onetrans_v1_20261114.html"
TITLE = sys.argv[3] if len(sys.argv) > 3 else "OneTrans 迁移昇腾 950 指导 · 2026-11-14"

CSS = """
:root{--ink:#1a1a2e;--sub:#5a6270;--line:#e3e6ea;--brand:#C00000;--bg:#fbfbfc;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.78;
 font-family:"Microsoft YaHei","Noto Sans CJK SC","PingFang SC",Arial,sans-serif;-webkit-font-smoothing:antialiased;}
.wrap{max-width:1120px;margin:0 auto;padding:44px 40px 90px;background:#fff;box-shadow:0 1px 24px rgba(0,0,0,.06);}
h1{font-size:26px;font-weight:700;color:var(--brand);margin:0 0 8px;}
h2{font-size:20px;font-weight:700;margin:40px 0 14px;padding-bottom:9px;border-bottom:2px solid var(--brand);}
h3{font-size:16.5px;font-weight:700;margin:26px 0 10px;}
p{margin:10px 0;}
code{font-family:"JetBrains Mono",Consolas,Monaco,monospace;font-size:12.8px;background:#f2f4f6;
 padding:2px 6px;border-radius:4px;color:#b02a37;}
pre{background:#1e2633;color:#e6edf3;padding:15px 18px;border-radius:7px;overflow-x:auto;font-size:12.8px;line-height:1.6;}
pre code{background:none;color:inherit;padding:0;}
pre.hl{background:#272822;color:#f8f8f2;padding:15px 18px;border-radius:7px;overflow-x:auto;
 font-size:12.8px;line-height:1.62;border-left:3px solid var(--brand);}
pre.hl .c,.pre.hl .c1{font-style:italic;}
.tw{overflow-x:auto;max-width:100%;margin:16px 0;}
table{border-collapse:collapse;width:max-content;min-width:100%;font-size:13.4px;background:#fff;
 box-shadow:0 1px 4px rgba(0,0,0,.05);}
th{background:#C00000;color:#fff;font-weight:600;padding:9px 11px;text-align:left;border:1px solid #a30000;white-space:nowrap;}
td{padding:9px 11px;border:1px solid var(--line);vertical-align:top;}
tbody tr:nth-child(even){background:#f8fafb;}
tbody tr:hover{background:#eef4fa;}
.fig{margin:22px 0 26px;text-align:center;}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:7px;box-shadow:0 2px 10px rgba(0,0,0,.07);}
@media(max-width:760px){.wrap{padding:22px 16px 60px}h1{font-size:21px}h2{font-size:18px}table{font-size:12px}th,td{padding:7px 8px}}
"""


def main():
    md = open(os.path.join(ROOT, SRC), encoding="utf-8").read()
    imgs = {}

    def repl(m):
        rel = m.group(2)
        key = "@@FIG_" + os.path.basename(rel).replace(".", "_") + "@@"
        imgs[key] = rel
        return "\n\n" + key + "\n\n"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", repl, md)
    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])
    body = highlight_code(body)   # 带 ```python / ```bash 的块上语法高亮
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")

    for key, rel in imgs.items():
        path = os.path.join(ROOT, rel)
        b64 = base64.b64encode(open(path, "rb").read()).decode()
        body = body.replace(f"<p>{key}</p>", f'<div class="fig"><img src="data:image/png;base64,{b64}" alt="{os.path.basename(rel)}"></div>')
        body = body.replace(key, "")

    doc = (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'
           f'\n<style>{CSS}</style>\n</head>\n<body>\n<div class="wrap">\n{body}\n</div>\n</body></html>')
    open(os.path.join(ROOT, OUT), "w", encoding="utf-8").write(doc)
    print(f"OK -> {OUT} ({os.path.getsize(os.path.join(ROOT, OUT))/1024:.0f} KB)  表 {body.count('<table>')} 图 {len(imgs)}")


if __name__ == "__main__":
    main()
