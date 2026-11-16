#!/usr/bin/env python3
# coding=utf-8
"""把本仓 docs/*.md 报告转成自包含 HTML（支持标题/表格/列表/代码块/粗体/行内代码）。

用法: python3 tools/md2html.py <in.md> <out.html> --title "标题"
"""
from __future__ import annotations

import argparse
import html
import re

CSS = """
body{font-family:"Microsoft YaHei","PingFang SC",-apple-system,Segoe UI,sans-serif;max-width:1180px;margin:0 auto;
padding:32px 28px 60px;color:#1f2328;line-height:1.65;font-size:15px;background:#fff}
h1{font-size:26px;color:#C00000;border-bottom:3px solid #C00000;padding-bottom:10px;margin:6px 0 18px}
h2{font-size:20px;color:#C00000;border-left:5px solid #C00000;padding-left:10px;margin:34px 0 12px}
h3{font-size:16.5px;color:#8B0000;margin:22px 0 8px}
p{margin:9px 0}
ul,ol{margin:8px 0 8px 22px;padding:0}li{margin:4px 0}
.tw{overflow-x:auto;margin:12px 0 16px}
table{border-collapse:collapse;width:100%;font-size:13.5px}
th{background:#fbeaea;color:#8B0000;font-weight:600;text-align:left;padding:7px 9px;border:1px solid #e3c9c9}
td{padding:6px 9px;border:1px solid #e3c9c9;vertical-align:top}
tbody tr:nth-child(even){background:#fcf8f8}
code{background:#f4f5f7;border-radius:3px;padding:1px 4px;font-family:Consolas,Menlo,monospace;font-size:12.8px;color:#a03030}
pre{background:#f7f7f9;border:1px solid #e5e5ea;border-left:4px solid #C00000;border-radius:4px;padding:11px 13px;overflow-x:auto}
pre code{background:none;color:#24292f;padding:0;font-size:12.6px;line-height:1.5}
blockquote{margin:10px 0;padding:8px 14px;background:#fdf6f6;border-left:4px solid #d98b8b;color:#5b4b4b}
hr{border:none;border-top:1px solid #eee;margin:22px 0}
.meta{color:#666;font-size:13px}
"""


def inline(t: str) -> str:
    t = html.escape(t)
    t = re.sub(r"`([^`]+)`", r"<code>\1</code>", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", t)
    t = re.sub(r"~~([^~]+)~~", r"<del>\1</del>", t)
    return t


def convert(md: str) -> str:
    out: list[str] = []
    lines = md.split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        if ln.startswith("```"):
            i += 1
            buf = []
            while i < len(lines) and not lines[i].startswith("```"):
                buf.append(html.escape(lines[i]))
                i += 1
            i += 1
            out.append("<pre><code>" + "\n".join(buf) + "</code></pre>")
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", ln)
        if m:
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{inline(m.group(2))}</h{lvl}>")
            i += 1
            continue
        if ln.strip().startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:|-]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in ln.strip().strip("|").split("|")]
            i += 2
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append([c.strip() for c in lines[i].strip().strip("|").split("|")])
                i += 1
            t = ["<table><thead><tr>"] + [f"<th>{inline(c)}</th>" for c in header] + ["</tr></thead><tbody>"]
            for r in rows:
                t.append("<tr>" + "".join(f"<td>{inline(c)}</td>" for c in r) + "</tr>")
            t.append("</tbody></table>")
            out.append('<div class="tw">' + "".join(t) + "</div>")
            continue
        if re.match(r"^\s*[-*]\s+", ln):
            buf = []
            while i < len(lines):
                if re.match(r"^\s*[-*]\s+", lines[i]):
                    buf.append(re.sub(r"^\s*[-*]\s+", "", lines[i]))
                    i += 1
                elif lines[i].startswith(("  ", "\t")) and lines[i].strip() and buf:
                    buf[-1] += " " + lines[i].strip()      # 续行并入上一条
                    i += 1
                else:
                    break
            out.append("<ul>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ul>")
            continue
        if re.match(r"^\s*\d+\.\s+", ln):
            buf = []
            while i < len(lines):
                if re.match(r"^\s*\d+\.\s+", lines[i]):
                    buf.append(re.sub(r"^\s*\d+\.\s+", "", lines[i]))
                    i += 1
                elif lines[i].startswith(("  ", "\t")) and lines[i].strip() and buf:
                    buf[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            out.append("<ol>" + "".join(f"<li>{inline(x)}</li>" for x in buf) + "</ol>")
            continue
        if ln.strip() == "---":
            out.append("<hr>")
            i += 1
            continue
        if not ln.strip():
            i += 1
            continue
        buf = []
        while i < len(lines) and lines[i].strip() and not re.match(r"^(#{1,4}\s|\||```|\s*[-*]\s|\s*\d+\.\s)", lines[i]):
            buf.append(lines[i].strip())
            i += 1
        out.append("<p>" + inline(" ".join(buf)) + "</p>")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("md")
    ap.add_argument("html")
    ap.add_argument("--title", default="报告")
    a = ap.parse_args()
    body = convert(open(a.md, encoding="utf-8").read())
    doc = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(a.title)}</title>
<style>{CSS}</style></head><body>
{body}
</body></html>"""
    open(a.html, "w", encoding="utf-8").write(doc)
    print(f"wrote {a.html} ({len(doc)} bytes)")


if __name__ == "__main__":
    main()
