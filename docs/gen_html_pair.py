#!/usr/bin/env python3
# coding=utf-8
"""md -> 自包含 HTML：GPU / NPU 实现并排 + Pygments 语法高亮 + 表格横滚 + 图片内嵌。

围栏约定：
    ```python gpu GPU_Reference/main_pytorch.py:65
    ```python npu OneTrans_NPU/main_pytorch.py:70
相邻的 gpu / npu 块自动合成两栏面板；单独出现时按单栏面板渲染。
用法：python3 gen_html_pair.py <src.md> <out.html> "<标题>"
"""
import base64
import html as _html
import os
import re
import sys

import markdown
from pygments import highlight as _pyg_highlight
from pygments.formatters import HtmlFormatter
from pygments.lexers import BashLexer, TextLexer, get_lexer_by_name

ROOT = "/root/bytedance-onetrans"
SRC = sys.argv[1]
OUT = sys.argv[2]
TITLE = sys.argv[3] if len(sys.argv) > 3 else "bytedance-onetrans 迁移指导"

_LEXERS = {"python": "python", "py": "python", "bash": "bash", "sh": "bash", "shell": "bash"}
_FMT = HtmlFormatter(style="monokai", noclasses=True, nowrap=True)
_ADD, _DEL = "#a6e22e", "#f92672"

SIDE_LABEL = {"gpu": "GPU 实现", "npu": "NPU 实现"}


def _lexer(lang: str):
    lang = (lang or "").strip().lower()
    if lang in _LEXERS:
        lang = _LEXERS[lang]
    if not lang:
        return TextLexer(stripnl=False)
    try:
        return get_lexer_by_name(lang)
    except Exception:
        return TextLexer(stripnl=False)


def _diff_aware(code: str, lexer) -> str:
    """+/- 前缀的对照块：标记单独着色，其余按语言高亮（不要用 DiffLexer，会丢 Python 语法色）。"""
    lines = code.split("\n")
    body = [l for l in lines if l.strip()]
    marks = [l for l in body if l[:1] in "+-" and not l.startswith(("+++", "---"))]
    if len(marks) < 2 or len(marks) * 4 < len(body):
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


def _pane(side: str, path: str, code: str) -> str:
    lang_lexer = _lexer("python")
    body = _diff_aware(code.rstrip("\n"), lang_lexer)
    fp = f'<span class="fp">{_html.escape(path)}</span>' if path else ""
    return (f'<div class="pane {side}"><div class="hd"><span>{SIDE_LABEL[side]}</span>{fp}</div>'
            f'<pre class="hl">{body}</pre></div>')


FENCE = re.compile(r"```(?P<info>[^\n]*)\n(?P<code>.*?)```", re.S)


def extract_panels(md: str):
    """单趟扫描替换（不能用「边改边按原偏移替换」，否则偏移错位、残留裸围栏）。"""
    units = []
    for m in FENCE.finditer(md):
        info = m.group("info").strip().split()
        side = info[1].lower() if len(info) > 1 and info[0].lower() in ("python", "py") else None
        if side not in ("gpu", "npu"):
            continue
        path = info[2] if len(info) > 2 else ""
        units.append({"start": m.start(), "end": m.end(), "side": side,
                      "path": path, "code": m.group("code")})

    placeholders, parts, pos, i, n = {}, [], 0, 0, 0
    while i < len(units):
        cur = units[i]
        nxt = units[i + 1] if i + 1 < len(units) else None
        paired = bool(nxt and cur["side"] == "gpu" and nxt["side"] == "npu"
                      and not md[cur["end"]:nxt["start"]].strip())
        key = f"@@PANEL{n}@@"
        n += 1
        if paired:
            placeholders[key] = (f'<div class="pair">{_pane("gpu", cur["path"], cur["code"])}'
                                 f'{_pane("npu", nxt["path"], nxt["code"])}</div>')
            end, i = nxt["end"], i + 2
        else:
            placeholders[key] = f'<div class="pair single">{_pane(cur["side"], cur["path"], cur["code"])}</div>'
            end, i = cur["end"], i + 1
        parts.append(md[pos:cur["start"]])
        parts.append(f"\n\n{key}\n\n")
        pos = end
    parts.append(md[pos:])
    return "".join(parts), placeholders


def main():
    md = open(os.path.join(ROOT, SRC), encoding="utf-8").read()

    # 1) 并排代码块 → 占位符（单趟替换）
    md, placeholders = extract_panels(md)

    # 2) 图片 → 占位符（base64 内嵌）
    imgs = {}

    def img_repl(m):
        rel = m.group(2)
        key = "@@FIG_" + os.path.basename(rel).replace(".", "_") + "@@"
        imgs[key] = rel
        return f"\n\n{key}\n\n"

    md = re.sub(r"!\[([^\]]*)\]\(([^)]+\.png)\)", img_repl, md)

    body = markdown.markdown(md, extensions=["tables", "fenced_code", "sane_lists"])

    # 3) 普通围栏块高亮
    def _one(m):
        lang, code = m.group(1), _html.unescape(m.group(2))
        return f'<pre class="hl">{_diff_aware(code, _lexer(lang))}</pre>'

    body = re.sub(r'<pre><code class="language-([a-z0-9+#-]*)">(.*?)</code></pre>', _one, body, flags=re.S)

    # 4) 回填并排面板
    for key, block in placeholders.items():
        body = body.replace(f"<p>{key}</p>", block).replace(key, block)

    # 5) 表格横滚 + 图片内嵌
    body = body.replace("<table>", '<div class="tw"><table>').replace("</table>", "</table></div>")
    for key, rel in imgs.items():
        b64 = base64.b64encode(open(os.path.join(ROOT, rel), "rb").read()).decode()
        body = body.replace(f"<p>{key}</p>",
                            f'<div class="fig"><img src="data:image/png;base64,{b64}" alt="{os.path.basename(rel)}"></div>')
        body = body.replace(key, "")

    # 6) 标题锚点 + 目录
    heads = []

    def h_repl(m):
        level, text = m.group(1), m.group(2)
        anchor = "h" + str(len(heads))
        heads.append((level, text, anchor))
        return f'<h{level} id="{anchor}">{text}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", h_repl, body, flags=re.S)
    # 目录链接：href 保留（脚本被禁时仍能跳），data-target 供 JS 精确滚动（飞书/沙箱预览里 hash 跳转常失效）
    _links = []
    for lv, t, a in heads:
        cls = "l2" if lv == "2" else "l3"
        esc = re.sub(r"<[^>]+>", "", t).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        _links.append(f'<a class="{cls}" href="#{a}" data-target="{a}" title="{esc}">{t}</a>')
    toc_items = "".join(_links)
    toc = (f'<nav class="toc" id="tocSidebar"><div class="toc-h">目录</div>'
           f'<div class="toc-nav" id="tocNav">{toc_items}</div>'
           f'<div class="toc-top"><a href="#top" data-target="top">↑ 回到顶部</a></div></nav>'
           f'<div class="toc-mask" id="tocMask"></div>'
           f'<button id="tocBtn" type="button" aria-label="目录">目录</button>') if toc_items else ""

    doc = (f'<!DOCTYPE html><html lang="zh-CN"><head><meta charset="utf-8">'
           f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title>'
           f'<style>{CSS}</style></head><body><div class="layout">{toc}'
           f'<div class="wrap" id="top">{body}</div></div>{TOC_JS}</body></html>')
    open(os.path.join(ROOT, OUT), "w", encoding="utf-8").write(doc)
    print(f"OK -> {OUT} ({os.path.getsize(os.path.join(ROOT, OUT))/1024:.0f} KB)  "
          f"表 {body.count('<table>')} 图 {len(imgs)} 面板 {len(placeholders)} 目录 {len(heads)}")


CSS = """
:root{--ink:#1a1a2e;--sub:#5a6270;--line:#e3e6ea;--brand:#C00000;--bg:#fbfbfc;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font-size:15px;line-height:1.78;
 font-family:"Microsoft YaHei","Noto Sans CJK SC","PingFang SC",Arial,sans-serif;-webkit-font-smoothing:antialiased;}
.layout{display:flex;gap:0;align-items:flex-start;max-width:1460px;margin:0 auto;}
html{scroll-padding-top:14px;}
.toc{position:sticky;top:0;flex:0 0 236px;width:236px;height:100vh;display:flex;flex-direction:column;z-index:50;
 background:#fbfbfc;font-size:13px;border-right:1px solid var(--line);}
.toc-h{flex:0 0 auto;padding:20px 14px 10px 18px;font-weight:700;color:var(--brand);font-size:13px;
 letter-spacing:1px;border-bottom:1px solid var(--line);}
.toc-nav{flex:1 1 auto;overflow-y:auto;overscroll-behavior:contain;padding:10px 8px 30px;}
.toc-nav::-webkit-scrollbar{width:7px}
.toc-nav::-webkit-scrollbar-thumb{background:#c9d1da;border-radius:4px}
.toc a{display:block;color:var(--sub);text-decoration:none;padding:4px 6px;border-radius:4px;line-height:1.5;
 cursor:pointer;-webkit-user-select:none;user-select:none;}
.toc a:hover{background:#f1f3f6;color:var(--ink);}
.toc a.l2{font-weight:700;color:#1c2733;margin-top:6px;}
.toc a.l3{font-size:12.4px;padding-left:18px;color:#5a6270;}
.toc a.active{background:#fff;color:var(--brand);font-weight:700;box-shadow:inset 3px 0 0 var(--brand);}
.toc-top{flex:0 0 auto;border-top:1px solid var(--line);padding:8px 16px;}
.toc-top a{color:var(--sub);font-size:12px;}
h2[id],h3[id]{scroll-margin-top:14px;}
#tocBtn{display:none;position:fixed;right:14px;bottom:18px;z-index:200;width:52px;height:52px;border:none;
 border-radius:50%;background:var(--brand);color:#fff;font-size:12px;font-weight:700;cursor:pointer;font-family:inherit;}
.toc-mask{display:none;position:fixed;inset:0;background:rgba(16,24,32,.42);z-index:40;}
.wrap{flex:1 1 auto;min-width:0;max-width:1120px;padding:40px 38px 90px;background:#fff;
 box-shadow:0 1px 24px rgba(0,0,0,.06);}
h1{font-size:26px;font-weight:700;color:var(--brand);margin:0 0 8px;}
h2{font-size:20px;font-weight:700;margin:40px 0 14px;padding-bottom:9px;border-bottom:2px solid var(--brand);}
h3{font-size:16.5px;font-weight:700;margin:26px 0 10px;}
p{margin:10px 0;}
code{font-family:"JetBrains Mono",Consolas,Monaco,monospace;font-size:12.8px;background:#f2f4f6;
 padding:2px 6px;border-radius:4px;color:#b02a37;white-space:normal;overflow-wrap:anywhere;}
pre{background:#1e2633;color:#e6edf3;padding:15px 18px;border-radius:7px;overflow-x:auto;font-size:12.8px;line-height:1.6;}
pre code{background:none;color:inherit;padding:0;}
pre.hl{background:#272822;color:#f8f8f2;padding:13px 16px;margin:0;overflow:auto;
 font-size:12.4px;line-height:1.62;}
.pair{display:grid;grid-template-columns:1fr 1fr;gap:12px;margin:16px 0 20px;align-items:start;}
.pair.single{grid-template-columns:1fr;}
@media(max-width:1180px){.pair{grid-template-columns:1fr}}
.pane{border:1px solid var(--line);border-radius:8px;overflow:hidden;background:#272822;
 display:flex;flex-direction:column;min-width:0;}
.pane .hd{display:flex;justify-content:space-between;align-items:center;gap:8px;
 font-size:12px;font-weight:700;padding:6px 11px;color:#fff;letter-spacing:.3px;}
.pane.gpu .hd{background:#2f6f4f;}
.pane.npu .hd{background:#8e1b1b;}
.pane .hd .fp{font-weight:400;font-size:11px;opacity:.9;
 font-family:"JetBrains Mono",Consolas,monospace;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.pane pre.hl{max-height:540px;}
.tw{overflow-x:auto;max-width:100%;margin:16px 0;}
table{border-collapse:collapse;width:max-content;min-width:100%;font-size:13.4px;background:#fff;
 box-shadow:0 1px 4px rgba(0,0,0,.05);}
th{background:#C00000;color:#fff;font-weight:600;padding:9px 11px;text-align:left;border:1px solid #a30000;white-space:nowrap;}
td{padding:9px 11px;border:1px solid var(--line);vertical-align:top;}
tbody tr:nth-child(even){background:#f8fafb;}
tbody tr:hover{background:#eef4fa;}
.fig{margin:22px 0 26px;text-align:center;}
.fig img{max-width:100%;height:auto;border:1px solid var(--line);border-radius:7px;box-shadow:0 2px 10px rgba(0,0,0,.07);}
blockquote{margin:18px 0;padding:13px 18px;background:#f7f9fb;border-left:4px solid #8fa8c0;color:#3a4756;font-size:14px;}
blockquote p{margin:6px 0;}
@media(max-width:1180px){
 .layout{display:block;}
 .toc{position:fixed;top:0;left:0;transform:translateX(-104%);transition:transform .22s ease;
  box-shadow:2px 0 18px rgba(0,0,0,.18);width:280px;flex-basis:280px;}
 body.toc-open .toc{transform:translateX(0);}
 body.toc-open .toc-mask{display:block;}
 #tocBtn{display:block;}
 body.toc-open #tocBtn{display:none;}
 .wrap{padding:26px 18px 70px;max-width:none;}
}
@media(max-width:760px){h1{font-size:21px}h2{font-size:18px}table{font-size:12px}th,td{padding:7px 8px}}
"""

TOC_JS = """
<script>
(function(){
  var nav = document.getElementById('tocNav');
  if(!nav) return;
  var links = Array.prototype.slice.call(nav.querySelectorAll('a'));
  var active = -1;
  function jump(el){
    if(!el) return;
    try { el.scrollIntoView({block:'start', inline:'nearest'}); }
    catch(e){ window.scrollTo(0, el.getBoundingClientRect().top + window.pageYOffset - 14); }
    clearTimeout(window.__tocFix);
    window.__tocFix = setTimeout(function(){
      var top = el.getBoundingClientRect().top;
      if(Math.abs(top) > 24){ window.scrollBy(0, top - 14); }
    }, 260);
  }
  function setActive(i){
    if(i === active || i < 0 || !links[i]) return;
    active = i;
    links.forEach(function(a){ a.classList.remove('active'); });
    var a = links[i]; a.classList.add('active');
    var r = a.getBoundingClientRect(), nr = nav.getBoundingClientRect();
    if(r.top < nr.top + 8 || r.bottom > nr.bottom - 8){
      nav.scrollTop += (r.top - nr.top) - nav.clientHeight/2 + r.height/2;
    }
  }
  function onScroll(){
    var best = 0, y = 60;
    for(var i = 0; i < links.length; i++){
      var el = document.getElementById(links[i].dataset.target);
      if(!el) continue;
      if(el.getBoundingClientRect().top - y <= 0) best = i; else break;
    }
    setActive(best);
  }
  links.forEach(function(a){
    a.addEventListener('click', function(ev){
      var el = document.getElementById(a.dataset.target);
      if(!el) return;
      ev.preventDefault();
      jump(el);
      document.body.classList.remove('toc-open');
      var i = links.indexOf(a);
      if(i >= 0){ active = -1; setActive(i); }
      if(history.replaceState){
        try { history.replaceState(null, '', '#' + a.dataset.target); } catch(e){}
      }
    });
  });
  var ticking = false;
  window.addEventListener('scroll', function(){
    if(!ticking){ ticking = true; requestAnimationFrame(function(){ onScroll(); ticking = false; }); }
  }, {passive:true});
  window.addEventListener('resize', onScroll);
  onScroll();
  var btn = document.getElementById('tocBtn'), mask = document.getElementById('tocMask');
  if(btn) btn.addEventListener('click', function(){ document.body.classList.toggle('toc-open'); });
  if(mask) mask.addEventListener('click', function(){ document.body.classList.remove('toc-open'); });
  document.addEventListener('keydown', function(e){ if(e.key === 'Escape') document.body.classList.remove('toc-open'); });
})();
</script>
"""

if __name__ == "__main__":
    main()
