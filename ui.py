#!/usr/bin/env python3
"""Small browser dashboard for running the Rust dataset pipeline."""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from core import Config
from repo import parse_repo

ROOT = Path(__file__).resolve().parent
STATE = {"running": False, "logs": [], "error": None, "result": None}
STATE_LOCK = threading.Lock()

PAGE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rustset · Dataset console</title>
  <style>
    :root{color-scheme:dark;--ink:#101820;--panel:#17232c;--panel-2:#1d2b35;--line:#33434d;--paper:#edf0e9;--muted:#9aa9a9;--green:#a2e4ad;--orange:#ff9d69;--red:#fa8072;--mono:ui-monospace,SFMono-Regular,Consolas,monospace;--sans:Inter,ui-sans-serif,system-ui,-apple-system,"Segoe UI",sans-serif}
    *{box-sizing:border-box}body{margin:0;background:var(--ink);color:var(--paper);font:15px/1.5 var(--sans)}
    .shell{max-width:1160px;margin:0 auto;padding:36px 28px 56px}.mast{display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid var(--line);padding-bottom:17px;color:var(--muted);font:11px var(--mono);letter-spacing:.12em;text-transform:uppercase}
    .brand{display:flex;align-items:center;gap:11px;color:var(--paper)}.mark{width:22px;height:22px;border:2px solid var(--orange);border-radius:50%;position:relative}.mark:after{content:"";position:absolute;left:8px;top:-3px;height:24px;border-left:2px solid var(--orange);transform:rotate(36deg)}
    .eyebrow{color:var(--green);font:11px var(--mono);letter-spacing:.16em;text-transform:uppercase;margin:34px 0 11px}h1{font-family:Georgia,"Times New Roman",serif;font-size:clamp(36px,5vw,58px);font-weight:400;letter-spacing:-.035em;line-height:1.05;max-width:740px;margin:0}.intro{color:#b4bfba;max-width:650px;margin:17px 0 30px;font-size:16px}
    .grid{display:grid;grid-template-columns:minmax(0,1.12fr) minmax(310px,.88fr);gap:18px;align-items:start}.panel{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:22px}.panel h2{font:13px var(--mono);letter-spacing:.06em;text-transform:uppercase;margin:0 0 19px;color:var(--green)}
    .field{margin:0 0 15px}.row{display:grid;grid-template-columns:1fr 1fr;gap:14px}label{display:block;font-size:13px;margin-bottom:6px;color:#d7dfd8}input,textarea{display:block;width:100%;border:1px solid #43545e;border-radius:5px;background:#0f1920;color:var(--paper);font:14px/1.45 var(--sans);padding:10px 11px;outline:none}input:focus,textarea:focus{border-color:var(--green);box-shadow:0 0 0 2px #a2e4ad33}textarea{resize:vertical;min-height:92px}#template{min-height:210px;font:12px/1.5 var(--mono);color:#d4dfd6}.hint{margin:5px 0 0;color:var(--muted);font-size:12px}.button{appearance:none;border:0;border-radius:5px;background:var(--orange);color:#201810;font-weight:700;padding:12px 16px;font-size:14px;cursor:pointer;width:100%;margin-top:2px}.button:hover{background:#ffb18a}.button:focus-visible{outline:2px solid var(--paper);outline-offset:3px}.button:disabled{opacity:.55;cursor:wait}
    .statusline{display:flex;align-items:center;justify-content:space-between;margin-bottom:13px}.pill{font:11px var(--mono);letter-spacing:.06em;text-transform:uppercase;border:1px solid var(--line);border-radius:20px;padding:5px 9px;color:var(--muted)}.pill.running{color:var(--orange);border-color:#89583d}.pill.done{color:var(--green);border-color:#54765d}.pill.failed{color:var(--red);border-color:#874943}.summary{font-size:13px;color:#d1dbd3;min-height:21px;margin-bottom:12px}.logs{background:#0c141a;border:1px solid #293740;border-radius:5px;min-height:278px;max-height:420px;overflow:auto;padding:12px;color:#b8c6c0;font:12px/1.65 var(--mono);white-space:pre-wrap;overflow-wrap:anywhere}.tunnel{margin-top:17px;padding:13px 14px;border-left:2px solid var(--orange);background:var(--panel-2);font-size:12px;color:#c4cec8}.tunnel strong{display:block;color:var(--paper);font:11px var(--mono);letter-spacing:.08em;text-transform:uppercase;margin-bottom:4px}.foot{margin-top:18px;color:var(--muted);font-size:12px}
    @media(max-width:780px){.shell{padding:22px 16px 40px}.grid{grid-template-columns:1fr}.mast span:last-child{display:none}.row{grid-template-columns:1fr;gap:0}.logs{min-height:190px}}
    @media(prefers-reduced-motion:reduce){*{scroll-behavior:auto!important;transition:none!important}}
  </style>
</head>
<body>
<main class="shell">
  <div class="mast"><div class="brand"><span class="mark" aria-hidden="true"></span><span>Rustset · Dataset console</span></div><span>Notebook runner / quick tunnel</span></div>
  <p class="eyebrow">Source in · instruction data out</p>
  <h1>Turn Rust repositories into training data.</h1>
  <p class="intro">Choose the source repositories, connect an OpenAI-compatible model, and send the finished dataset to Hugging Face.</p>
  <div class="grid">
    <form class="panel" id="run-form">
      <h2>Pipeline settings</h2>
      <div class="field"><label for="repos">GitHub repositories</label><textarea id="repos" name="repos" placeholder="owner/repo&#10;https://github.com/owner/another-repo" required></textarea><p class="hint">One repository per line, up to 10. Public GitHub repositories are supported.</p></div>
      <div class="row">
        <div class="field"><label for="api-key">API key</label><input id="api-key" name="api_key" type="password" autocomplete="new-password" placeholder="sk-…"></div>
        <div class="field"><label for="model">Model</label><input id="model" name="model" value="gpt-4o-mini" required></div>
      </div>
      <div class="field"><label for="base-url">API endpoint</label><input id="base-url" name="base_url" type="url" placeholder="https://api.openai.com/v1"><p class="hint">Leave blank for OpenAI. Use a compatible API base URL for other providers.</p></div>
      <div class="row">
        <div class="field"><label for="hf-token">Hugging Face token</label><input id="hf-token" name="hf_token" type="password" autocomplete="new-password" placeholder="hf_…"></div>
        <div class="field"><label for="hf-repo">Hugging Face dataset repo</label><input id="hf-repo" name="hf_repo" placeholder="username/dataset-name"></div>
      </div>
      <div class="field"><label for="template">README template</label><textarea id="template" name="readme_template" spellcheck="false"></textarea><p class="hint">Tokens: {{dataset_repo}}, {{model}}, {{row_count}}, {{repo_count}}, {{license}}, {{size_category}}, {{source_repositories}}, {{rows_by_kind}}, {{example}}.</p></div>
      <button class="button" id="run-button" type="submit">Generate dataset</button>
    </form>
    <section class="panel" aria-labelledby="run-title">
      <div class="statusline"><h2 id="run-title" style="margin:0">Run output</h2><span class="pill" id="state">Ready</span></div>
      <div class="summary" id="summary" aria-live="polite">Add settings and start a run.</div>
      <pre class="logs" id="logs" aria-live="polite">Pipeline logs will appear here.</pre>
      <div class="tunnel"><strong>Cloudflare Quick Tunnel</strong>Open the <code>trycloudflare.com</code> address printed in notebook output and verify the allowed email. Stop the launcher when finished; the URL is temporary.</div>
      <p class="foot">Credentials stay in this process's memory and are not saved to a file. Cloudflare requires the configured email before it serves this dashboard.</p>
    </section>
  </div>
</main>
<script>
const form=document.querySelector('#run-form'),button=document.querySelector('#run-button'),state=document.querySelector('#state'),summary=document.querySelector('#summary'),logs=document.querySelector('#logs');
fetch('/api/template').then(r=>r.json()).then(d=>{document.querySelector('#template').value=d.template}).catch(()=>{});
function showStatus(data){state.textContent=data.running?'Running':data.error?'Failed':data.result?'Complete':'Ready';state.className='pill'+(data.running?' running':data.error?' failed':data.result?' done':'');button.disabled=data.running;summary.textContent=data.error||data.result?.message||(data.running?'Pipeline is working.':'Add settings and start a run.');logs.textContent=data.logs.length?data.logs.join('\n'):'Pipeline logs will appear here.';logs.scrollTop=logs.scrollHeight}
async function refresh(){try{const r=await fetch('/api/status');showStatus(await r.json())}catch{}}
form.addEventListener('submit',async e=>{e.preventDefault();button.disabled=true;summary.textContent='Starting pipeline…';try{const body=Object.fromEntries(new FormData(form));const request=fetch('/api/run',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});document.querySelector('#api-key').value='';document.querySelector('#hf-token').value='';const r=await request;const data=await r.json();if(!r.ok)throw new Error(data.error);await refresh()}catch(err){summary.textContent=err.message;button.disabled=false}});
refresh();setInterval(refresh,1800);
</script>
</body>
</html>"""


def _redact(message: str, secrets: list[str]) -> str:
    for secret in secrets:
        if secret:
            message = message.replace(secret, "[redacted]")
    return message


def _run_pipeline(data: dict, specs: list[str]) -> None:
    secrets = [data.get("api_key", ""), data.get("hf_token", "")]

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            message = _redact(self.format(record), secrets)
            with STATE_LOCK:
                STATE["logs"].append(message)
                STATE["logs"] = STATE["logs"][-250:]

    logger = logging.getLogger("rustset")
    handler = Capture()
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    logger.propagate = False
    try:
        from pip import Pipeline  # local pipeline module

        hf_repo = data.get("hf_repo", "").strip() or None
        cfg = Config(
            api_key=data.get("api_key", "").strip() or None,
            base_url=data.get("base_url", "").strip() or None,
            model=data.get("model", "gpt-4o-mini").strip(),
            hf_repo=hf_repo,
            hf_token=data.get("hf_token", "").strip() or None,
            upload=bool(hf_repo),
            readme_template=data.get("readme_template") or None,
        )
        result = asyncio.run(Pipeline(cfg).run(specs))
        message = f"Created {result['rows']:,} rows · {result['parquet']} · {result['readme']}"
        if result["url"]:
            message += f" · {result['url']}"
        with STATE_LOCK:
            STATE["result"] = {"message": message}
    except (Exception, SystemExit) as exc:
        with STATE_LOCK:
            STATE["error"] = _redact(str(exc), secrets) or "Pipeline stopped before producing output."
    finally:
        logger.removeHandler(handler)
        with STATE_LOCK:
            STATE["running"] = False


class Handler(BaseHTTPRequestHandler):
    def _json(self, status: int, data: dict) -> None:
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/api/status":
            with STATE_LOCK:
                data = {**STATE, "logs": list(STATE["logs"])}
            self._json(200, data)
        elif self.path == "/api/template":
            try:
                template = (ROOT / "README_TEMPLATE.md").read_text(encoding="utf-8")
            except OSError:
                template = ""
            self._json(200, {"template": template})
        elif self.path == "/":
            body = PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/api/run":
            self.send_error(404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            if not 0 < length <= 512_000:
                raise ValueError("Request must be under 512 KB.")
            data = json.loads(self.rfile.read(length))
            if not isinstance(data, dict):
                raise ValueError("Request must contain dashboard settings.")
            raw_repos = [part.strip() for line in data.get("repos", "").splitlines() for part in line.split(",") if part.strip()]
            refs = list(dict.fromkeys(parse_repo(item).full_name for item in raw_repos))
            if not 1 <= len(refs) <= 10:
                raise ValueError("Enter between 1 and 10 GitHub repositories.")
            if not data.get("api_key", "").strip() and not data.get("base_url", "").strip():
                raise ValueError("Enter an API key, or set a compatible endpoint that does not need one.")
            if data.get("hf_repo", "").strip() and not data.get("hf_token", "").strip():
                raise ValueError("Add a Hugging Face token to upload, or clear the dataset repo to save locally.")
            if len(data.get("readme_template", "")) > 200_000:
                raise ValueError("README template must be under 200 KB.")
            data["model"] = data.get("model", "").strip()
            if not data["model"]:
                raise ValueError("Enter a model name.")
        except (ValueError, TypeError, json.JSONDecodeError, AttributeError) as exc:
            self._json(400, {"error": str(exc)})
            return

        with STATE_LOCK:
            if STATE["running"]:
                self._json(409, {"error": "A pipeline is already running."})
                return
            STATE.update(running=True, logs=[], error=None, result=None)
        threading.Thread(target=_run_pipeline, args=(data, refs), daemon=True).start()
        self._json(202, {"running": True})

    def log_message(self, fmt: str, *args) -> None:
        logging.getLogger("http.server").info("%s - %s", self.address_string(), fmt % args)


def main() -> None:
    parser = argparse.ArgumentParser(description="Rustset browser dashboard")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Rustset dashboard listening at http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
