#!/usr/bin/env bash
set -euo pipefail

APP_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
RUSTSET_PORT="${RUSTSET_PORT:-7860}"

if [[ -z "${CLOUDFLARE_ALLOWED_EMAIL:-}" ]]; then
  echo "Set CLOUDFLARE_ALLOWED_EMAIL to the email allowed to open the dashboard." >&2
  echo "Example: export CLOUDFLARE_ALLOWED_EMAIL=you@example.com" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  echo "Python 3 is required." >&2
  exit 1
fi

# pip.py in this repository shadows `python -m pip` when run from APP_ROOT.
if command -v pip >/dev/null 2>&1; then
  pip install -r "$APP_ROOT/requirements.txt"
else
  (cd /tmp && python3 -m pip install -r "$APP_ROOT/requirements.txt")
fi

if command -v cloudflared >/dev/null 2>&1; then
  CLOUDFLARED_BIN="$(command -v cloudflared)"
else
  case "$(uname -m)" in
    x86_64|amd64) CLOUDFLARED_ARCH="amd64" ;;
    aarch64|arm64) CLOUDFLARED_ARCH="arm64" ;;
    *) echo "Unsupported CPU architecture: $(uname -m)" >&2; exit 1 ;;
  esac
  CLOUDFLARED_BIN="${HOME}/.local/bin/cloudflared"
  mkdir -p "$(dirname "$CLOUDFLARED_BIN")"
  curl -fsSL "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-${CLOUDFLARED_ARCH}" -o "$CLOUDFLARED_BIN"
  chmod +x "$CLOUDFLARED_BIN"
fi

python3 "$APP_ROOT/ui.py" --host 127.0.0.1 --port "$RUSTSET_PORT" &
UI_PID=$!
cleanup() {
  kill "$UI_PID" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "Dashboard is starting on http://127.0.0.1:${RUSTSET_PORT}"
echo "Open the https://*.trycloudflare.com URL printed below and verify the allowed email."

"$CLOUDFLARED_BIN" tunnel --url "http://127.0.0.1:${RUSTSET_PORT}" --allowed-mail "$CLOUDFLARE_ALLOWED_EMAIL"
