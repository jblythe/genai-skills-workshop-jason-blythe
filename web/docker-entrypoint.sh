#!/bin/sh
set -e

API_URL="${VITE_API_URL:-}"
if [ -z "$API_URL" ]; then
  echo "[ADS] Warning: VITE_API_URL not set; frontend will default to its own origin." >&2
fi

cat <<EOF > /usr/share/nginx/html/config.js
window.__ADS_CONFIG__ = { apiUrl: "${API_URL}" };
EOF

echo "[ADS] Generated runtime config with apiUrl='${API_URL}'" >&2

exec "$@"
