#!/bin/sh
sleep 35

URL="http://127.0.0.1:5000"

if command -v chromium-browser >/dev/null 2>&1; then
    chromium-browser \
        --no-sandbox \
        --no-first-run \
        --password-store=basic \
        --user-data-dir=/tmp/ev-camera-browser \
        --start-fullscreen \
        "$URL" >/tmp/ev_open_web.log 2>&1 &
elif command -v chromium >/dev/null 2>&1; then
    chromium \
        --no-sandbox \
        --no-first-run \
        --password-store=basic \
        --user-data-dir=/tmp/ev-camera-browser \
        --start-fullscreen \
        "$URL" >/tmp/ev_open_web.log 2>&1 &
else
    xdg-open "$URL" >/tmp/ev_open_web.log 2>&1 &
fi
