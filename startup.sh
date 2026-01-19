#!/bin/bash
cd "$(dirname "$0")"
echo "Starting SillyTavern Sprite Generator..."
uv run sprites.py gui
