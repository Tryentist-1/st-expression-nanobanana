#!/bin/bash
cd "$(dirname "$0")"
echo "Starting Silly Sprites..."
# Try to find uv in standard locations if not in PATH
export PATH="$HOME/.local/bin:$PATH"

if command -v uv &> /dev/null; then
    uv run sprites.py gui
else
    echo "Error: 'uv' not found. Please install uv or make sure it's in your PATH."
    read -p "Press enter to close..."
fi
