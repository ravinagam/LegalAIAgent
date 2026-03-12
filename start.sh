#!/usr/bin/env bash
# Legal AI Agent — startup script
set -e

echo "=== Legal AI Agent ==="

# 1. Create virtual environment if missing
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment..."
  python -m venv .venv
fi

# 2. Activate
source .venv/bin/activate 2>/dev/null || source .venv/Scripts/activate

# 3. Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# 4. Check API key
if [ -z "$ANTHROPIC_API_KEY" ]; then
  if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
  fi
fi

if [ -z "$ANTHROPIC_API_KEY" ]; then
  echo ""
  echo "ERROR: ANTHROPIC_API_KEY is not set."
  echo "  1. Copy .env.example to .env"
  echo "  2. Add your API key from https://console.anthropic.com"
  exit 1
fi

echo ""
echo "Starting server at http://localhost:8000"
echo ""
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
