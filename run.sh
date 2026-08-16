#!/usr/bin/env bash
# qml-evalkit — demo | test
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -d .venv ]; then
    python3 -m venv .venv
    ./.venv/bin/pip install -q -e ".[dev]"
fi

case "${1:-demo}" in
    demo) ./.venv/bin/python -m qml_evalkit.demo ;;
    test) ./.venv/bin/python -m pytest tests/ -q ;;
    *) echo "usage: ./run.sh [demo|test]" >&2; exit 1 ;;
esac
