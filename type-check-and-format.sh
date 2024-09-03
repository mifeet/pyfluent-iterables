#!/usr/bin/env bash
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
poetry run mypy "$BASE_DIR"
poetry run black "$BASE_DIR"