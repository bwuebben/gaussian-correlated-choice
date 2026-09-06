#!/usr/bin/env bash
set -euo pipefail
TASK_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
build_paper() {
  local directory="$1" name="$2"
  (
    cd "$TASK_DIR/$directory"
    latexmk -pdf -interaction=nonstopmode -halt-on-error \
      -outdir=build -jobname="$name" main.tex
    cp "build/$name.pdf" "$name.pdf"
  )
}
TARGET="${1:-all}"
if [[ "$#" -gt 1 ]]; then
  echo 'Usage: ./build.sh [all|paper|companion|commercial]' >&2
  exit 2
fi
case "$TARGET" in
  all)
    build_paper paper gaussian-choice-identification
    build_paper companion winning-probabilities-to-decisions
    build_paper commercial commercial-value-correlated-choice ;;
  paper) build_paper paper gaussian-choice-identification ;;
  companion) build_paper companion winning-probabilities-to-decisions ;;
  commercial) build_paper commercial commercial-value-correlated-choice ;;
  *) echo 'Usage: ./build.sh [all|paper|companion|commercial]' >&2; exit 2 ;;
esac
