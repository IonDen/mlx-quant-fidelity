#!/usr/bin/env bash
# Re-render every committed PlantUML diagram to SVG.
#
# Usage: bash scripts/render_diagrams.sh
#
# Uses PlantUML's built-in Smetana layout engine so graphviz is not required.
# Sources and rendered output are committed side by side.
set -euo pipefail

cd "$(dirname "$0")/.."

for src in docs/assets/diagrams/*.puml; do
  [ -e "$src" ] || continue
  echo "rendering $src"
  plantuml -tsvg -Playout=smetana -o "$(pwd)/$(dirname "$src")" "$src"
done

echo "done"
