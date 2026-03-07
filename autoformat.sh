#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

venv_candidates=("$script_dir/.venv/bin/activate")

parent_dir="$(dirname -- "$script_dir")"
if [ "$(basename -- "$parent_dir")" = ".worktrees" ]; then
    main_checkout_dir="$(dirname -- "$parent_dir")"
    venv_candidates+=("$main_checkout_dir/.venv/bin/activate")
fi

venv_activate=""
for candidate in "${venv_candidates[@]}"; do
    if [ -f "$candidate" ]; then
        venv_activate="$candidate"
        break
    fi
done

if [ -z "$venv_activate" ]; then
    printf 'autoformat.sh: no virtualenv found; checked:\n' >&2
    for candidate in "${venv_candidates[@]}"; do
        printf '  %s\n' "$candidate" >&2
    done
    exit 1
fi

# shellcheck source=/dev/null
source "$venv_activate"

cd "$script_dir"

ruff check --fix src/ tests/ scripts/
ruff format src/ tests/ scripts/
