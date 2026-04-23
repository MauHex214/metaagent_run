#!/usr/bin/env bash
# Local fallback mirror script. Identical scope to the GitHub Action
# .github/workflows/mirror-to-hydrosphere-meta-norm.yml — useful when
# the Action is not wired up or when you want to preview the diff
# before pushing.
#
# Scope:
#   metaagent_run/core/                       →  hydrosphere_meta_norm/metaagent_run/core/
#   metaagent_run/steps/env_field_pipeline/   →  hydrosphere_meta_norm/metaagent_run/steps/env_field_pipeline/
#   docs/                                     →  hydrosphere_meta_norm/docs/
#
# README / requirements / other steps in hydrosphere_meta_norm are left alone
# (the two repos' root-level docs intentionally differ).
#
# Usage:
#   bash scripts/sync_to_hydrosphere_meta_norm.sh [--push] [/path/to/hydrosphere_meta_norm]
#
# By default the target is ../hydrosphere_meta_norm relative to this repo.
# Without --push the script stages + commits locally and shows the diff;
# pass --push to also push to origin/main on hydrosphere_meta_norm.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

DO_PUSH=0
TGT_ROOT=""
for arg in "$@"; do
  case "$arg" in
    --push) DO_PUSH=1 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *) TGT_ROOT="$arg" ;;
  esac
done

if [[ -z "$TGT_ROOT" ]]; then
  TGT_ROOT="$(cd "${SRC_ROOT}/../hydrosphere_meta_norm" 2>/dev/null && pwd || true)"
fi
if [[ -z "$TGT_ROOT" || ! -d "$TGT_ROOT/.git" ]]; then
  echo "error: hydrosphere_meta_norm clone not found." >&2
  echo "  expected:   ${SRC_ROOT%/*}/hydrosphere_meta_norm" >&2
  echo "  or pass path:  bash $0 /path/to/hydrosphere_meta_norm" >&2
  exit 2
fi

echo "source:  $SRC_ROOT"
echo "target:  $TGT_ROOT"
echo

# Pull target first so we don't fight remote changes.
( cd "$TGT_ROOT" && git pull --ff-only origin main )

# Mirror the three subtrees.
rsync -a --delete "${SRC_ROOT}/metaagent_run/core/"                     "${TGT_ROOT}/metaagent_run/core/"
rsync -a --delete "${SRC_ROOT}/metaagent_run/steps/env_field_pipeline/" "${TGT_ROOT}/metaagent_run/steps/env_field_pipeline/"
rsync -a --delete "${SRC_ROOT}/docs/"                                    "${TGT_ROOT}/docs/"

cd "$TGT_ROOT"
if git diff --quiet && git diff --cached --quiet; then
  echo "No changes to mirror."
  exit 0
fi

SRC_SHA=$(cd "$SRC_ROOT" && git rev-parse HEAD)
SRC_SHORT=${SRC_SHA:0:7}

git add -A metaagent_run/core metaagent_run/steps/env_field_pipeline docs
git status --short
echo
git commit -m "Mirror from metaagent_run@${SRC_SHORT}

Auto-synced paths:
  - metaagent_run/core/
  - metaagent_run/steps/env_field_pipeline/
  - docs/

Source: https://github.com/MauHex214/metaagent_run/commit/${SRC_SHA}"

if [[ "$DO_PUSH" == "1" ]]; then
  git push origin main
  echo "✓ pushed to hydrosphere_meta_norm"
else
  echo
  echo "Commit staged locally. Re-run with --push to publish:"
  echo "    bash $0 --push"
fi
