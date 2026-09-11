#!/usr/bin/env bash
# Publish a completed experiment report to origin/main.
#
# Usage:
#   scripts/publish_report.sh <EXP-ID> [--message "<subject>"] [--yes]
#
# Without --yes this is a DRY RUN: it prints the working-tree changes, the
# commit subject it would use, and stops. Nothing is staged, committed or
# pushed. Pass --yes only after the user has approved the printed plan.
#
# Heavy artifacts (checkpoints/, logs/, metrics/, runs/, *.pth, *.png, *.npy)
# are excluded by .gitignore and are never published by this script.

set -euo pipefail

BRANCH="main"
REMOTE="origin"

exp_id=""
message=""
confirmed=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --message)
            message="$2"
            shift 2
            ;;
        --yes)
            confirmed=1
            shift
            ;;
        -*)
            echo "unknown option: $1" >&2
            exit 2
            ;;
        *)
            if [[ -n "$exp_id" ]]; then
                echo "unexpected extra argument: $1" >&2
                exit 2
            fi
            exp_id="$1"
            shift
            ;;
    esac
done

if [[ -z "$exp_id" ]]; then
    echo "usage: scripts/publish_report.sh <EXP-ID> [--message \"<subject>\"] [--yes]" >&2
    exit 2
fi

if [[ -z "$message" ]]; then
    message="Add ${exp_id} report and results"
fi

cd "$(git rev-parse --show-toplevel)"

current_branch="$(git rev-parse --abbrev-ref HEAD)"
if [[ "$current_branch" != "$BRANCH" ]]; then
    echo "[ABORT] on branch '${current_branch}', expected '${BRANCH}'." >&2
    exit 1
fi

echo "===== working tree changes (tracked + untracked, gitignore applied) ====="
git status --short
echo "===== files that WOULD be staged ====="
git add -A --dry-run
echo "===== commit subject ====="
echo "$message"
echo "===== target ====="
echo "${REMOTE}/${BRANCH}  ($(git remote get-url "$REMOTE"))"

if [[ -z "$(git status --porcelain)" ]]; then
    echo "[NOTHING TO DO] working tree is clean."
    exit 0
fi

if [[ "$confirmed" -ne 1 ]]; then
    echo
    echo "[DRY RUN] nothing staged, committed or pushed."
    echo "Re-run with --yes to publish."
    exit 0
fi

git add -A
git commit -m "$message"

sha="$(git rev-parse HEAD)"
echo "[COMMITTED] ${sha}"

git push "$REMOTE" "$BRANCH"
echo "[PUSHED] ${sha} -> ${REMOTE}/${BRANCH}"
