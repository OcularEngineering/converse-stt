#!/usr/bin/env bash
# Publish the Converse-STT board to both sites as review PRs.
#
#   scripts/publish.sh [--dry-run] [--no-pr] [--new-version] [--generated 'Oct 27, 2026']
#
# Runs here, on a machine that has the scored runs (runs/ is not in git, so this
# cannot run in CI). Steps:
#
#   1. export/public    <- ocular-evals export                      (pre-release models excluded)
#   2. export/internal  <- ocular-evals export --include-prerelease
#   3. guard: the public export names no pre-release run and carries no stray transcript
#   4. sync each export into its site checkout on a fresh branch, run the site's own
#      pre-build check on the result, commit, push, open or update a PR
#
# Nothing is merged by this script. Each PR's diff is the complete set of
# numbers that will change on that site; Vercel builds a preview per PR.
#
# Site checkouts default to the folder layout on the publishing machine and can
# be overridden:
#   OCULAR_EVALS_WEB   internal site  (default: ../../ocular-evals-web, relative to this repo)
#   OCULAR_WEBSITE     public site    (default: ~/Desktop/Engineering/website-latest)
#
# Needs: uv, bun, gh (logged in), and both checkouts clean on their main branch.

set -euo pipefail

HARNESS="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
INTERNAL_SITE="${OCULAR_EVALS_WEB:-$HARNESS/../../ocular-evals-web}"
PUBLIC_SITE="${OCULAR_WEBSITE:-$HOME/Desktop/Engineering/website-latest}"
BENCHMARK="converse-stt"

DRY_RUN=0
OPEN_PR=1
NEW_VERSION=""
GENERATED=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1 ;;
    --no-pr) OPEN_PR=0 ;;
    --new-version) NEW_VERSION="--new-version" ;;
    --generated) GENERATED=(--generated "$2"); shift ;;
    -h|--help) sed -n '2,24p' "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

say() { printf '\n\033[1m%s\033[0m\n' "$*"; }
die() { echo "error: $*" >&2; exit 1; }

for dir in "$INTERNAL_SITE" "$PUBLIC_SITE"; do
  [[ -d "$dir/.git" ]] || die "site checkout not found: $dir"
done

# ------------------------------------------------------------------ 1-2. export
cd "$HARNESS"
say "exporting public profile"
uv run ocular-evals export --out export/public "${GENERATED[@]}"
say "exporting internal profile"
uv run ocular-evals export --out export/internal --include-prerelease "${GENERATED[@]}"

PUBLIC_BOARD="export/public/$BENCHMARK"
VERSION="$(ls "$PUBLIC_BOARD" | sed -n 's/^v//p' | sort -n | tail -1)"
[[ -n "$VERSION" ]] || die "no version directory under $PUBLIC_BOARD"
PUBLIC_DIR="$PUBLIC_BOARD/v$VERSION"
INTERNAL_DIR="export/internal/$BENCHMARK/v$VERSION"

# ------------------------------------------------------------------ 3. guard
say "checking the public export"
python3 - "$PUBLIC_DIR" "$INTERNAL_DIR" <<'EOF'
import json, os, sys
public, internal = sys.argv[1], sys.argv[2]
board = json.load(open(os.path.join(public, "board.json")))
leaked = [r["id"] for r in board["runs"] + board["operatingPoints"] if r.get("prerelease")]
if leaked:
    sys.exit(f"public export carries pre-release row(s): {leaked}")
ids = {r["id"] for r in board["runs"]}
files = {f[:-5] for f in os.listdir(os.path.join(public, "transcripts")) if f.endswith(".json")}
if files != ids:
    sys.exit(f"public transcripts do not match board runs: extra={sorted(files - ids)} missing={sorted(ids - files)}")
hidden = [r["id"] for r in json.load(open(os.path.join(internal, "board.json")))["runs"] if r.get("prerelease")]
text = open(os.path.join(public, "board.json")).read()
for name in hidden:
    if name in text:
        sys.exit(f"public board.json mentions pre-release run {name}")
print(f"ok: {len(ids)} public runs, {len(hidden)} pre-release run(s) internal-only")
EOF

GENERATED_LABEL="$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['generated'])" "$PUBLIC_DIR/board.json")"
COMMIT_SHA="$(git -C "$HARNESS" rev-parse --short HEAD)"
BRANCH="harness/$BENCHMARK-v$VERSION-$(date +%Y%m%d)"
TITLE="Converse-STT v$VERSION board: $GENERATED_LABEL (harness $COMMIT_SHA)"

# ------------------------------------------------------------------ 4. sync + PR
publish_site() {
  local label="$1" site="$2" export_dir="$3" check="$4"; shift 4
  say "syncing $label ($site)"
  cd "$site"
  [[ -z "$(git status --porcelain)" ]] || die "$label checkout has uncommitted changes; commit or stash them first"
  git fetch -q origin main
  git checkout -q -B "$BRANCH" origin/main
  "$@" "$export_dir"
  # The site's own pre-build check: schema, transcripts, logos, and (public) no pre-release rows.
  bun run "$check" || die "$label: synced data failed its check; nothing committed"
  if [[ -z "$(git status --porcelain)" ]]; then
    echo "$label: nothing changed"
    git checkout -q main
    return
  fi
  git add -A
  git --no-pager diff --cached --stat | tail -5
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "$label: dry run; leaving changes staged on $BRANCH"
    return
  fi
  git commit -q -m "$TITLE" -m "Synced from ocular-evals export (commit $COMMIT_SHA). Review the diff: it is the complete set of numbers that change on the site."
  git push -q -u origin "$BRANCH"
  if [[ $OPEN_PR -eq 1 ]]; then
    if gh pr view "$BRANCH" >/dev/null 2>&1; then
      echo "$label: PR already open, branch updated"
      gh pr view "$BRANCH" --json url -q .url
    else
      gh pr create --base main --head "$BRANCH" --title "$TITLE" \
        --body "Board data synced from the ocular-evals harness at commit \`$COMMIT_SHA\`, generated $GENERATED_LABEL. The diff is the complete set of numbers that will change on this site; the Vercel preview shows the result. Merging publishes."
    fi
  fi
  git checkout -q main
}

publish_site "internal site" "$INTERNAL_SITE" "$HARNESS/export/internal" check \
  bun run sync --benchmark "$BENCHMARK" --version "$VERSION" $NEW_VERSION --from

publish_site "public site" "$PUBLIC_SITE/apps/web" "$HARNESS/export/public" stt:check \
  bun run stt:sync -- --version "$VERSION" $NEW_VERSION --from

say "done"
