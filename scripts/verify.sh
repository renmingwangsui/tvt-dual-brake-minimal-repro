#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-python}"

for command_name in sha256sum latexmk rsvg-convert pdfinfo; do
  command -v "$command_name" >/dev/null || { printf 'MISSING_TOOL=%s\n' "$command_name" >&2; exit 1; }
done
command -v "$python_bin" >/dev/null || { printf 'MISSING_PYTHON=%s\n' "$python_bin" >&2; exit 1; }

(cd "$repo_root" && sha256sum -c SHA256SUMS >/dev/null)
printf 'PACKAGE_HASHES=PASS\n'

(cd "$repo_root/science" && "$python_bin" scripts/check_autodiff_backend.py)
(cd "$repo_root/science" && "$python_bin" scripts/run_regression_suite.py)

work_dir="$(mktemp -d /tmp/tvt-repro-XXXXXX)"
cleanup() {
  resolved="$(realpath "$work_dir")"
  case "$resolved" in
    /tmp/tvt-repro-*) rm -rf -- "$resolved" ;;
    *) printf 'REFUSING_UNSAFE_CLEANUP=%s\n' "$resolved" >&2 ;;
  esac
}
trap cleanup EXIT
cp -a "$repo_root/paper" "$work_dir/paper"

"$python_bin" "$work_dir/paper/scripts/generate_submission_figures.py"
figure_dir="generated/submission_revision/figures"
for name in feasibility_frontier predictive_examples communication_age; do
  cmp "$repo_root/paper/$figure_dir/$name.svg" "$work_dir/paper/$figure_dir/$name.svg"
  rsvg-convert -f pdf "$work_dir/paper/$figure_dir/$name.svg" -o "$work_dir/paper/$figure_dir/$name.pdf"
done
printf 'DETERMINISTIC_SVG_FIGURES=PASS 3/3\n'

(cd "$work_dir/paper/manuscript_tvt" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex >/dev/null)
pages="$(pdfinfo "$work_dir/paper/manuscript_tvt/main.pdf" | awk '/^Pages:/ {print $2}')"
refs="$(grep -c '\\bibitem' "$work_dir/paper/manuscript_tvt/main.bbl")"
if [[ "$pages" != 14 || "$refs" != 28 ]]; then
  printf 'PAPER_BUILD=FAIL pages=%s references=%s\n' "$pages" "$refs" >&2
  exit 1
fi
if grep -Eq 'Overfull|LaTeX Error|undefined' "$work_dir/paper/manuscript_tvt/main.log"; then
  printf 'PAPER_LOG=FAIL\n' >&2
  exit 1
fi
if grep -Eq 'Warning|Error' "$work_dir/paper/manuscript_tvt/main.blg"; then
  printf 'BIBTEX_LOG=FAIL\n' >&2
  exit 1
fi
printf 'PAPER_BUILD=PASS pages=%s references=%s\n' "$pages" "$refs"
