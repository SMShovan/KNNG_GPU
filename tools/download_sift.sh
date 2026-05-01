#!/usr/bin/env bash
#
# tools/download_sift.sh
#
# Download SIFT-small (10K base vectors, 128-d, plus 100 queries and
# the corresponding 100×100 ground truth) from the IRISA `corpus-texmex`
# benchmark site. Files land under `datasets/siftsmall/`, which is
# gitignored.
#
# The full SIFT1M (1M base vectors) and GIST1M downloads will get
# their own scripts; SIFT-small is the right-sized working set for the
# Mac dev box (every Phase-1 step that needs a dataset can use it).
#
# Usage:
#   ./tools/download_sift.sh                    # downloads if missing
#   FORCE=1 ./tools/download_sift.sh            # re-download even if present
#   DEST=/some/path ./tools/download_sift.sh    # override destination root
#
# Exit codes:
#   0 — files present (already there or successfully downloaded)
#   1 — could not download (network error, missing curl/wget, etc.)
#   2 — downloaded archive failed checksum / extraction
#
# This script does not invoke any C++ tooling — it is a pure
# provisioning helper. The build system never depends on it.

set -euo pipefail

readonly SIFT_URL="ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz"
readonly SIFT_TARBALL="siftsmall.tar.gz"
readonly SIFT_DIR_NAME="siftsmall"

dest_root="${DEST:-$(cd "$(dirname "$0")/.." && pwd)/datasets}"
target_dir="${dest_root}/${SIFT_DIR_NAME}"

if [[ -d "${target_dir}" && -z "${FORCE:-}" ]]; then
    echo "siftsmall: already present at ${target_dir} — skipping download"
    echo "           (set FORCE=1 to re-download)"
    exit 0
fi

mkdir -p "${dest_root}"
cd "${dest_root}"

echo "siftsmall: downloading ${SIFT_URL} → ${dest_root}/${SIFT_TARBALL}"
if command -v curl >/dev/null 2>&1; then
    curl --fail --location --output "${SIFT_TARBALL}" "${SIFT_URL}"
elif command -v wget >/dev/null 2>&1; then
    wget --output-document="${SIFT_TARBALL}" "${SIFT_URL}"
else
    echo "siftsmall: neither curl nor wget is on PATH" >&2
    exit 1
fi

echo "siftsmall: extracting ${SIFT_TARBALL}"
tar -xzf "${SIFT_TARBALL}"
rm -f "${SIFT_TARBALL}"

if [[ ! -f "${target_dir}/siftsmall_base.fvecs" ]]; then
    echo "siftsmall: extraction failed — base.fvecs missing" >&2
    exit 2
fi

echo "siftsmall: ready at ${target_dir}"
echo "           base   : ${target_dir}/siftsmall_base.fvecs"
echo "           query  : ${target_dir}/siftsmall_query.fvecs"
echo "           ground : ${target_dir}/siftsmall_groundtruth.ivecs"
