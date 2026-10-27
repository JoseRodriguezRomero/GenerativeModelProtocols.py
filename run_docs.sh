#!/bin/sh
set -eu

PROJECT_ROOT="$(CDPATH= cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-python3}"
DOCS_BUILD_DIR="${DOCS_BUILD_DIR:-${PROJECT_ROOT}/docs/_build/html}"
DOCS_HOST="${DOCS_HOST:-127.0.0.1}"
DOCS_PORT="${DOCS_PORT:-8000}"

if ! "${PYTHON}" -c 'import sphinx, myst_parser' >/dev/null 2>&1; then
    echo "Sphinx and MyST Parser are required. Install them with: ${PYTHON} -m pip install -e '.[docs]'" >&2
    exit 1
fi

cd "${PROJECT_ROOT}"
"${PYTHON}" -m sphinx -E -b html docs "${DOCS_BUILD_DIR}"

echo "Serving documentation at http://${DOCS_HOST}:${DOCS_PORT}/"
cd "${DOCS_BUILD_DIR}"
exec "${PYTHON}" -m http.server "${DOCS_PORT}" --bind "${DOCS_HOST}"
