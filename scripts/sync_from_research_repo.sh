#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLIC_REPO="$(cd "${SCRIPT_DIR}/.." && pwd)"
RESEARCH_REPO="${1:-${PUBLIC_REPO}/../Imputation_VAE}"

SRC="${RESEARCH_REPO}/imputation_vae"
DST="${PUBLIC_REPO}/graph_tcn_vae"

if [[ ! -d "${SRC}" ]]; then
  echo "Research package not found: ${SRC}" >&2
  exit 1
fi

for file in \
  model.py \
  model_uq.py \
  model_graph_uq.py \
  model_graph_pred.py \
  masking_utils.py \
  utils.py \
  dataset.py
do
  cp "${SRC}/${file}" "${DST}/${file}"
done

perl -0pi -e 's/, see CLAUDE\.md//g' "${DST}/model_graph_uq.py"

python -m pytest -q
