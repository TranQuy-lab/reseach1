#!/usr/bin/env bash
set -euo pipefail

ACCELERATOR="${ACCELERATOR:-cu128}"
PYTHON_COMMAND="${PYTHON_COMMAND:-python3.12}"
ENV_DIR="${ENV_DIR:-.venv-server}"

"${PYTHON_COMMAND}" -m venv "${ENV_DIR}"
"${ENV_DIR}/bin/python" -m pip install --upgrade pip==25.2
"${ENV_DIR}/bin/python" -m pip install -r research/server/requirements-server-core.txt

if [[ "${ACCELERATOR}" == "cu128" ]]; then
  "${ENV_DIR}/bin/python" -m pip install torch==2.8.0 \
    --index-url https://download.pytorch.org/whl/cu128
  "${ENV_DIR}/bin/python" -m pip install pyg-lib==0.5.0 \
    --find-links https://data.pyg.org/whl/torch-2.8.0+cu128.html
elif [[ "${ACCELERATOR}" == "cpu" ]]; then
  "${ENV_DIR}/bin/python" -m pip install torch==2.8.0+cpu \
    --index-url https://download.pytorch.org/whl/cpu
  "${ENV_DIR}/bin/python" -m pip install pyg-lib==0.5.0 \
    --find-links https://data.pyg.org/whl/torch-2.8.0+cpu.html
else
  echo "ACCELERATOR must be cu128 or cpu" >&2
  exit 2
fi

"${ENV_DIR}/bin/python" -m pip install torch-geometric==2.7.0

"${ENV_DIR}/bin/python" -m ipykernel install --user \
  --name nids-server --display-name "NIDS E-GraphSAGE server"
if [[ "${ACCELERATOR}" == "cu128" ]]; then
  PYTHONPATH=src "${ENV_DIR}/bin/python" research/server/check_server.py \
    --output research/results/server_environment.json --scope full --require-cuda
else
  PYTHONPATH=src "${ENV_DIR}/bin/python" research/server/check_server.py \
    --output research/results/server_environment.json --scope minibatch
fi

echo "Environment ready: ${ENV_DIR} (${ACCELERATOR})"
