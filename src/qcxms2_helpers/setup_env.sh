#!/usr/bin/env bash
# ======================================================
# QCxMS2 Environment Setup Script (no shell init required)
# Works both locally and on HPC (PBS)
# Author: W. Verastegui
# ======================================================

set -e

# --- CONFIGURATION ---
ENV_NAME="qcxmstwo"
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC_DIR="$BASE_DIR/../qcxms2_src_env"

echo "============================================"
echo " Setting up QCxMS2 environment"
echo " Base directory:  $BASE_DIR"
echo " Environment:     $ENV_NAME"
echo "============================================"

# --- LOCATE MAMBA/CONDA BASE ---
if command -v mamba &>/dev/null; then
    MAMBA_BASE=$(mamba info --base 2>/dev/null)
elif command -v conda &>/dev/null; then
    MAMBA_BASE=$(conda info --base 2>/dev/null)
else
    echo "Error: Neither conda nor mamba found. Please install Miniforge or Mambaforge."
    exit 1
fi

# --- LOAD ENVIRONMENT SYSTEM (only if available) ---
if [ -f "$MAMBA_BASE/etc/profile.d/conda.sh" ]; then
    source "$MAMBA_BASE/etc/profile.d/conda.sh"
elif [ -f "$MAMBA_BASE/etc/profile.d/mamba.sh" ]; then
    source "$MAMBA_BASE/etc/profile.d/mamba.sh"
else
    echo "Warning: Could not find conda.sh or mamba.sh in $MAMBA_BASE"
fi

# --- CREATE ENVIRONMENT IF NOT EXISTS ---
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "Environment '$ENV_NAME' not found — creating from environment.yml..."
    if [ -f "$BASE_DIR/environment.yml" ]; then
        (mamba env create -f "$BASE_DIR/environment.yml" -n "$ENV_NAME" || \
         conda env create -f "$BASE_DIR/environment.yml" -n "$ENV_NAME")
    else
        echo "Error: environment.yml not found in $BASE_DIR"
        exit 1
    fi
else
    echo "Environment '$ENV_NAME' already exists."
fi

# --- SET PATHS ---
export ORCA_PATH="$SRC_DIR/orca-6.1.0-f.0_linux_x86-64/bin"
export QCXMS2_PATH="$SRC_DIR/qcxms2_v_1.2.0"

export PATH="$ORCA_PATH:$QCXMS2_PATH:$PATH"
export LD_LIBRARY_PATH="$ORCA_PATH:$LD_LIBRARY_PATH"

# --- TEST ENVIRONMENT (without activation) ---
echo ""
echo "Testing environment..."
mamba run -n "$ENV_NAME" which python || conda run -n "$ENV_NAME" which python

echo ""
echo "Environment '$ENV_NAME' configured successfully."
echo "ORCA path:  $ORCA_PATH"
echo "QCxMS2 path: $QCXMS2_PATH"
echo ""
echo "You can now run commands like:"
echo "   mamba run -n $ENV_NAME qcxms2 input.xyz"
echo "   mamba run -n $ENV_NAME python script.py"
echo "============================================"
