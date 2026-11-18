#!/usr/bin/env bash
# One-time setup inside the hip-gpt2-nv container: build the engine and export GPT-2 weights.
set -e
git config --global --add safe.directory /work
if [ ! -x /build/build-nv/gpt2 ]; then
  echo "Building the engine (one time)..."
  cmake -S . -B /build/build-nv -DCMAKE_HIP_PLATFORM=nvidia -DCMAKE_HIP_ARCHITECTURES=89 > /dev/null
  cmake --build /build/build-nv -j 8 --target gpt2 > /dev/null
fi
if [ ! -f models/gpt2/model.safetensors ]; then
  echo "Downloading GPT-2 and exporting weights (one time)..."
  [ -d /venv/bin ] || python3 -m venv /venv
  . /venv/bin/activate
  pip install -q --index-url https://download.pytorch.org/whl/cu126 torch
  pip install -q -r tools/requirements.txt
  python3 tools/export_reference_logits.py --out models/gpt2 --prompt "hi" --ref /tmp/ref.bin > /dev/null
fi
