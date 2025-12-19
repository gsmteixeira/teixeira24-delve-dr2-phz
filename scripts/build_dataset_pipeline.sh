#!/usr/bin/env bash
set -e 

echo "Building MCAT..."
python3 -m scripts.build_mcat

echo "Building SZCAT..."
python3 -m scripts.build_szcat

echo "Building DLCAT..."
python3 -m scripts.build_dlcat

echo "Pipeline completed successfully."