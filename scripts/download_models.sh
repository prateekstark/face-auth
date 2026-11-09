#!/usr/bin/env bash
set -e

# Download OpenCV Zoo ONNX models for FaceAuth (YuNet face detector + SFace recognizer)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${1:-"$SCRIPT_DIR/../models"}"
mkdir -p "$TARGET_DIR"

YUNET_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
SFACE_URL="https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx"

download_file() {
    local url="$1"
    local dest="$2"
    local name="$(basename "$dest")"

    if [ -f "$dest" ] && [ -s "$dest" ]; then
        echo "  ✓ $name already exists, skipping download."
        return 0
    fi

    echo "  ⬇ Downloading $name..."
    if command -v curl >/dev/null 2>&1; then
        curl -fL --progress-bar -o "$dest.tmp" "$url"
    elif command -v wget >/dev/null 2>&1; then
        wget --show-progress -q -O "$dest.tmp" "$url"
    else
        echo "❌ Error: Neither curl nor wget is available." >&2
        return 1
    fi

    mv "$dest.tmp" "$dest"
    echo "  ✓ Downloaded $name successfully."
}

echo "=========================================================="
echo "  Downloading AI Biometric Models for FaceAuth           "
echo "=========================================================="

download_file "$YUNET_URL" "$TARGET_DIR/face_detection_yunet.onnx"
download_file "$SFACE_URL" "$TARGET_DIR/face_recognition_sface.onnx"

echo "  All models are ready in $TARGET_DIR"
