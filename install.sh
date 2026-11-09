#!/usr/bin/env bash
set -e

# FaceAuth Installer for Ubuntu/Debian
# Non-invasive PAM integration complementing existing authentication methods.

if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: Please run this installer with sudo: sudo ./install.sh"
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=========================================================="
echo "  Installing FaceAuth (Biometric Face Login for Linux)   "
echo "=========================================================="

# 1. Dependency Checks & Auto-Install
echo "[1/6] Checking system dependencies..."
MISSING_PKGS=()

if ! dpkg -s python3-opencv >/dev/null 2>&1; then
    MISSING_PKGS+=("python3-opencv")
fi
if ! dpkg -s python3-numpy >/dev/null 2>&1; then
    MISSING_PKGS+=("python3-numpy")
fi
if ! dpkg -s libpam-runtime >/dev/null 2>&1; then
    MISSING_PKGS+=("libpam-runtime")
fi
if ! command -v curl >/dev/null 2>&1 && ! command -v wget >/dev/null 2>&1; then
    MISSING_PKGS+=("curl")
fi

if [ ${#MISSING_PKGS[@]} -gt 0 ]; then
    echo "       Installing required packages: ${MISSING_PKGS[*]}..."
    apt-get update -qq
    apt-get install -y --no-install-recommends "${MISSING_PKGS[@]}"
else
    echo "       ✓ All prerequisites met (Python 3, OpenCV, NumPy, PAM)."
fi

# 2. Check & Download AI Models if Missing
echo "[2/6] Verifying biometric models..."
YUNET_FILE="$SCRIPT_DIR/models/face_detection_yunet.onnx"
SFACE_FILE="$SCRIPT_DIR/models/face_recognition_sface.onnx"

if [ ! -f "$YUNET_FILE" ] || [ ! -f "$SFACE_FILE" ]; then
    echo "       Models missing in repo. Downloading from official OpenCV Zoo..."
    bash "$SCRIPT_DIR/scripts/download_models.sh" "$SCRIPT_DIR/models"
else
    echo "       ✓ Biometric models found in repo."
fi

# 3. Create System Directories
echo "[3/6] Setting up system directories..."
mkdir -p /etc/face-auth
mkdir -p /var/lib/face-auth/models
mkdir -p /var/lib/face-auth/users
chmod 755 /etc/face-auth
chmod 755 /var/lib/face-auth
chmod 755 /var/lib/face-auth/models
chmod 700 /var/lib/face-auth/users

# 4. Install Configuration & Models
echo "[4/6] Deploying configuration & AI models..."
if [ ! -f /etc/face-auth/config.json ]; then
    cp "$SCRIPT_DIR/config.json" /etc/face-auth/config.json
    chmod 644 /etc/face-auth/config.json
else
    echo "       (Existing /etc/face-auth/config.json preserved)"
fi

cp "$SCRIPT_DIR/models/"*.onnx /var/lib/face-auth/models/
chmod 644 /var/lib/face-auth/models/*.onnx

# 5. Install Executables & Symlinks
echo "[5/6] Installing executables to /usr/local/bin..."
cp "$SCRIPT_DIR/face_auth.py" /usr/local/bin/face-auth
chmod 755 /usr/local/bin/face-auth
ln -sf /usr/local/bin/face-auth /usr/local/bin/face-auth-verify

# Migrate local biometric profile if one exists in the repo
if [ -n "$SUDO_USER" ] && [ -d "$SCRIPT_DIR/users" ]; then
    if [ -f "$SCRIPT_DIR/users/$SUDO_USER.json" ]; then
        echo "       Migrating local biometric profile for '$SUDO_USER'..."
        cp "$SCRIPT_DIR/users/$SUDO_USER.json" "/var/lib/face-auth/users/$SUDO_USER.json"
        chmod 600 "/var/lib/face-auth/users/$SUDO_USER.json"
    fi
fi

# 6. Install PAM Module Configuration
echo "[6/6] Registering PAM authentication hook..."
mkdir -p /usr/share/pam-configs
cp "$SCRIPT_DIR/pam-config/face-auth" /usr/share/pam-configs/face-auth
chmod 644 /usr/share/pam-configs/face-auth
pam-auth-update --package

echo "=========================================================="
echo "  ✓ FaceAuth Installation & PAM Integration Complete!     "
echo "=========================================================="
echo ""
echo "Next steps:"
echo "  1. Enroll your face:"
if [ -n "$SUDO_USER" ]; then
    echo "     sudo face-auth enroll"
else
    echo "     sudo face-auth enroll <username>"
fi
echo ""
echo "  2. Test in diagnostic camera preview:"
echo "     sudo face-auth test"
echo ""
echo "  3. Test PAM authentication in a new terminal:"
echo "     sudo echo 'Face authentication successful!'"
echo ""
echo "  (Note: Passwords, PINs, and fingerprints remain 100% active"
echo "   as seamless fallbacks whenever face scan times out)."
echo "=========================================================="
