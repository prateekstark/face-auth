#!/usr/bin/env bash
set -e

# FaceAuth Uninstaller for Ubuntu/Debian
# Completely and safely removes PAM hooks and restores standard configuration.

if [ "$EUID" -ne 0 ]; then
  echo "❌ Error: Please run this uninstaller with sudo: sudo ./uninstall.sh"
  exit 1
fi

PURGE=false
for arg in "$@"; do
    case "$arg" in
        --purge|-y|--yes)
            PURGE=true
            ;;
    esac
done

echo "=========================================================="
echo "  Uninstalling FaceAuth from Ubuntu                       "
echo "=========================================================="

# 1. Disable PAM Profile
echo "[1/4] Safely removing PAM hook..."
if [ -f /usr/share/pam-configs/face-auth ]; then
    pam-auth-update --package --remove face-auth || true
    rm -f /usr/share/pam-configs/face-auth
    pam-auth-update --package
fi

# 2. Remove Executables
echo "[2/4] Removing binaries from /usr/local/bin..."
rm -f /usr/local/bin/face-auth
rm -f /usr/local/bin/face-auth-verify

# 3. Clean up config & data
echo "[3/4] Biometric database and configuration cleanup..."
if [ "$PURGE" = true ]; then
    rm -rf /var/lib/face-auth
    rm -rf /etc/face-auth
    rm -f /var/log/face-auth.log
    echo "       Purged /var/lib/face-auth, /etc/face-auth, and logs."
elif [ -t 0 ]; then
    read -p "Do you want to delete stored biometric face data in /var/lib/face-auth? [y/N]: " choice
    case "$choice" in 
      y|Y ) 
        rm -rf /var/lib/face-auth
        rm -rf /etc/face-auth
        rm -f /var/log/face-auth.log
        echo "       Deleted /var/lib/face-auth and /etc/face-auth"
        ;;
      * ) 
        echo "       Retained face profiles in /var/lib/face-auth for future use."
        ;;
    esac
else
    echo "       (Non-interactive: Retained /var/lib/face-auth. Use --purge to remove data)."
fi

echo "[4/4] System restored."
echo "=========================================================="
echo "  ✓ FaceAuth has been completely uninstalled.            "
echo "=========================================================="
