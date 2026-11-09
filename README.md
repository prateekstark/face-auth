# FaceAuth: Linux Facial Recognition Authentication

[![CI](https://github.com/prateekstark/face-auth/actions/workflows/ci.yml/badge.svg)](https://github.com/prateekstark/face-auth/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Platform: Linux](https://img.shields.io/badge/Platform-Linux%20%7C%20Ubuntu-orange.svg)](#)
[![Python: 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](#)

**FaceAuth** brings Windows Hello-style biometric facial recognition to Ubuntu/Debian Linux without replacing or tampering with any of your existing authentication methods (password, fingerprint, smartcard).

Powered by modern deep-learning computer vision models (**YuNet** face detection + **SFace** 128-dimensional biometric embeddings) through OpenCV, it runs natively on Linux with zero third-party PPA instability and ultra-low latency verification (<200ms camera capture via V4L2).

---

## Architecture & How It Works

```
                        [Authentication Trigger]
                     (GDM3 Login / Lock Screen / sudo)
                                   │
                                   ▼
                       [PAM: /etc/pam.d/common-auth]
                                   │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
     [Priority 265: FaceAuth]                [Priority 256: Unix Auth]
       /usr/local/bin/face-auth-verify          /lib/security/pam_unix.so
                │                                       │
      ┌─────────┴─────────┐                             │
      ▼                   ▼                             ▼
 [Face Matched]    [Timeout / Unrecognized]     [Standard Password Prompt]
    (Exit 0)            (Exit 1)                        │
      │                   │                             ▼
      ▼                   └──────────────────────► [User Enters Password]
 [Logged In Instantly!]
```

- **Additive & Non-Invasive**: Installed with `[success=end default=ignore]` via Ubuntu's official `pam-auth-update` system.
- **Fail-Safe Fallback**: If your webcam is covered, you are away from the camera, or in complete darkness, PAM seamlessly defaults to your normal password prompt after a 2-3s timeout.
- **Remote / SSH Protection**: Automatically disables facial scanning during SSH sessions, SCP/SFTP, and remote sessions (`PAM_RHOST`, `SSH_CLIENT`, `SSH_TTY`, `SSH_CONNECTION`).
- **Laptop Lid Detection**: Skips facial scan immediately if the laptop lid is closed (e.g. docked or closed in a backpack).
- **Privacy First**: Only 128-dimensional mathematical vector embeddings are stored in `/var/lib/face-auth/users/` (permissions `0600 root:root`). No raw images or photos are ever saved to disk.
- **Ultra-Fast V4L2 Camera Initialization**: Directly queries Linux Video4Linux2 backend, opening camera streams in ~180ms.

---

## Prerequisites

FaceAuth requires standard Ubuntu/Debian packages:
```bash
sudo apt update
sudo apt install -y python3-opencv python3-numpy libpam-runtime curl
```
*(The installer will automatically detect and install any missing prerequisites).*

---

## Quick Setup

### Step 1: Clone & Install System Hook
```bash
git clone https://github.com/prateekstark/face-auth.git
cd face-auth
sudo ./install.sh
```
This registers the PAM profile in `/usr/share/pam-configs/face-auth`, deploys the models to `/var/lib/face-auth/models/`, and installs `/usr/local/bin/face-auth`.

### Step 2: Enroll Your Face
Run the interactive enrollment wizard:
```bash
sudo face-auth enroll
```
Follow the on-screen prompts to capture 8 diverse facial poses (straight, slight smile, slight left, slight right, chin tilt). If a desktop GUI is available, a visual camera preview window with facial bounding box and landmarks will open automatically.

### Step 3: Test Real-Time Camera Feed & Verification
Open a diagnostic visual preview:
```bash
sudo face-auth test
```
A live window will open showing your webcam feed, green bounding box, 5-point facial landmarks, and live similarity score against your enrolled profile.

### Step 4: Test Login / Sudo
Open a new terminal window and test:
```bash
sudo echo "Authentication successful!"
```
The camera will scan your face and immediately grant access without asking for your password!

---

## CLI Reference

| Command | Description |
| :--- | :--- |
| `sudo face-auth enroll [user]` | Enrolls/updates face biometric samples for a user |
| `sudo face-auth test [user]` | Real-time diagnostic camera preview showing landmarks & match score |
| `face-auth verify [user]` | Fast verification hook executed by PAM (exit 0 on match, 1 on fail) |
| `sudo face-auth list` | Lists all enrolled users and sample counts |
| `sudo face-auth remove <user>` | Deletes facial biometric profile for a user |
| `face-auth status` | Inspects camera, AI models, user database, and PAM status |
| `face-auth -v, --version` | Displays current FaceAuth version |

---

## Configuration (`/etc/face-auth/config.json`)

You can tune verification parameters in `/etc/face-auth/config.json`:

```json
{
  "camera_id": 0,
  "frame_width": 640,
  "frame_height": 480,
  "detection_threshold": 0.65,
  "recognition_threshold": 0.38,
  "timeout_seconds": 3.0,
  "required_matches": 2,
  "warmup_frames": 2,
  "min_face_size": 80,
  "disable_lid_closed": true,
  "feedback": true,
  "liveness_check": false,
  "log_file": "/var/log/face-auth.log"
}
```

- `recognition_threshold`: Default `0.38` (OpenCV SFace cosine similarity threshold). Increase (e.g. `0.45`) for higher security, decrease (e.g. `0.35`) for more lenient matching.
- `timeout_seconds`: Default `3.0` (How long the camera scans before falling back to password).
- `required_matches`: Default `2` (Number of matching frames required to authenticate).
- `liveness_check`: Default `false`. Set to `true` to require micro-motion detection (anti-spoofing against static printed photos).
- `feedback`: Default `true`. Prints "Scanning face..." and "Verified!" feedback during PAM authentication.
- `camera_id`: Default `0` (`/dev/video0`). Can also be a device path like `"/dev/video2"`.

---

## Running Unit Tests

Run the test suite with:
```bash
python3 -m unittest discover tests/
```

---

## Uninstallation

To cleanly remove FaceAuth and restore previous PAM configuration:
```bash
sudo ./uninstall.sh
```
To purge all stored user face profiles and configuration:
```bash
sudo ./uninstall.sh --purge
```

---

## Contributing

Pull requests and issues are welcome! Please make sure to run unit tests before submitting PRs.

## License

This project is licensed under the [MIT License](LICENSE).
Biometric models (YuNet and SFace) are provided under the Apache 2.0 license by OpenCV Zoo.
