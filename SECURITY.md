# Security & Privacy Policy

## Architecture & Threat Model

**FaceAuth** is designed as a convenience-oriented biometric authentication helper integrated with Linux PAM (Pluggable Authentication Modules). It complements existing authentication methods (such as passwords, smartcards, or hardware tokens) and falls back safely to password authentication whenever verification fails or times out.

### Key Security Principles

1. **Additive & Non-Destructive PAM Stacking**:
   FaceAuth is configured in PAM via `[success=end default=ignore]`. This ensures:
   - If facial verification succeeds, access is granted.
   - If verification fails, times out, or encounters any error, PAM immediately falls through to the next configured authentication module (standard Unix password prompt).
   - FaceAuth **never** locks you out of your system.

2. **Zero Image Retention**:
   - FaceAuth never saves, transmits, or caches raw camera images or video frames to disk.
   - Only high-dimensional mathematical vector embeddings (128-D floating-point arrays) are saved to `/var/lib/face-auth/users/<username>.json`.
   - Stored biometric files are set with restrictive permissions (`0600 root:root`) inside a root-only directory (`0700 root:root`).

3. **Remote Session Protection**:
   FaceAuth explicitly detects remote connections (SSH sessions, SCP, SFTP, XRDP) via PAM environment variables (`PAM_RHOST`, `PAM_SERVICE`, `SSH_CLIENT`, `SSH_TTY`, `SSH_CONNECTION`). When a remote session is detected, facial scanning is immediately bypassed.

4. **Hardware State Awareness**:
   FaceAuth checks ACPI laptop lid switch state. If the laptop lid is closed (e.g. docked or closed in a bag), scanning is skipped immediately without opening camera hardware.

5. **2D Optical Camera Limitations**:
   Standard RGB webcams lack 3D structured light or infrared depth sensors. While FaceAuth includes motion and landmark consistency checks, 2D facial recognition can be susceptible to presentation attacks (photos/screens) in certain lighting conditions. For high-security environments:
   - Adjust `recognition_threshold` in `/etc/face-auth/config.json` to a higher value (e.g., `0.45` to `0.50`).
   - Increase `required_matches` to require more consecutive matching frames.
   - Keep your system password strong and unique.

## Reporting a Vulnerability

If you discover a security vulnerability in FaceAuth, please report it privately rather than opening a public issue. Open an advisory via GitHub Security Advisories or contact the repository maintainers directly.
