#!/usr/bin/env python3
"""
FaceAuth: Biometric Facial Recognition Authentication for Ubuntu/Linux.
Integrated seamlessly with PAM (Pluggable Authentication Modules)
as a complementary authentication factor without replacing any existing methods.
"""

import os
import sys
import re
import time
import json
import argparse
import datetime
from pathlib import Path
import numpy as np

# Suppress noisy OpenCV/GStreamer warnings
os.environ["OPENCV_LOG_LEVEL"] = "OFF"
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "-8"

try:
    import cv2
except ImportError:
    print("❌ Error: OpenCV (cv2) is not installed.", file=sys.stderr)
    print("   Please install it with: sudo apt install python3-opencv", file=sys.stderr)
    sys.exit(1)

VERSION = "1.0.0"

# Default Paths
SYSTEM_CONF_DIR = Path("/etc/face-auth")
SYSTEM_DATA_DIR = Path("/var/lib/face-auth")
LOCAL_DIR = Path(__file__).resolve().parent

DEFAULT_CONFIG = {
    "camera_id": 0,
    "frame_width": 640,
    "frame_height": 480,
    "detection_threshold": 0.65,
    "recognition_threshold": 0.38,  # SFace cosine similarity threshold
    "timeout_seconds": 3.0,
    "required_matches": 2,          # Require 2 matching frames to avoid false positives
    "warmup_frames": 2,
    "min_face_size": 80,            # Minimum width/height in pixels
    "disable_lid_closed": True,
    "feedback": True,               # User feedback during authentication
    "feedback_tty": True,           # Kept for backward compatibility
    "liveness_check": False,        # Optional motion variance check
    "log_file": "/var/log/face-auth.log"
}


def validate_username(username: str) -> bool:
    """Validate username to prevent path traversal or special character exploits."""
    if not username or not isinstance(username, str):
        return False
    # Standard POSIX username: alphanumeric, underscores, hyphens, optional trailing $
    pattern = r'^[a-zA-Z0-9_.][a-zA-Z0-9_.-]*\$?$'
    return bool(re.match(pattern, username)) and ".." not in username and "/" not in username


def get_config(custom_path=None):
    """Load configuration from custom path, /etc/face-auth/config.json, or local fallback."""
    cfg = dict(DEFAULT_CONFIG)
    conf_path = None
    if custom_path:
        conf_path = Path(custom_path)
    elif (SYSTEM_CONF_DIR / "config.json").exists():
        conf_path = SYSTEM_CONF_DIR / "config.json"
    elif (LOCAL_DIR / "config.json").exists():
        conf_path = LOCAL_DIR / "config.json"

    if conf_path and conf_path.exists():
        try:
            with open(conf_path, "r") as f:
                cfg.update(json.load(f))
        except Exception as e:
            log_msg(f"Config load error: {e}")

    # Synchronize feedback flags
    if "feedback" in cfg:
        cfg["feedback_tty"] = cfg["feedback"]
    elif "feedback_tty" in cfg:
        cfg["feedback"] = cfg["feedback_tty"]

    return cfg


def get_data_dir():
    """Return data directory, avoiding standard binary directories like /usr/local/bin."""
    # 1. System data directory if root or directory is writable
    if os.geteuid() == 0 or (SYSTEM_DATA_DIR.exists() and os.access(SYSTEM_DATA_DIR, os.W_OK)):
        return SYSTEM_DATA_DIR

    # 2. Local repository directory if running directly in dev tree (not installed to /usr/bin)
    if LOCAL_DIR.name not in ("bin", "sbin") and (LOCAL_DIR / "models").exists():
        return LOCAL_DIR

    # 3. User XDG data directory fallback
    xdg_dir = Path.home() / ".local/share/face-auth"
    try:
        xdg_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    return xdg_dir


def get_model_paths():
    """Locate YuNet and SFace ONNX models in standard locations."""
    locations = [
        SYSTEM_DATA_DIR / "models",
        LOCAL_DIR / "models",
        LOCAL_DIR.parent / "models",
        Path.home() / ".local/share/face-auth/models"
    ]
    yunet = None
    sface = None
    for loc in locations:
        y_candidate = loc / "face_detection_yunet.onnx"
        s_candidate = loc / "face_recognition_sface.onnx"
        if not yunet and y_candidate.exists():
            yunet = y_candidate
        if not sface and s_candidate.exists():
            sface = s_candidate
    return yunet, sface


def is_lid_closed():
    """Check if laptop lid is closed to skip face auth immediately."""
    lid_dirs = list(Path("/proc/acpi/button/lid").glob("*/state"))
    for state_file in lid_dirs:
        try:
            with open(state_file, "r") as f:
                if "closed" in f.read().lower():
                    return True
        except Exception:
            pass
    return False


def is_remote_session():
    """Detect if running via SSH or remote session where camera is irrelevant."""
    if os.environ.get("PAM_RHOST"):
        return True
    service = os.environ.get("PAM_SERVICE", "").lower()
    if service in ("sshd", "ssh", "scp", "sftp", "xrdp-sesman", "vnc"):
        return True
    if os.environ.get("SSH_CLIENT") or os.environ.get("SSH_TTY") or os.environ.get("SSH_CONNECTION"):
        return True
    return False


def log_msg(msg: str, config=None):
    """Log formatted entry to log file."""
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted = f"[{timestamp}] [PID {os.getpid()}] {msg}\n"
    try:
        cfg = config or get_config()
        log_path = Path(cfg.get("log_file", "/var/log/face-auth.log"))
        # Only log if directory is writable
        if log_path.parent.exists() and os.access(log_path.parent, os.W_OK):
            with open(log_path, "a") as f:
                f.write(formatted)
    except Exception:
        pass


class FacePipeline:
    def __init__(self, config=None):
        self.config = config or get_config()
        yunet_path, sface_path = get_model_paths()
        if not yunet_path or not yunet_path.exists():
            raise FileNotFoundError(
                f"YuNet face detection model not found. "
                f"Please run scripts/download_models.sh or install to {SYSTEM_DATA_DIR}/models/"
            )
        if not sface_path or not sface_path.exists():
            raise FileNotFoundError(
                f"SFace face recognition model not found. "
                f"Please run scripts/download_models.sh or install to {SYSTEM_DATA_DIR}/models/"
            )

        self.w = int(self.config.get("frame_width", 640))
        self.h = int(self.config.get("frame_height", 480))

        # Initialize YuNet Face Detector
        self.detector = cv2.FaceDetectorYN.create(
            str(yunet_path),
            "",
            (self.w, self.h),
            float(self.config.get("detection_threshold", 0.65)),
            0.3,
            5000
        )

        # Initialize SFace Face Recognizer
        self.recognizer = cv2.FaceRecognizerSF.create(str(sface_path), "")

    def open_camera(self):
        """Open camera with V4L2 backend on Linux for ultra-low startup latency."""
        cam_id = self.config.get("camera_id", 0)
        if isinstance(cam_id, str) and cam_id.isdigit():
            cam_id = int(cam_id)

        cap = None
        # Try V4L2 first on Linux (skips GStreamer probing, opens in ~180ms)
        if sys.platform.startswith("linux") and isinstance(cam_id, int):
            try:
                cap = cv2.VideoCapture(cam_id, cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    cap = None
            except Exception:
                cap = None

        # Fallback to standard backend if V4L2 failed
        if cap is None or not cap.isOpened():
            cap = cv2.VideoCapture(cam_id)

        if not cap.isOpened():
            return None

        # Optimize camera parameters
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.h)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        return cap

    def detect_face(self, frame):
        """Returns faces array from YuNet. faces shape: (N, 15)"""
        h, w, _ = frame.shape
        self.detector.setInputSize((w, h))
        _, faces = self.detector.detect(frame)
        return faces

    def extract_embedding(self, frame, face_box):
        """Aligns face and extracts L2-normalized 128-D embedding vector."""
        aligned = self.recognizer.alignCrop(frame, face_box)
        feat = self.recognizer.feature(aligned)
        # L2 normalization ensures cosine similarity = dot product
        norm = np.linalg.norm(feat)
        if norm > 0:
            feat = feat / norm
        return feat

    def match_embeddings(self, feat1, feat2):
        """Returns cosine similarity score between two normalized embeddings."""
        return float(np.dot(feat1.flatten(), feat2.flatten()))

    def match_embeddings_batch(self, feat, enrolled_matrix):
        """Fast vectorized cosine similarity against all enrolled samples."""
        if enrolled_matrix is None or len(enrolled_matrix) == 0:
            return 0.0
        scores = np.dot(enrolled_matrix, feat.flatten())
        return float(np.max(scores))


class UserManager:
    def __init__(self, data_dir=None):
        self.data_dir = data_dir or get_data_dir()
        self.users_dir = self.data_dir / "users"
        try:
            self.users_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

    def get_user_file(self, username: str) -> Path:
        """Get path to user's profile file with path traversal protection."""
        if not validate_username(username):
            raise ValueError(f"Invalid username: '{username}'")

        # Prefer system location if it exists
        sys_file = SYSTEM_DATA_DIR / "users" / f"{username}.json"
        if sys_file.exists():
            return sys_file
        return self.users_dir / f"{username}.json"

    def is_enrolled(self, username: str) -> bool:
        if not validate_username(username):
            return False
        try:
            return self.get_user_file(username).exists()
        except Exception:
            return False

    def load_embeddings(self, username: str):
        """Loads enrolled embeddings and returns (list_of_arrays, matrix_2d)."""
        if not validate_username(username):
            return None, None
        try:
            u_file = self.get_user_file(username)
            if not u_file.exists():
                return None, None
            with open(u_file, "r") as f:
                data = json.load(f)
            raw = data.get("embeddings", [])
            if not raw:
                return None, None
            emb_list = [np.array(e, dtype=np.float32).reshape(1, -1) for e in raw]
            matrix = np.array(raw, dtype=np.float32)
            # Ensure normalized
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            matrix = matrix / norms
            return emb_list, matrix
        except PermissionError:
            log_msg(f"Permission denied reading user file for '{username}'")
            return None, None
        except Exception as e:
            log_msg(f"Error loading user file for '{username}': {e}")
            return None, None

    def save_embeddings(self, username: str, embeddings):
        """Save embeddings with restrictive file permissions (0600)."""
        if not validate_username(username):
            raise ValueError(f"Invalid username: '{username}'")

        target_dir = self.users_dir
        # If running as root, prefer SYSTEM_DATA_DIR
        if os.geteuid() == 0:
            target_dir = SYSTEM_DATA_DIR / "users"
            target_dir.mkdir(parents=True, exist_ok=True)

        u_file = target_dir / f"{username}.json"
        serializable = [e.flatten().tolist() for e in embeddings]
        payload = {
            "username": username,
            "version": VERSION,
            "created_at": datetime.datetime.now().isoformat(),
            "embeddings_count": len(serializable),
            "embeddings": serializable
        }
        with open(u_file, "w") as f:
            json.dump(payload, f, indent=2)

        # Secure file permissions (rw-------)
        try:
            os.chmod(u_file, 0o600)
        except Exception:
            pass
        return u_file

    def list_users(self):
        """List all enrolled users."""
        users = []
        checked = set()
        search_dirs = [self.users_dir]
        if SYSTEM_DATA_DIR.exists() and SYSTEM_DATA_DIR / "users" != self.users_dir:
            search_dirs.append(SYSTEM_DATA_DIR / "users")

        for d in search_dirs:
            if not d.exists():
                continue
            try:
                for p in d.glob("*.json"):
                    if p.stem not in checked and validate_username(p.stem):
                        checked.add(p.stem)
                        try:
                            with open(p, "r") as f:
                                data = json.load(f)
                            users.append({
                                "username": data.get("username", p.stem),
                                "count": data.get("embeddings_count", len(data.get("embeddings", []))),
                                "created_at": data.get("created_at", "unknown"),
                                "path": str(p)
                            })
                        except Exception:
                            pass
            except PermissionError:
                pass
        return users

    def remove_user(self, username: str) -> bool:
        if not validate_username(username):
            return False
        removed = False
        search_dirs = [self.users_dir]
        if SYSTEM_DATA_DIR.exists():
            search_dirs.append(SYSTEM_DATA_DIR / "users")

        for d in search_dirs:
            p = d / f"{username}.json"
            if p.exists():
                try:
                    p.unlink()
                    removed = True
                except Exception:
                    pass
        return removed


# =====================================================================
# Commands
# =====================================================================

def cmd_enroll(args):
    """Enroll a user by capturing diverse face samples."""
    username = args.username or os.environ.get("SUDO_USER") or os.environ.get("USER")
    if not username:
        print("❌ Error: Could not determine username to enroll.", file=sys.stderr)
        return 1

    if not validate_username(username):
        print(f"❌ Error: Invalid username '{username}'.", file=sys.stderr)
        return 1

    # Check root privileges if system installation exists
    if SYSTEM_DATA_DIR.exists() and os.geteuid() != 0:
        print(f"❌ Error: Administrator privileges required to enroll biometrics for system login.")
        print(f"   Please run with sudo: sudo face-auth enroll {username}")
        return 1

    print("=" * 65)
    print(f"  FaceAuth: Enrolling Facial Biometrics for '{username}'")
    print("=" * 65)
    print("Instructions:")
    print("  1. Look directly into your webcam.")
    print("  2. Ensure your face is well-lit and unobstructed (no masks or sunglasses).")
    print("  3. Follow the onscreen prompts to slightly vary your angle/expression.")
    print("=" * 65)

    if sys.stdin.isatty():
        try:
            input("Press [Enter] when ready to start camera capture...")
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return 1
    else:
        print("Non-interactive mode: Starting camera capture in 2 seconds...")
        time.sleep(2)

    try:
        pipeline = FacePipeline()
    except Exception as e:
        print(f"❌ Error initializing face pipeline: {e}")
        return 1

    cap = pipeline.open_camera()
    if not cap:
        print(f"❌ Error: Unable to access camera device {pipeline.config['camera_id']}.")
        return 1

    # Discard warmup frames
    for _ in range(pipeline.config.get("warmup_frames", 2)):
        cap.read()

    prompts = [
        "1/8: Look straight at camera with neutral expression",
        "2/8: Look straight with a natural smile",
        "3/8: Turn head slightly to the left (5-10°)",
        "4/8: Turn head slightly to the right (5-10°)",
        "5/8: Tilt chin slightly up",
        "6/8: Tilt chin slightly down",
        "7/8: Look straight with slight angle variation",
        "8/8: Final confirmation sample"
    ]

    enrolled_embeddings = []
    min_size = int(pipeline.config.get("min_face_size", 80))

    has_gui = not args.no_gui and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    window_name = "FaceAuth Enrollment Wizard"
    if has_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 640, 480)
        except Exception:
            has_gui = False

    print("\nStarting camera capture...\n")

    try:
        for i, prompt in enumerate(prompts):
            print(f"[*] Step {prompt}...")
            sample_captured = False
            start_step = time.time()

            while not sample_captured:
                ret, frame = cap.read()
                if not ret:
                    time.sleep(0.03)
                    continue

                faces = pipeline.detect_face(frame)
                best_face = None

                if faces is not None and len(faces) > 0:
                    max_area = 0
                    for face in faces:
                        w, h = face[2], face[3]
                        if w >= min_size and h >= min_size and (w * h) > max_area:
                            max_area = w * h
                            best_face = face

                if has_gui:
                    display_frame = frame.copy()
                    # Banner overlay
                    cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], 50), (30, 30, 30), -1)
                    cv2.putText(
                        display_frame,
                        f"Step {i+1}/8: {prompt.split(': ')[1]}",
                        (15, 32),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2
                    )

                    if best_face is not None:
                        bx, by, bw, bh = [int(v) for v in best_face[:4]]
                        cv2.rectangle(display_frame, (bx, by), (bx + bw, by + bh), (0, 255, 0), 2)
                        cv2.putText(
                            display_frame,
                            "Face Detected - Hold Still",
                            (bx, max(by - 10, 65)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            (0, 255, 0),
                            2
                        )
                    else:
                        cv2.putText(
                            display_frame,
                            "No Face Detected",
                            (20, 90),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (0, 0, 255),
                            2
                        )

                    cv2.imshow(window_name, display_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord('q'), 27):
                        print("\nEnrollment cancelled by user.")
                        return 1

                # Capture condition: stable face detected
                if best_face is not None:
                    feat = pipeline.extract_embedding(frame, best_face)
                    enrolled_embeddings.append(feat)
                    sample_captured = True
                    print(f"    ✓ Captured sample {i + 1}/{len(prompts)}")
                    time.sleep(0.4)
                    break

                if time.time() - start_step > 10.0:
                    print("    ⚠ Timeout waiting for face. Moving to next sample...")
                    break
                time.sleep(0.03)

    except KeyboardInterrupt:
        print("\nEnrollment interrupted.")
        return 1
    finally:
        cap.release()
        if has_gui:
            cv2.destroyAllWindows()

    if len(enrolled_embeddings) < 3:
        print(f"\n❌ Enrollment failed: Only {len(enrolled_embeddings)} valid samples were captured.")
        print("Please ensure your camera is functioning and your face is well-lit.")
        return 1

    user_mgr = UserManager()
    saved_path = user_mgr.save_embeddings(username, enrolled_embeddings)

    print(f"\n🎉 Successfully enrolled {len(enrolled_embeddings)} biometric samples for '{username}'!")
    print(f"   Stored at: {saved_path}")
    return 0


def cmd_verify(args):
    """
    PAM Verification Command.
    Fast, non-interactive authentication verification executed by pam_exec.
    Returns 0 on match (PAM_SUCCESS).
    Returns 1 on failure/timeout (PAM_AUTH_ERR -> falls back to password).
    """
    # Prefer PAM_USER if invoked by PAM, else args or environment
    username = os.environ.get("PAM_USER") or args.username or os.environ.get("USER")
    if not username or not validate_username(username):
        log_msg("Verify aborted: Invalid or missing username")
        sys.exit(1)

    # Remote session safety check
    if is_remote_session():
        log_msg(f"Verify bypassed: Remote session detected for {username}")
        sys.exit(1)

    config = get_config(args.config if hasattr(args, 'config') else None)

    # Lid check
    if config.get("disable_lid_closed", True) and is_lid_closed():
        log_msg("Verify bypassed: Laptop lid is closed")
        sys.exit(1)

    user_mgr = UserManager()
    emb_list, enrolled_matrix = user_mgr.load_embeddings(username)
    if enrolled_matrix is None or len(enrolled_matrix) == 0:
        # User not enrolled: fail silently and immediately so password prompt appears
        sys.exit(1)

    # In PAM mode, stdout is captured by pam_exec to display messages to the user
    show_feedback = config.get("feedback", config.get("feedback_tty", True)) and not getattr(args, "quiet", False)
    if show_feedback:
        print(f"📷 [FaceAuth] Scanning face for '{username}'...", flush=True)

    try:
        pipeline = FacePipeline(config)
    except Exception as e:
        log_msg(f"Pipeline init error: {e}")
        sys.exit(1)

    cap = pipeline.open_camera()
    if not cap:
        log_msg(f"Camera open failed on {config['camera_id']}")
        sys.exit(1)

    # Warmup frames
    for _ in range(int(config.get("warmup_frames", 2))):
        cap.read()

    timeout = float(config.get("timeout_seconds", 3.0))
    threshold = float(config.get("recognition_threshold", 0.38))
    required_matches = int(config.get("required_matches", 2))
    min_size = int(config.get("min_face_size", 80))
    liveness_enabled = bool(config.get("liveness_check", False))

    match_count = 0
    start_time = time.time()
    verified = False
    prev_landmarks = None
    motion_detected = not liveness_enabled

    while (time.time() - start_time) < timeout:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.02)
            continue

        faces = pipeline.detect_face(frame)
        if faces is not None and len(faces) > 0:
            for face in faces:
                if face[2] >= min_size and face[3] >= min_size:
                    # Optional anti-spoofing motion check
                    if liveness_enabled and not motion_detected:
                        curr_landmarks = face[4:14].reshape(-1, 2)
                        if prev_landmarks is not None:
                            diff = np.mean(np.linalg.norm(curr_landmarks - prev_landmarks, axis=1))
                            if diff > 0.8:  # Micro-movement detected
                                motion_detected = True
                        prev_landmarks = curr_landmarks

                    feat = pipeline.extract_embedding(frame, face)
                    best_score = pipeline.match_embeddings_batch(feat, enrolled_matrix)

                    if best_score >= threshold:
                        match_count += 1
                        if match_count >= required_matches and motion_detected:
                            verified = True
                            log_msg(f"Authentication SUCCESS for {username} (score={best_score:.3f})")
                            break
                    else:
                        match_count = max(0, match_count - 1)
        if verified:
            break
        time.sleep(0.02)

    cap.release()

    if verified:
        if show_feedback:
            print(f"✓ [FaceAuth] Verified! Welcome, {username}.", flush=True)
        sys.exit(0)
    else:
        log_msg(f"Authentication TIMEOUT/FAILED for {username}")
        if show_feedback:
            print("[FaceAuth] Face not recognized. Falling back to password...", flush=True)
        sys.exit(1)


def cmd_test(args):
    """Interactive visual/console test of camera and face recognition."""
    username = args.username or os.environ.get("SUDO_USER") or os.environ.get("USER")
    config = get_config(args.config if hasattr(args, 'config') else None)
    user_mgr = UserManager()
    emb_list, enrolled_matrix = user_mgr.load_embeddings(username) if username else (None, None)

    print("=" * 65)
    print(f"  FaceAuth Live Diagnostic Test (Target User: '{username}')")
    print("=" * 65)
    if enrolled_matrix is not None:
        print(f"  Biometric Profile : Enrolled ({len(enrolled_matrix)} samples loaded)")
    else:
        print(f"  Biometric Profile : NOT enrolled or unreadable (Face detection mode only)")
        if os.geteuid() != 0 and SYSTEM_DATA_DIR.exists():
            print(f"  Tip               : Run 'sudo face-auth test' to inspect system-enrolled profiles")
    print(f"  Camera Device     : {config['camera_id']}")
    print(f"  Threshold         : {config['recognition_threshold']}")
    print("=" * 65)
    print("Starting camera feed... Press 'q' or [Esc] in window (or Ctrl+C) to exit.")

    try:
        pipeline = FacePipeline(config)
    except Exception as e:
        print(f"❌ Error initializing face pipeline: {e}")
        return 1

    cap = pipeline.open_camera()
    if not cap:
        print("❌ Error: Could not open camera device.")
        return 1

    has_gui = not args.no_gui and bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    threshold = float(config["recognition_threshold"])
    window_name = "FaceAuth Diagnostic Feed"

    if has_gui:
        try:
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        except Exception:
            has_gui = False

    try:
        prev_time = time.time()
        while True:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.02)
                continue

            curr_time = time.time()
            fps = 1.0 / max(curr_time - prev_time, 0.001)
            prev_time = curr_time

            faces = pipeline.detect_face(frame)
            status_text = "No face detected"
            box_color = (0, 0, 255)  # Red

            if faces is not None and len(faces) > 0:
                for face in faces:
                    x, y, w, h = int(face[0]), int(face[1]), int(face[2]), int(face[3])
                    landmarks = [
                        (int(face[4]), int(face[5])),
                        (int(face[6]), int(face[7])),
                        (int(face[8]), int(face[9])),
                        (int(face[10]), int(face[11])),
                        (int(face[12]), int(face[13]))
                    ]

                    match_score = 0.0
                    is_match = False
                    if enrolled_matrix is not None:
                        feat = pipeline.extract_embedding(frame, face)
                        match_score = pipeline.match_embeddings_batch(feat, enrolled_matrix)
                        is_match = match_score >= threshold

                    if is_match:
                        box_color = (0, 255, 0)  # Green
                        status_text = f"MATCH: {username} ({match_score:.2f} >= {threshold})"
                    else:
                        box_color = (0, 165, 255) if enrolled_matrix is not None else (255, 255, 0)
                        status_text = (
                            f"Score: {match_score:.2f} (Need >= {threshold})"
                            if enrolled_matrix is not None
                            else "Face Detected"
                        )

                    if has_gui:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), box_color, 2)
                        for pt in landmarks:
                            cv2.circle(frame, pt, 3, (0, 255, 255), -1)
                        cv2.putText(
                            frame, status_text, (x, max(y - 10, 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, box_color, 2
                        )

            if has_gui:
                cv2.putText(
                    frame, f"FPS: {fps:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2
                )
                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
            else:
                sys.stdout.write(f"\r[Diagnostic] FPS: {fps:.1f} | Status: {status_text:<40}")
                sys.stdout.flush()
                time.sleep(0.05)

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        if has_gui:
            cv2.destroyAllWindows()
        print("\nDiagnostic test finished.")
    return 0


def cmd_list(args):
    """List all enrolled users."""
    mgr = UserManager()
    users = mgr.list_users()

    print("=" * 68)
    print(f"{'USERNAME':<20} {'SAMPLES':<10} {'ENROLLED AT':<25} {'STORAGE'}")
    print("-" * 68)
    if not users:
        print("  (No enrolled users found)")
        if os.geteuid() != 0 and SYSTEM_DATA_DIR.exists():
            print("\n  Note: System user profiles are protected. Run 'sudo face-auth list' to view.")
    else:
        for u in users:
            loc = "System" if "/var/lib/face-auth" in u.get("path", "") else "Local"
            print(f"{u['username']:<20} {u['count']:<10} {u['created_at'][:19]:<25} {loc}")
    print("=" * 68)
    return 0


def cmd_remove(args):
    """Remove enrolled user biometric data."""
    if not args.username:
        print("❌ Error: Specify username to remove (e.g. sudo face-auth remove stark)", file=sys.stderr)
        return 1

    if not validate_username(args.username):
        print(f"❌ Error: Invalid username '{args.username}'.", file=sys.stderr)
        return 1

    if SYSTEM_DATA_DIR.exists() and os.geteuid() != 0:
        print("❌ Error: Administrator privileges required to remove system biometric profiles.")
        print(f"   Please run with sudo: sudo face-auth remove {args.username}")
        return 1

    mgr = UserManager()
    if mgr.remove_user(args.username):
        print(f"✓ Removed enrolled facial biometric data for '{args.username}'.")
        return 0
    else:
        print(f"User '{args.username}' is not enrolled.")
        return 1


def cmd_status(args):
    """Check system health, camera, models, and PAM hooks."""
    config = get_config(args.config if hasattr(args, 'config') else None)
    print("=" * 65)
    print(f"  FaceAuth v{VERSION} System Diagnostics")
    print("=" * 65)

    # 1. Models
    yunet, sface = get_model_paths()
    print("[1] Biometric Models:")
    yunet_ok = yunet and yunet.exists()
    sface_ok = sface and sface.exists()
    print(f"    - YuNet Detector : {'✓ Found' if yunet_ok else '❌ Missing'} ({yunet or 'Not found'})")
    print(f"    - SFace Recognizer: {'✓ Found' if sface_ok else '❌ Missing'} ({sface or 'Not found'})")

    # 2. Camera Device
    cam_id = config.get("camera_id", 0)
    print(f"\n[2] Camera Device ({cam_id}):")
    try:
        pipeline = FacePipeline(config)
        cap = pipeline.open_camera()
        if cap and cap.isOpened():
            ret, frame = cap.read()
            cap.release()
            if ret:
                print(f"    ✓ Camera initialized successfully ({frame.shape[1]}x{frame.shape[0]} px, low-latency backend)")
            else:
                print("    ⚠ Camera opened but frame read failed")
        else:
            print(f"    ❌ Camera {cam_id} failed to open")
    except Exception as e:
        print(f"    ❌ Camera check error: {e}")

    # 3. Enrolled Users
    mgr = UserManager()
    users = mgr.list_users()
    print(f"\n[3] Enrolled Users ({len(users)}):")
    for u in users:
        print(f"    - {u['username']} ({u['count']} biometric samples)")
    if not users:
        if os.geteuid() != 0 and SYSTEM_DATA_DIR.exists():
            print("    (Run 'sudo face-auth status' to inspect protected system user profiles)")
        else:
            print("    (None enrolled yet. Run: sudo face-auth enroll)")

    # 4. PAM Integration
    pam_config_path = Path("/usr/share/pam-configs/face-auth")
    pam_common_auth = Path("/etc/pam.d/common-auth")
    has_pam_config = pam_config_path.exists()
    has_hook = False
    if pam_common_auth.exists():
        try:
            with open(pam_common_auth, "r") as f:
                has_hook = "face-auth" in f.read()
        except Exception:
            pass

    print("\n[4] PAM Configuration:")
    print(f"    - PAM Profile (/usr/share/pam-configs/face-auth): {'✓ Installed' if has_pam_config else 'Not installed'}")
    print(f"    - PAM Active in common-auth: {'✓ Active (Face Login Enabled)' if has_hook else 'Not active'}")

    print("=" * 65)
    return 0


def main():
    # If invoked directly as face-auth-verify or by PAM
    invoked_name = Path(sys.argv[0]).name
    if invoked_name in ("face-auth-verify", "pam_face_verify"):
        parser = argparse.ArgumentParser()
        parser.add_argument("username", nargs="?", default=None)
        parser.add_argument("--config", default=None)
        parser.add_argument("--quiet", action="store_true", default=False)
        args, _ = parser.parse_known_args()
        cmd_verify(args)
        return

    parser = argparse.ArgumentParser(
        description=f"FaceAuth v{VERSION}: Biometric Facial Recognition Authentication for Ubuntu/Linux"
    )
    parser.add_argument("-v", "--version", action="version", version=f"FaceAuth v{VERSION}")
    parser.add_argument("-c", "--config", help="Custom path to config.json", default=None)

    subparsers = parser.add_subparsers(dest="command")

    # enroll
    p_enroll = subparsers.add_parser("enroll", help="Enroll face biometric samples")
    p_enroll.add_argument("username", nargs="?", default=None, help="User to enroll (default: current user)")
    p_enroll.add_argument("--no-gui", action="store_true", help="Force headless console mode")

    # verify
    p_verify = subparsers.add_parser("verify", help="Verify user face (used by PAM)")
    p_verify.add_argument("username", nargs="?", default=None, help="User to verify")
    p_verify.add_argument("--quiet", action="store_true", help="Suppress terminal feedback")

    # test
    p_test = subparsers.add_parser("test", help="Live diagnostic camera feed and match test")
    p_test.add_argument("username", nargs="?", default=None, help="User to test against")
    p_test.add_argument("--no-gui", action="store_true", help="Force console-only FPS/status output")

    # list
    subparsers.add_parser("list", help="List enrolled users")

    # remove
    p_remove = subparsers.add_parser("remove", help="Remove user face data")
    p_remove.add_argument("username", help="User to remove")

    # status
    subparsers.add_parser("status", help="Show system and PAM status")

    args = parser.parse_args()

    if args.command == "enroll":
        sys.exit(cmd_enroll(args))
    elif args.command == "verify":
        cmd_verify(args)
    elif args.command == "test":
        sys.exit(cmd_test(args))
    elif args.command == "list":
        sys.exit(cmd_list(args))
    elif args.command == "remove":
        sys.exit(cmd_remove(args))
    elif args.command == "status":
        sys.exit(cmd_status(args))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
