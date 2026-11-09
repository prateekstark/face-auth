#!/usr/bin/env python3
"""
Unit tests for FaceAuth.
Run with: python3 -m unittest discover tests/
"""

import os
import sys
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np

# Ensure parent directory is on python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import face_auth


class TestUsernameValidation(unittest.TestCase):
    def test_valid_usernames(self):
        valid = ["stark", "john_doe", "user-123", "alice", "admin$", "_daemon"]
        for u in valid:
            self.assertTrue(face_auth.validate_username(u), f"Failed on valid username: {u}")

    def test_invalid_usernames(self):
        invalid = [
            "../etc/passwd",
            "../../shadow",
            "stark/test",
            "user;rm -rf /",
            "user\x00name",
            "",
            None,
            123,
            "has space",
            "-leadinghyphen",
        ]
        for u in invalid:
            self.assertFalse(face_auth.validate_username(u), f"Should reject invalid username: {u}")


class TestConfiguration(unittest.TestCase):
    def test_default_config(self):
        cfg = face_auth.get_config()
        self.assertIn("recognition_threshold", cfg)
        self.assertIn("timeout_seconds", cfg)
        self.assertIn("required_matches", cfg)
        self.assertIn("camera_id", cfg)
        self.assertEqual(cfg["camera_id"], 0)

    def test_custom_config(self):
        with tempfile.NamedTemporaryFile("w+", delete=False, suffix=".json") as f:
            json.dump({"camera_id": 2, "timeout_seconds": 5.0, "recognition_threshold": 0.42}, f)
            f_path = f.name

        try:
            cfg = face_auth.get_config(f_path)
            self.assertEqual(cfg["camera_id"], 2)
            self.assertEqual(cfg["timeout_seconds"], 5.0)
            self.assertEqual(cfg["recognition_threshold"], 0.42)
        finally:
            os.unlink(f_path)


class TestEmbeddingMath(unittest.TestCase):
    def test_cosine_similarity(self):
        # Normalized vector
        v1 = np.random.randn(1, 128).astype(np.float32)
        v1 /= np.linalg.norm(v1)

        # Self-similarity should be ~1.0
        dot_self = float(np.dot(v1.flatten(), v1.flatten()))
        self.assertAlmostEqual(dot_self, 1.0, places=5)

        # Orthogonal vector
        v2 = np.zeros_like(v1)
        v2[0, 0] = 1.0
        v3 = np.zeros_like(v1)
        v3[0, 1] = 1.0
        dot_ortho = float(np.dot(v2.flatten(), v3.flatten()))
        self.assertAlmostEqual(dot_ortho, 0.0, places=5)

    def test_batch_matching(self):
        # Create 5 enrolled sample embeddings
        samples = [np.random.randn(128).astype(np.float32) for _ in range(5)]
        samples = [s / np.linalg.norm(s) for s in samples]
        matrix = np.array(samples, dtype=np.float32)

        # Test query identical to sample 2
        query = samples[2].copy()
        pipeline = face_auth.FacePipeline.__new__(face_auth.FacePipeline)
        best_score = pipeline.match_embeddings_batch(query, matrix)
        self.assertAlmostEqual(best_score, 1.0, places=5)


class TestUserManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.mgr = face_auth.UserManager(data_dir=self.data_dir)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_save_and_load_embeddings(self):
        username = "testuser"
        embeddings = [np.random.randn(1, 128).astype(np.float32) for _ in range(3)]
        embeddings = [e / np.linalg.norm(e) for e in embeddings]

        self.mgr.save_embeddings(username, embeddings)
        self.assertTrue(self.mgr.is_enrolled(username))

        emb_list, matrix = self.mgr.load_embeddings(username)
        self.assertIsNotNone(matrix)
        self.assertEqual(len(emb_list), 3)
        self.assertEqual(matrix.shape, (3, 128))

    def test_remove_user(self):
        username = "removeme"
        embeddings = [np.random.randn(1, 128).astype(np.float32)]
        self.mgr.save_embeddings(username, embeddings)
        self.assertTrue(self.mgr.is_enrolled(username))

        removed = self.mgr.remove_user(username)
        self.assertTrue(removed)
        self.assertFalse(self.mgr.is_enrolled(username))

    def test_path_traversal_blocked(self):
        with self.assertRaises(ValueError):
            self.mgr.save_embeddings("../../etc/bad", [])

        self.assertFalse(self.mgr.is_enrolled("../evil"))


class TestEnvironmentSafety(unittest.TestCase):
    def test_remote_session_detection(self):
        # Default environment in test
        os.environ["PAM_SERVICE"] = "sshd"
        self.assertTrue(face_auth.is_remote_session())

        os.environ["PAM_SERVICE"] = "sudo"
        del os.environ["PAM_SERVICE"]
        if "PAM_RHOST" in os.environ:
            del os.environ["PAM_RHOST"]
        if "SSH_CLIENT" in os.environ:
            del os.environ["SSH_CLIENT"]
        if "SSH_TTY" in os.environ:
            del os.environ["SSH_TTY"]
        if "SSH_CONNECTION" in os.environ:
            del os.environ["SSH_CONNECTION"]

        self.assertFalse(face_auth.is_remote_session())


if __name__ == "__main__":
    unittest.main()
