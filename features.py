"""Hand landmark detection (MediaPipe) and feature normalization.

Every sign is turned into 63 numbers (21 landmarks x [x, y, z]) that are:
  * translated so the wrist is the origin   -> position independent
  * scaled by the hand size                 -> distance-from-camera independent
  * mirrored for left hands                 -> one model works for either hand
"""
import numpy as np

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def normalize_landmarks(landmarks, is_left: bool = False) -> np.ndarray:
    """landmarks: (21, 3) array-like of raw x, y, z -> flat (63,) float32 vector."""
    pts = np.asarray(landmarks, dtype=np.float32).reshape(NUM_LANDMARKS, 3).copy()
    pts -= pts[0]                      # wrist becomes (0, 0, 0)
    if is_left:
        pts[:, 0] *= -1                # mirror left hand into a right hand
    scale = float(np.max(np.linalg.norm(pts[:, :2], axis=1)))
    pts /= scale if scale > 1e-6 else 1.0
    return pts.flatten()


class HandDetector:
    """Thin wrapper around MediaPipe Hands. Frames should be mirrored (selfie view)."""

    def __init__(self, static_image_mode=False, detection_conf=0.6, tracking_conf=0.5):
        import mediapipe as mp  # imported lazily so tests don't need a camera stack

        self._mp_hands = mp.solutions.hands
        self._drawing = mp.solutions.drawing_utils
        self._styles = mp.solutions.drawing_styles
        self._hands = self._mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=detection_conf,
            min_tracking_confidence=tracking_conf,
        )

    def process(self, frame_bgr):
        """Returns None if no hand, else dict(features, landmarks, bbox)."""
        import cv2

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None
        hand = result.multi_hand_landmarks[0]
        is_left = result.multi_handedness[0].classification[0].label == "Left"
        pts = np.array([[p.x, p.y, p.z] for p in hand.landmark], dtype=np.float32)
        h, w = frame_bgr.shape[:2]
        xs, ys = pts[:, 0] * w, pts[:, 1] * h
        pad = 20
        bbox = (int(max(xs.min() - pad, 0)), int(max(ys.min() - pad, 0)),
                int(min(xs.max() + pad, w - 1)), int(min(ys.max() + pad, h - 1)))
        return {"features": normalize_landmarks(pts, is_left), "landmarks": hand, "bbox": bbox}

    def draw(self, frame_bgr, landmarks):
        self._drawing.draw_landmarks(
            frame_bgr, landmarks, self._mp_hands.HAND_CONNECTIONS,
            self._styles.get_default_hand_landmarks_style(),
            self._styles.get_default_hand_connections_style())

    def close(self):
        self._hands.close()
