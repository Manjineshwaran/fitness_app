# MediaPipe logic for pose estimation (Not UI code)
# Optimizations: model_complexity=0 + adaptive smoothing + threading

import cv2
import math
import threading
import queue
import mediapipe as mp

mp_pose = mp.solutions.pose


class PoseDetector:
    def __init__(self, alpha: float = 0.4, model_complexity: int = 0):
        """
        alpha: landmark smoothing strength.
        Higher alpha follows previous landmarks more (smoother but less responsive).
        model_complexity: MediaPipe pose model complexity (0, 1, 2).
        """
        safe_complexity = max(0, min(int(model_complexity), 2))
        try:
            self.pose = mp_pose.Pose(
                model_complexity=safe_complexity,  # 0,1,2
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.7
            )
        except PermissionError:
            # Some cloud hosts mount site-packages read-only, and MediaPipe may try
            # to download extra models for complexity 0/2 into that directory.
            # Fall back to complexity 1, which avoids that download path.
            self.pose = mp_pose.Pose(
                model_complexity=1,
                smooth_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.7
            )
        self.prev_landmarks = None
        # Clamp to a safe range to avoid extreme behavior.
        self.alpha = max(0.05, min(float(alpha), 0.95))  # base smoothing (lower = more responsive)

        # Threading: separate camera capture from MediaPipe processing
        self.frame_queue = queue.Queue(maxsize=1)   # only keep latest frame
        self.result_queue = queue.Queue(maxsize=1)  # only keep latest result
        self._last_annotated = None                 # fallback if no result yet

        self.thread = threading.Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    # ─── Background thread: runs MediaPipe without blocking camera ───────────

    def _process_loop(self):
        while True:
            frame = self.frame_queue.get()          # blocks until a frame arrives
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb)

            annotated = frame.copy()

            if results.pose_landmarks:
                smoothed = self._smooth_landmarks(results.pose_landmarks.landmark)
                for i, lm in enumerate(smoothed):
                    results.pose_landmarks.landmark[i].x = lm.x
                    results.pose_landmarks.landmark[i].y = lm.y
                    results.pose_landmarks.landmark[i].z = lm.z

                mp.solutions.drawing_utils.draw_landmarks(
                    annotated,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            # Swap out old result so queue never backs up
            if self.result_queue.full():
                try:
                    self.result_queue.get_nowait()
                except queue.Empty:
                    pass
            self.result_queue.put(annotated)

    # ─── Adaptive smoothing: less alpha when moving fast ─────────────────────

    def _smooth_landmarks(self, landmarks):
        if self.prev_landmarks is None:
            self.prev_landmarks = list(landmarks)
            return list(landmarks)

        smoothed = []
        for prev, curr in zip(self.prev_landmarks, landmarks):
            dx = curr.x - prev.x
            dy = curr.y - prev.y
            speed = math.sqrt(dx * dx + dy * dy)

            # Fast movement → alpha drops toward 0.1 (follows you)
            # Slow movement → alpha stays near self.alpha (smooths jitter)
            alpha = max(0.1, self.alpha - speed * 10)

            curr.x = alpha * prev.x + (1 - alpha) * curr.x
            curr.y = alpha * prev.y + (1 - alpha) * curr.y
            curr.z = alpha * prev.z + (1 - alpha) * curr.z
            smoothed.append(curr)

        self.prev_landmarks = smoothed
        return smoothed

    # ─── Called by app.py every frame ────────────────────────────────────────

    def process_frame(self, frame):
        # Send to background thread (drop if still busy — never wait)
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                pass
        self.frame_queue.put(frame.copy())

        # Return latest annotated frame if ready, else last known or raw
        if not self.result_queue.empty():
            self._last_annotated = self.result_queue.get_nowait()

        return self._last_annotated if self._last_annotated is not None else frame


if __name__ == "__main__":
    detector = PoseDetector()
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = detector.process_frame(frame)
        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()