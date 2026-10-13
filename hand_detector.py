import os
import math
import zipfile
import config

try:
    import numpy as np
except ImportError:
    np = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

try:
    import ai_edge_litert.interpreter as litert
    LITERT_AVAILABLE = True
except ImportError:
    try:
        import tflite_runtime.interpreter as litert
        LITERT_AVAILABLE = True
    except ImportError:
        litert = None
        LITERT_AVAILABLE = False


class GestureType:
    NONE = "NONE"
    ONE_FINGER = "ONE_FINGER"
    TWO_FINGERS = "TWO_FINGERS"
    THREE_FINGERS = "THREE_FINGERS"
    FOUR_FINGERS = "FOUR_FINGERS"
    FIVE_FINGERS = "FIVE_FINGERS"
    CLOSED_FIST = "CLOSED_FIST"


class LandmarkPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17)
]


class NeuralHandDetector:
    def __init__(self, model_dir=None, roi_box=(config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2)):
        self.roi_box = roi_box
        if model_dir is None:
            model_dir = os.path.dirname(os.path.abspath(__file__))

        self.model_path = os.path.join(model_dir, "hand_landmarks_detector.tflite")
        self.task_path = os.path.join(model_dir, "hand_landmarker.task")

        if not os.path.exists(self.model_path) and os.path.exists(self.task_path):
            try:
                with zipfile.ZipFile(self.task_path, "r") as z:
                    z.extract("hand_landmarks_detector.tflite", model_dir)
            except Exception as e:
                print("Could not extract model:", e)

        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Hand landmark model not found at {self.model_path}")

        self.interp = litert.Interpreter(model_path=self.model_path, num_threads=4)
        self.interp.allocate_tensors()
        self.input_details = self.interp.get_input_details()
        self.output_details = self.interp.get_output_details()

        self.WRIST = 0
        self.THUMB_CMC = 1
        self.THUMB_MCP = 2
        self.THUMB_IP = 3
        self.THUMB_TIP = 4
        self.INDEX_MCP = 5
        self.INDEX_PIP = 6
        self.INDEX_DIP = 7
        self.INDEX_TIP = 8
        self.MIDDLE_MCP = 9
        self.MIDDLE_PIP = 10
        self.MIDDLE_DIP = 11
        self.MIDDLE_TIP = 12
        self.RING_MCP = 13
        self.RING_PIP = 14
        self.RING_DIP = 15
        self.RING_TIP = 16
        self.PINKY_MCP = 17
        self.PINKY_PIP = 18
        self.PINKY_DIP = 19
        self.PINKY_TIP = 20

    @staticmethod
    def _calc_dist(p1, p2):
        return math.hypot(p1.x - p2.x, p1.y - p2.y)

    def detect(self, frame, draw=True):
        if frame is None or not OPENCV_AVAILABLE or np is None:
            return frame, GestureType.NONE

        x1, y1, x2, y2 = self.roi_box
        h, w, _ = frame.shape
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Show Hand in Box", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame, GestureType.NONE

        roi_resized = cv2.resize(roi, (224, 224))
        rgb = cv2.cvtColor(roi_resized, cv2.COLOR_BGR2RGB)
        input_tensor = (rgb.astype(np.float32) / 255.0)[np.newaxis, ...]

        self.interp.set_tensor(self.input_details[0]["index"], input_tensor)
        self.interp.invoke()

        presence_score = float(self.interp.get_tensor(self.output_details[1]["index"])[0, 0])

        if presence_score < 0.60:
            return frame, GestureType.NONE

        raw_landmarks = self.interp.get_tensor(self.output_details[0]["index"]).reshape(21, 3)

        landmarks = []
        roi_w = x2 - x1
        roi_h = y2 - y1

        for pt in raw_landmarks:
            landmarks.append(LandmarkPoint(
                x=float(pt[0]) / 224.0,
                y=float(pt[1]) / 224.0,
                z=float(pt[2]) / 224.0
            ))

        gesture = self.classify_gesture(landmarks)

        if draw:
            pts_frame = []
            for lm in landmarks:
                px = int(x1 + lm.x * roi_w)
                py = int(y1 + lm.y * roi_h)
                pts_frame.append((px, py))

            for p_start, p_end in HAND_CONNECTIONS:
                cv2.line(frame, pts_frame[p_start], pts_frame[p_end], (255, 200, 0), 2)

            for idx, (px, py) in enumerate(pts_frame):
                if idx in (4, 8, 12, 16, 20):
                    cv2.circle(frame, (px, py), 6, (0, 0, 255), -1)
                else:
                    cv2.circle(frame, (px, py), 4, (0, 255, 255), -1)

            cv2.putText(frame, f"AI Hand: {int(presence_score * 100)}%", (x1 + 5, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return frame, gesture

    def get_finger_states(self, lm):
        wrist = lm[self.WRIST]

        index_extended = self._calc_dist(lm[self.INDEX_TIP], wrist) > (self._calc_dist(lm[self.INDEX_PIP], wrist) * 1.12)
        middle_extended = self._calc_dist(lm[self.MIDDLE_TIP], wrist) > (self._calc_dist(lm[self.MIDDLE_PIP], wrist) * 1.12)
        ring_extended = self._calc_dist(lm[self.RING_TIP], wrist) > (self._calc_dist(lm[self.RING_PIP], wrist) * 1.12)
        pinky_extended = self._calc_dist(lm[self.PINKY_TIP], wrist) > (self._calc_dist(lm[self.PINKY_PIP], wrist) * 1.12)

        thumb_extended = self._calc_dist(lm[self.THUMB_TIP], lm[self.PINKY_MCP]) > (self._calc_dist(lm[self.THUMB_IP], lm[self.PINKY_MCP]) * 1.12)

        return {
            "thumb": thumb_extended,
            "index": index_extended,
            "middle": middle_extended,
            "ring": ring_extended,
            "pinky": pinky_extended
        }

    def classify_gesture(self, landmarks):
        if not landmarks or len(landmarks) < 21:
            return GestureType.NONE

        fingers = self.get_finger_states(landmarks)

        non_thumb_count = sum([fingers["index"], fingers["middle"], fingers["ring"], fingers["pinky"]])
        total_count = non_thumb_count + (1 if fingers["thumb"] else 0)

        if total_count == 0 or (non_thumb_count == 0 and not fingers["thumb"]):
            return GestureType.CLOSED_FIST

        if total_count == 5 or (non_thumb_count == 4 and fingers["thumb"]):
            return GestureType.FIVE_FINGERS

        if non_thumb_count == 4 and not fingers["thumb"]:
            return GestureType.FOUR_FINGERS

        if total_count == 3 or (non_thumb_count == 3 and not fingers["thumb"]):
            return GestureType.THREE_FINGERS

        if total_count == 2 or (non_thumb_count == 2 and not fingers["thumb"]):
            return GestureType.TWO_FINGERS

        if total_count == 1 or (non_thumb_count == 1 and not fingers["thumb"]):
            return GestureType.ONE_FINGER

        return GestureType.NONE


class OpenCVHandDetector:
    def __init__(self, roi_box=(config.ROI_X1, config.ROI_Y1, config.ROI_X2, config.ROI_Y2),
                 min_area=config.MIN_CONTOUR_AREA):
        self.roi_box = roi_box
        self.min_area = min_area
        if np is not None:
            self.lower_ycrcb = np.array([0, 133, 77], dtype=np.uint8)
            self.upper_ycrcb = np.array([255, 173, 127], dtype=np.uint8)
            self.lower_hsv = np.array([0, 30, 50], dtype=np.uint8)
            self.upper_hsv = np.array([25, 255, 255], dtype=np.uint8)
            self.kernel = np.ones((5, 5), np.uint8)
        else:
            self.lower_ycrcb = None
            self.upper_ycrcb = None
            self.lower_hsv = None
            self.upper_hsv = None
            self.kernel = None

    def process_frame(self, frame, draw=True):
        if frame is None or not OPENCV_AVAILABLE or np is None:
            return frame, GestureType.NONE, None

        x1, y1, x2, y2 = self.roi_box
        h, w, _ = frame.shape
        x1, x2 = max(0, min(x1, w)), max(0, min(x2, w))
        y1, y2 = max(0, min(y1, h)), max(0, min(y2, h))

        if draw:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, "Show Hand in Box", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)

        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            return frame, GestureType.NONE, None

        blurred = cv2.GaussianBlur(roi, (7, 7), 0)
        ycrcb = cv2.cvtColor(blurred, cv2.COLOR_BGR2YCrCb)
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_and(
            cv2.inRange(ycrcb, self.lower_ycrcb, self.upper_ycrcb),
            cv2.inRange(hsv, self.lower_hsv, self.upper_hsv)
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel, iterations=2)
        _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gesture = GestureType.NONE

        if contours:
            max_c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(max_c) >= self.min_area:
                gesture = self._classify(max_c)
                if draw:
                    cv2.drawContours(frame, [max_c + [x1, y1]], -1, (255, 0, 0), 2)

        return frame, gesture, mask

    def _classify(self, contour):
        hull = cv2.convexHull(contour, returnPoints=False)
        if hull is None or len(hull) < 3:
            return GestureType.NONE

        bx, by, bw, bh = cv2.boundingRect(contour)
        topmost = tuple(contour[contour[:, :, 1].argmin()][0])
        bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return GestureType.NONE
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])

        defects_cnt = 0
        try:
            d = cv2.convexityDefects(contour, hull)
            if d is not None:
                for i in range(d.shape[0]):
                    s, e, f, depth = d[i, 0]
                    if depth > (15 * 256):
                        defects_cnt += 1
        except Exception:
            pass

        if defects_cnt >= 4 or (bw > 120 and bh > 120 and defects_cnt >= 3):
            return GestureType.FIVE_FINGERS
        if defects_cnt == 3:
            return GestureType.FOUR_FINGERS
        if defects_cnt == 2:
            return GestureType.THREE_FINGERS
        if defects_cnt == 1 and bh > bw * 1.1:
            return GestureType.TWO_FINGERS
        if defects_cnt == 0 and bh > bw * 1.25 and (cy - topmost[1] > 60):
            return GestureType.ONE_FINGER
        if defects_cnt == 0 and cv2.contourArea(contour) > 6500:
            return GestureType.CLOSED_FIST

        return GestureType.NONE


class HandDetector:
    def __init__(self, detector_type=config.DETECTOR_TYPE, initialize_mediapipe=True):
        self.detector_type = detector_type.lower()
        self.neural_detector = None
        self.opencv_detector = OpenCVHandDetector()
        self.active_backend = "OpenCV (Contour)"

        if LITERT_AVAILABLE and self.detector_type != "opencv_only":
            try:
                self.neural_detector = NeuralHandDetector()
                self.active_backend = "LiteRT 21-Landmarks"
                print(">> [VISION] Initialized LiteRT Deep Neural Network 21-Hand-Landmark Engine.")
            except Exception as e:
                print(f">> [VISION] Neural detector initialization failed ({e}). Falling back to OpenCV.")
                self.neural_detector = None
                self.active_backend = "OpenCV (Contour)"
        else:
            self.active_backend = "OpenCV (Contour)"

    def detect(self, frame, draw=True):
        if self.neural_detector is not None:
            try:
                return self.neural_detector.detect(frame, draw=draw)
            except Exception:
                pass

        annotated_frame, gesture, _ = self.opencv_detector.process_frame(frame, draw=draw)
        return annotated_frame, gesture

    def classify_gesture(self, landmarks):
        if self.neural_detector is not None:
            return self.neural_detector.classify_gesture(landmarks)
        dummy = NeuralHandDetector.__new__(NeuralHandDetector)
        dummy.WRIST = 0
        dummy.THUMB_MCP = 2
        dummy.THUMB_IP = 3
        dummy.THUMB_TIP = 4
        dummy.INDEX_PIP = 6
        dummy.INDEX_TIP = 8
        dummy.MIDDLE_PIP = 10
        dummy.MIDDLE_TIP = 12
        dummy.RING_PIP = 14
        dummy.RING_TIP = 16
        dummy.PINKY_MCP = 17
        dummy.PINKY_PIP = 18
        dummy.PINKY_TIP = 20
        return NeuralHandDetector.classify_gesture(dummy, landmarks)

    def close(self):
        pass
