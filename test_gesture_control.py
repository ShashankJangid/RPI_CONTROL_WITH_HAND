import os
import sys
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hardware_controller import HardwareController
from hand_detector import GestureType, HandDetector


class MockPoint:
    def __init__(self, x, y, z=0.0):
        self.x = x
        self.y = y
        self.z = z


def create_synthetic_landmarks(thumb_up=False, thumb_down=False, thumb_extended=False,
                               index=False, middle=False, ring=False, pinky=False):
    lm = [MockPoint(0.5, 0.8) for _ in range(21)]
    wrist = lm[0]

    lm[1] = MockPoint(0.42, 0.72)
    lm[2] = MockPoint(0.38, 0.65)
    lm[3] = MockPoint(0.34, 0.58)

    if thumb_up:
        lm[4] = MockPoint(0.36, 0.45)
    elif thumb_down:
        lm[1] = MockPoint(0.42, 0.75)
        lm[2] = MockPoint(0.38, 0.82)
        lm[3] = MockPoint(0.36, 0.88)
        lm[4] = MockPoint(0.35, 0.95)
    elif thumb_extended:
        lm[4] = MockPoint(0.20, 0.55)
    else:
        lm[4] = MockPoint(0.45, 0.68)

    mcp_indices = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    finger_x = [0.45, 0.50, 0.55, 0.60]
    finger_states = [index, middle, ring, pinky]

    for (mcp_i, pip_i, dip_i, tip_i), x, is_extended in zip(mcp_indices, finger_x, finger_states):
        lm[mcp_i] = MockPoint(x, 0.55)
        if is_extended:
            lm[pip_i] = MockPoint(x, 0.40)
            lm[dip_i] = MockPoint(x, 0.30)
            lm[tip_i] = MockPoint(x, 0.18)
        else:
            lm[pip_i] = MockPoint(x, 0.45)
            lm[dip_i] = MockPoint(x, 0.52)
            lm[tip_i] = MockPoint(x, 0.58)

    return lm


class TestGestureControl(unittest.TestCase):

    def setUp(self):
        self.hardware = HardwareController()

    def tearDown(self):
        self.hardware.cleanup()

    def test_hardware_controller_fan(self):
        self.assertFalse(self.hardware.get_status()["fan"])
        changed = self.hardware.set_fan(True)
        self.assertTrue(changed)
        self.assertTrue(self.hardware.get_status()["fan"])

        self.assertFalse(self.hardware.set_fan(True))

        changed = self.hardware.set_fan(False)
        self.assertTrue(changed)
        self.assertFalse(self.hardware.get_status()["fan"])

    def test_hardware_controller_light(self):
        self.assertFalse(self.hardware.get_status()["light"])
        self.assertTrue(self.hardware.set_light(True))
        self.assertTrue(self.hardware.get_status()["light"])
        self.assertTrue(self.hardware.set_light(False))
        self.assertFalse(self.hardware.get_status()["light"])

    def test_hardware_controller_gate(self):
        self.assertFalse(self.hardware.get_status()["gate"])
        self.assertTrue(self.hardware.set_gate(True))
        self.assertTrue(self.hardware.get_status()["gate"])
        self.assertTrue(self.hardware.set_gate(False))
        self.assertFalse(self.hardware.get_status()["gate"])

    def test_gesture_classification_one_finger(self):
        lm = create_synthetic_landmarks(index=True, middle=False, ring=False, pinky=False, thumb_extended=False)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.ONE_FINGER)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_gesture_classification_two_fingers(self):
        lm = create_synthetic_landmarks(index=True, middle=True, ring=False, pinky=False, thumb_extended=False)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.TWO_FINGERS)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_gesture_classification_three_fingers(self):
        lm = create_synthetic_landmarks(index=True, middle=True, ring=True, pinky=False, thumb_extended=False)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.THREE_FINGERS)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_gesture_classification_four_fingers(self):
        lm = create_synthetic_landmarks(index=True, middle=True, ring=True, pinky=True, thumb_extended=False)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.FOUR_FINGERS)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_gesture_classification_five_fingers(self):
        lm = create_synthetic_landmarks(thumb_extended=True, index=True, middle=True, ring=True, pinky=True)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.FIVE_FINGERS)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_gesture_classification_closed_fist(self):
        lm = create_synthetic_landmarks(thumb_extended=False, index=False, middle=False, ring=False, pinky=False)
        try:
            detector = HandDetector(initialize_mediapipe=False)
            gesture = detector.classify_gesture(lm)
            self.assertEqual(gesture, GestureType.CLOSED_FIST)
            detector.close()
        except ImportError:
            self.skipTest("mediapipe not installed on this test environment")

    def test_debounce_and_hardware_triggering(self):
        from main import GestureApp
        import config

        class DummyCamera:
            backend = "Dummy"
            def stop(self): pass
            def is_opened(self): return True
            def read(self): return None

        class DummyDetector:
            def close(self): pass

        app = GestureApp(
            headless=True,
            hardware=self.hardware,
            camera=DummyCamera(),
            detector=DummyDetector()
        )

        self.assertFalse(self.hardware.get_status()["fan"])

        for _ in range(5):
            app.process_gesture(GestureType.ONE_FINGER)

        self.assertFalse(self.hardware.get_status()["fan"])

        for _ in range(config.GESTURE_CONFIRM_FRAMES - 5):
            app.process_gesture(GestureType.ONE_FINGER)

        self.assertTrue(self.hardware.get_status()["fan"])

        app.last_action_time = 0.0
        for _ in range(config.GESTURE_CONFIRM_FRAMES):
            app.process_gesture(GestureType.TWO_FINGERS)

        self.assertFalse(self.hardware.get_status()["fan"])


if __name__ == "__main__":
    unittest.main()
