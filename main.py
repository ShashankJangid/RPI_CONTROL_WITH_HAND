import os
import sys
import time
import argparse
import logging
import signal

if sys.platform.startswith("linux"):
    if "DISPLAY" not in os.environ:
        os.environ["DISPLAY"] = ":0"
    if "XDG_RUNTIME_DIR" not in os.environ:
        os.environ["XDG_RUNTIME_DIR"] = f"/run/user/{os.getuid()}"
    if "WAYLAND_DISPLAY" not in os.environ:
        os.environ["WAYLAND_DISPLAY"] = "wayland-0"

try:
    import cv2
except ImportError:
    cv2 = None

import config
from camera_stream import CameraStream
from hand_detector import HandDetector, GestureType
from hardware_controller import HardwareController

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("Main")


class GestureApp:
    def __init__(self, headless=False, camera_type=config.CAMERA_TYPE, camera_index=config.CAMERA_INDEX,
                 hardware=None, camera=None, detector=None):
        self.headless = headless
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.running = True

        if not self.headless and cv2 is not None:
            try:
                cv2.namedWindow("Raspberry Pi Hand Gesture Control", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("Raspberry Pi Hand Gesture Control", config.FRAME_WIDTH, config.FRAME_HEIGHT)
                logger.info("GUI window initialized on display: %s", os.environ.get("DISPLAY", "default"))
            except Exception as e:
                logger.warning("Could not open GUI window (%s). Falling back to headless mode.", e)
                self.headless = True

        self.hardware = hardware if hardware is not None else HardwareController()

        if camera is not None:
            self.camera = camera
        else:
            logger.info("Starting camera stream...")
            self.camera = CameraStream(
                camera_type=self.camera_type,
                camera_index=self.camera_index,
                width=config.FRAME_WIDTH,
                height=config.FRAME_HEIGHT,
                fps=config.FPS
            )

        if detector is not None:
            self.detector = detector
        else:
            logger.info("Initializing Hand Detector...")
            self.detector = HandDetector()

        self.current_candidate_gesture = GestureType.NONE
        self.consecutive_frames = 0
        self.action_latched = False
        self.last_action_time = 0.0
        self.last_triggered_action = "System Ready"
        self.last_triggered_time = time.time()

        self.prev_time = time.time()
        self.fps = 0.0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, sig, frame):
        logger.info("Shutdown signal received.")
        self.running = False

    def process_gesture(self, gesture):
        now = time.time()

        if gesture == GestureType.NONE:
            self.current_candidate_gesture = GestureType.NONE
            self.consecutive_frames = 0
            self.action_latched = False
            return

        if gesture == self.current_candidate_gesture:
            if not self.action_latched:
                self.consecutive_frames += 1
        else:
            self.current_candidate_gesture = gesture
            self.consecutive_frames = 1
            self.action_latched = False

        if (not self.action_latched and
                self.consecutive_frames >= config.GESTURE_CONFIRM_FRAMES and
                (now - self.last_action_time) > config.ACTION_COOLDOWN_SEC):

            action_desc = self._execute_action(gesture)
            if action_desc:
                self.last_triggered_action = action_desc
                self.last_triggered_time = now
                self.last_action_time = now
                logger.info("Action Triggered: %s", action_desc)

            self.action_latched = True

    def _execute_action(self, gesture):
        if gesture == GestureType.ONE_FINGER:
            self.hardware.set_fan(True)
            return "1 Finger: Fan turned ON"
        elif gesture == GestureType.TWO_FINGERS:
            self.hardware.set_fan(False)
            return "2 Fingers: Fan turned OFF"
        elif gesture == GestureType.THREE_FINGERS:
            self.hardware.set_light(True)
            return "3 Fingers: Light turned ON"
        elif gesture == GestureType.FOUR_FINGERS:
            self.hardware.set_light(False)
            return "4 Fingers: Light turned OFF"
        elif gesture == GestureType.FIVE_FINGERS:
            self.hardware.set_gate(True)
            return "5 Fingers: Gate OPENED"
        elif gesture == GestureType.CLOSED_FIST:
            self.hardware.set_gate(False)
            return "Closed Fist: Gate CLOSED"
        return None

    def draw_hud(self, frame, gesture):
        h, w, _ = frame.shape
        status = self.hardware.get_status()

        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, 110), (20, 20, 20), -1)
        cv2.rectangle(overlay, (0, h - 35), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.65, frame, 0.35, 0, frame)

        fan_color = (0, 255, 0) if status["fan"] else (0, 0, 200)
        fan_text = f"FAN: {'ON' if status['fan'] else 'OFF'}"
        cv2.circle(frame, (20, 25), 8, fan_color, -1)
        cv2.putText(frame, fan_text, (36, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        light_color = (0, 255, 255) if status["light"] else (0, 0, 200)
        light_text = f"LIGHT: {'ON' if status['light'] else 'OFF'}"
        cv2.circle(frame, (170, 25), 8, light_color, -1)
        cv2.putText(frame, light_text, (186, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        gate_color = (255, 180, 0) if status["gate"] else (0, 0, 200)
        gate_text = f"GATE: {'OPEN' if status['gate'] else 'CLOSED'}"
        cv2.circle(frame, (330, 25), 8, gate_color, -1)
        cv2.putText(frame, gate_text, (346, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        gesture_names = {
            GestureType.NONE: "No Gesture",
            GestureType.ONE_FINGER: "1 Finger (Fan ON)",
            GestureType.TWO_FINGERS: "2 Fingers (Fan OFF)",
            GestureType.THREE_FINGERS: "3 Fingers (Light ON)",
            GestureType.FOUR_FINGERS: "4 Fingers (Light OFF)",
            GestureType.FIVE_FINGERS: "5 Fingers (Gate OPEN)",
            GestureType.CLOSED_FIST: "Fist Closed (Gate CLOSE)",
        }
        gesture_display = gesture_names.get(gesture, gesture)
        cv2.putText(frame, f"Gesture: {gesture_display}", (20, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)

        bar_x, bar_y, bar_w, bar_h = 20, 80, 220, 12
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (80, 80, 80), 1)
        if gesture != GestureType.NONE:
            fill_pct = min(1.0, self.consecutive_frames / float(config.GESTURE_CONFIRM_FRAMES))
            fill_w = int(bar_w * fill_pct)
            fill_color = (0, 255, 0) if fill_pct >= 1.0 else (0, 200, 255)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + bar_h), fill_color, -1)

        time_since_trigger = time.time() - self.last_triggered_time
        if time_since_trigger < 3.0:
            cv2.putText(frame, f">> {self.last_triggered_action}", (260, 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        pipe_name = getattr(self.detector, 'active_backend', 'OpenCV')
        footer_text = f"FPS: {self.fps:.1f} | {pipe_name}"
        if status.get("simulation_mode"):
            footer_text += " [MOCK]"
        cv2.putText(frame, footer_text, (15, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 180), 1)

        credit_text = "Developed by AI & Robotics Team in DPSI"
        cv2.putText(frame, credit_text, (w - 325, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 255, 255), 1)

        return frame

    def run(self):
        logger.info("Gesture recognition system is running. Press 'q' to exit.")

        while self.running:
            frame = self.camera.read()
            if frame is None:
                time.sleep(0.01)
                continue

            frame = cv2.flip(frame, 1)

            annotated_frame, detected_gesture = self.detector.detect(frame, draw=not self.headless)

            self.process_gesture(detected_gesture)

            curr_time = time.time()
            time_diff = curr_time - self.prev_time
            if time_diff > 0:
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / time_diff)
            self.prev_time = curr_time

            if not self.headless:
                display_frame = self.draw_hud(annotated_frame, detected_gesture)
                cv2.imshow("Raspberry Pi Hand Gesture Control", display_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == 27:
                    logger.info("User requested exit.")
                    break

        self.cleanup()

    def cleanup(self):
        logger.info("Shutting down Gesture Control System...")
        self.camera.stop()
        self.detector.close()
        self.hardware.cleanup()
        if not self.headless:
            cv2.destroyAllWindows()
        logger.info("Shutdown completed successfully.")


def parse_args():
    parser = argparse.ArgumentParser(description="Raspberry Pi Hand Gesture Control")
    parser.add_argument("--headless", action="store_true", help="Run without graphical display window")
    parser.add_argument("--camera-type", default=config.CAMERA_TYPE, choices=["auto", "picamera2", "opencv"],
                        help="Camera backend to use (default: auto)")
    parser.add_argument("--camera-index", type=int, default=config.CAMERA_INDEX,
                        help="OpenCV camera device index (default: 0)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logger.info("==================================================")
    logger.info(" Raspberry Pi Hand Gesture Control System")
    logger.info(" Developed by AI & Robotics Team in DPSI")
    logger.info("==================================================")
    app = GestureApp(
        headless=args.headless,
        camera_type=args.camera_type,
        camera_index=args.camera_index
    )
    app.run()
