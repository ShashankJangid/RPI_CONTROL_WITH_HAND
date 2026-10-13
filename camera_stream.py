import sys
import time
import logging
from threading import Thread, Lock
try:
    import cv2
except ImportError:
    cv2 = None
import config

logger = logging.getLogger("CameraStream")
logging.basicConfig(level=logging.INFO)


class CameraStream:
    def __init__(self, camera_type=config.CAMERA_TYPE, camera_index=config.CAMERA_INDEX,
                 width=config.FRAME_WIDTH, height=config.FRAME_HEIGHT, fps=config.FPS):
        self.camera_type = camera_type
        self.camera_index = camera_index
        self.width = width
        self.height = height
        self.fps = fps

        self.picam2 = None
        self.cap = None
        self.frame = None
        self.stopped = False
        self.lock = Lock()
        self.backend = None
        self.thread = None

        self._initialize_camera()

        if self.cap is not None and self.cap.isOpened():
            self.thread = Thread(target=self._update_opencv, args=(), daemon=True)
            self.thread.start()

    def _initialize_camera(self):
        if self.camera_type in ("picamera2", "auto"):
            try:
                from picamera2 import Picamera2
                logger.info("Initializing Picamera2 (Raspberry Pi Camera)...")
                self.picam2 = Picamera2()
                video_config = self.picam2.create_video_configuration(
                    main={"size": (self.width, self.height), "format": "BGR888"}
                )
                self.picam2.configure(video_config)
                self.picam2.start()
                time.sleep(0.8)
                test_frame = self.picam2.capture_array()
                if test_frame is not None and test_frame.size > 0:
                    self.backend = "Picamera2 (ov5647 / CSI)"
                    logger.info("Picamera2 initialized successfully (%dx%d BGR888).", self.width, self.height)
                    return
            except Exception as e:
                logger.warning("Picamera2 initialization failed (%s). Falling back to OpenCV.", e)
                if self.picam2 is not None:
                    try:
                        self.picam2.stop()
                        self.picam2.close()
                    except Exception:
                        pass
                    self.picam2 = None

        if cv2 is not None:
            logger.info("Initializing OpenCV VideoCapture(index=%s)...", self.camera_index)
            if sys.platform.startswith("linux"):
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            else:
                self.cap = cv2.VideoCapture(self.camera_index)

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            self.cap.set(cv2.CAP_PROP_FPS, self.fps)

            if self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret and frame is not None:
                    self.frame = frame
                    self.backend = f"OpenCV VideoCapture ({self.camera_index})"
                    logger.info("OpenCV VideoCapture started successfully (%dx%d).", self.width, self.height)
                    return

        logger.error("Could not open any camera backend.")

    def _update_opencv(self):
        while not self.stopped:
            try:
                if self.cap is not None and self.cap.isOpened():
                    ret, frame = self.cap.read()
                    if ret and frame is not None:
                        with self.lock:
                            self.frame = frame
                    else:
                        time.sleep(0.01)
                else:
                    time.sleep(0.02)
            except Exception as e:
                logger.error("Error capturing OpenCV frame: %s", e)
                time.sleep(0.05)

    def read(self):
        if self.stopped:
            return None

        if self.picam2 is not None:
            try:
                frame = self.picam2.capture_array()
                if frame is not None and frame.size > 0:
                    return frame
            except Exception as e:
                logger.warning("Picamera2 frame capture exception: %s", e)
                time.sleep(0.02)
                return None

        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def is_opened(self):
        if self.picam2 is not None:
            return True
        if self.cap is not None and self.cap.isOpened():
            return True
        return False

    def stop(self):
        self.stopped = True
        time.sleep(0.1)
        if self.picam2 is not None:
            try:
                self.picam2.stop()
                self.picam2.close()
            except Exception:
                pass
            self.picam2 = None

        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        logger.info("Camera stopped.")
