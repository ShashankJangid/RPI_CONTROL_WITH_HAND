# Raspberry Pi Hand Gesture Control System

> **Developed by AI & Robotics Team in DPSI**

A computer vision and IoT hardware control system built for Raspberry Pi that automates electrical appliances and physical access mechanisms (**Fan**, **Light**, and **Gate**) using real-time hand gestures captured via camera.

---

## Gesture & Action Mapping

| Hand Gesture | Target Appliance | Hardware Pin | Triggered Action |
| :--- | :--- | :--- | :--- |
| **1 Finger** (Index extended) | **Fan** | GPIO 17 (Pin 11) | **Turn ON** (Relay Closed) |
| **2 Fingers** (Index + Middle) | **Fan** | GPIO 17 (Pin 11) | **Turn OFF** (Relay Open) |
| **3 Fingers** (Index + Middle + Ring) | **Light** | GPIO 27 (Pin 13) | **Turn ON** (Relay Closed) |
| **4 Fingers** (4 fingers extended, thumb folded) | **Light** | GPIO 27 (Pin 13) | **Turn OFF** (Relay Open) |
| **5 Fingers** (Open Palm - all 5 extended) | **Gate** | GPIO 22 (Pin 15) | **OPEN** (Servo 90°) |
| **Fist Closed** (All fingers folded) | **Gate** | GPIO 22 (Pin 15) | **CLOSE** (Servo 0°) |

---

## Key Features

- **Multi-backend Hand Detection**: Supports both deep-learning neural landmark detectors (MediaPipe / LiteRT / TFLite) and an adaptive real-time OpenCV skin-segmentation and convexity-defect analyzer for maximum performance on constrained hardware.
- **Debounce & Latency Filter**: Eliminates false positives by requiring consistent gesture recognition across configurable consecutive frames (`GESTURE_CONFIRM_FRAMES = 12`) with a latch mechanism to prevent multi-firing.
- **Hardware Agnostic**: Includes a hardware simulation mode. The software can be executed and tested on standard workstations (Windows, macOS, Ubuntu) using a standard webcam without needing physical Raspberry Pi GPIO pins.
- **Headless & GUI Modes**: Supports headless background execution on Raspberry Pi servers, as well as a rich OpenCV graphical overlay display for debugging.
- **Camera Auto-Negotiation**: Automatically interfaces with native Raspberry Pi CSI cameras (`picamera2`) and USB webcams (`v4l2` / OpenCV).

---

## Hardware Architecture & Wiring

### Bill of Materials
1. **Raspberry Pi**: Raspberry Pi 4 Model B / Raspberry Pi 5 (Raspberry Pi 3B+ also supported).
2. **Camera**: Raspberry Pi Camera Module v2/v3 (CSI ribbon) or standard USB webcam.
3. **Relay Module**: 2-Channel 5V Relay board (controls Fan and Light).
4. **Servo Motor**: Micro Servo SG90 or MG90S (controls physical Gate barrier).
5. **Power Supply**: 5V 3A power adapter with common ground.

### GPIO Wiring Pinout (BCM Mode)

| Peripheral / Module | Module Pin | Raspberry Pi Pin (BCM) | Physical Header Pin |
| :--- | :--- | :--- | :--- |
| **Fan Relay** | IN1 (Signal) | **GPIO 17** | Pin 11 |
| | VCC | 5V Power | Pin 2 or Pin 4 |
| | GND | Ground | Pin 6 |
| **Light Relay** | IN2 (Signal) | **GPIO 27** | Pin 13 |
| | VCC | 5V Power | Pin 2 or Pin 4 |
| | GND | Ground | Pin 9 |
| **Gate Servo (SG90)** | PWM Signal (Yellow/Orange) | **GPIO 22** | Pin 15 |
| | VCC (Red) | 5V Power | Pin 2 |
| | GND (Brown/Black) | Ground | Pin 14 |

*(Pin configurations and pulse limits are configurable in `config.py`)*

---

## Software Installation on Raspberry Pi

### 1. System Package Requirements
Update apt repository and install runtime packages:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3-pip python3-venv python3-opencv libcamera-dev python3-picamera2
```

### 2. Camera Setup (CSI Ribbon)
If using the Raspberry Pi Camera module:
```bash
sudo raspi-config
# Interface Options -> Camera -> Enable -> Reboot
```
Verify the camera feed:
```bash
rpicam-hello -t 3000
```

### 3. Clone Repository & Setup Environment
```bash
cd ~
git clone https://github.com/ShashankJangid/RPI_CONTROL_WITH_HAND.git
cd RPI_CONTROL_WITH_HAND

python3 -m venv --system-site-packages venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## Usage

### Interactive GUI Mode
Displays live video feed, hand landmark tracking, active states, and confirmation progress:
```bash
python3 main.py
```
*(Press `q` or `Esc` to quit safely and reset actuators)*

### Headless Server Mode
Runs in the background without requiring a monitor or X11/Wayland display:
```bash
python3 main.py --headless
```

### Run using Startup Script
```bash
chmod +x start.sh
./start.sh
```

### Camera Selection Flags
- Force OpenCV USB webcam: `python3 main.py --camera-type opencv --camera-index 0`
- Force Pi Camera CSI stream: `python3 main.py --camera-type picamera2`

---

## Configuration

All operational parameters can be adjusted in `config.py`:

```python
CAMERA_TYPE = "auto"
CAMERA_INDEX = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30

DETECTOR_TYPE = "opencv"
ROI_X1, ROI_Y1 = 200, 80
ROI_X2, ROI_Y2 = 440, 360

GESTURE_CONFIRM_FRAMES = 12
ACTION_COOLDOWN_SEC = 1.0

FAN_PIN = 17
LIGHT_PIN = 27
GATE_PIN = 22
GATE_TYPE = "SERVO"
GATE_SERVO_OPEN_ANGLE = 90
GATE_SERVO_CLOSE_ANGLE = 0
RELAY_ACTIVE_HIGH = True
```

---

## Unit Testing

Run unit tests locally on any platform:
```bash
python -m unittest test_gesture_control.py
```

---

## Team & Credits

**Developed by AI & Robotics Team in DPSI**  
Author: [Shashank Jangid](https://github.com/ShashankJangid)
