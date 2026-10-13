#!/bin/bash

cd "$(dirname "$0")"

if [ -z "$DISPLAY" ]; then
    export DISPLAY=:0
fi

if [ -z "$XDG_RUNTIME_DIR" ]; then
    export XDG_RUNTIME_DIR=/run/user/$(id -u)
fi

if [ -z "$WAYLAND_DISPLAY" ]; then
    export WAYLAND_DISPLAY=wayland-0
fi

if command -v xhost >/dev/null 2>&1; then
    xhost +local:ailab >/dev/null 2>&1 || true
fi

echo "=========================================================="
echo " Starting Hand Gesture Control System"
echo " Developed by AI & Robotics Team in DPSI"
echo " Display: $DISPLAY | Wayland: $WAYLAND_DISPLAY"
echo "=========================================================="

/usr/bin/python3 main.py "$@"
