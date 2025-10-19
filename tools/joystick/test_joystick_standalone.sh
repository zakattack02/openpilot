#!/bin/bash
# Standalone joystick test without openpilot manager

echo "Stopping any running openpilot processes..."
pkill -f "system.manager.manager" 2>/dev/null
pkill -f "controlsd" 2>/dev/null
sleep 2

echo "Cleaning up IPC sockets..."
rm -f /dev/shm/controlsState* /dev/shm/testJoystick* 2>/dev/null

echo "Starting joystick control..."
cd "$(dirname "$0")/../.."
uv run python3 tools/joystick/joystick_control.py "$@"
