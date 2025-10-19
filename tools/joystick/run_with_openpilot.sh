#!/bin/bash
# Run joystick control alongside openpilot for car control

echo "======================================================"
echo "   JOYSTICK CONTROL FOR OPENPILOT"
echo "======================================================"
echo ""
cd "$(dirname "$0")/../.."

# Enable joystick debug mode
echo "[1/3] Enabling JoystickDebugMode..."
uv run python3 -c "from openpilot.common.params import Params; p = Params(); p.put_bool('JoystickDebugMode', True); print('✓ JoystickDebugMode enabled')"

echo ""
echo "[2/3] Checking if openpilot manager is running..."
if pgrep -f "system.manager.manager" > /dev/null; then
    echo "✓ Manager is running"
    echo ""
    echo "   NOTE: Manager will automatically start 'joystickd' process"
    echo "   which converts joystick messages to car control commands."
else
    echo "✗ Manager is NOT running!"
    echo ""
    echo "   Please start openpilot first:"
    echo "   ./launch_openpilot.sh"
    echo ""
    exit 1
fi

echo ""
echo "[3/3] Starting joystick control..."
echo "======================================================"
echo ""
echo "CONTROLS:"
echo "  Left Stick (horizontal):  Steering"
echo "  Right Stick:              Cruise speed adjustment"
echo "  X/Triangle Button:        Disengage/Cancel"
echo ""
echo "IMPORTANT:"
echo "  1. Engage cruise control on your car first"
echo "  2. openpilot will show 'Joystick Mode' alert"
echo "  3. Keep this script running!"
echo ""
echo "======================================================"
echo ""

# Run the joystick control - THIS MUST STAY RUNNING
uv run python3 tools/joystick/joystick_control.py "$@"

