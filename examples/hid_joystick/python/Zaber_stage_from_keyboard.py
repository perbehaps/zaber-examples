"""Simple Zaber stage control using a HID joystick under Windows.

Demonstrates using the Python input library to read a HID joystick, then
translate the stick deflections to stage velocities and button presses to
home or stop commands.

This example expects to find a two-axis device on the designated serial port.
Edit the constants below to change this.

Created 2023, Contributors: Soleil Lapierre
"""

import keyboard
import logging
import math
import time  # Add this import

from inputs import get_gamepad  # type: ignore

from zaber_motion import MotionLibException
from zaber_motion.ascii import Connection, Axis


log = logging.getLogger(__name__)

SERIAL_PORT = "COM8"

# Map joystick axes to Zaber (<device_address>, <axis_number>).
X_AXIS = (1, 1)
Y_AXIS = (1, 2)

# Constant for analog stick range from the inputs library.
MAX_DEFLECTION = 32768

# Define a dead zone that mapss to zero, with linear increase from the edge.
DEAD_ZONE = MAX_DEFLECTION / 5


def scale_deflection(deflection: float) -> float:
    """Map stick deflection to the range -1 to 1, with dead zone and curve."""
    defl_abs = math.fabs(deflection)
    if defl_abs < 1:
        return 0

    sign = deflection / defl_abs
    scaled = (max(DEAD_ZONE, defl_abs) - DEAD_ZONE) / (MAX_DEFLECTION - DEAD_ZONE)
    log.info(str(scaled))
    return sign * math.pow(scaled, 3)


def read_loop(x_axis: Axis, y_axis: Axis) -> None:
    """Read keyboard input and controls devices accordingly. Main loop of the program."""
    max_speed_x = x_axis.settings.get("maxspeed")
    max_speed_y = y_axis.settings.get("maxspeed")

    log.info("Use arrow keys to move the X and Y axes.")
    log.info("Press 'H' to home, 'S' to stop, or 'Esc' to exit.")

    while True:
        x_speed = 0
        y_speed = 0

        if keyboard.is_pressed("left"):
            x_speed = -max_speed_x
        elif keyboard.is_pressed("right"):
            x_speed = max_speed_x

        if keyboard.is_pressed("up"):
            y_speed = max_speed_y
        elif keyboard.is_pressed("down"):
            y_speed = -max_speed_y

        if keyboard.is_pressed("h"):
            log.info("Homing")
            x_axis.home(wait_until_idle=False)
            y_axis.home(wait_until_idle=False)
            x_axis.wait_until_idle()
            y_axis.wait_until_idle()
            log.info("Homing completed")
            keyboard.wait("h", suppress=True)  # Wait for key release

        elif keyboard.is_pressed("s"):
            log.info("Stopping")
            x_axis.stop(wait_until_idle=False)
            y_axis.stop(wait_until_idle=False)
            keyboard.wait("s", suppress=True)  # Wait for key release

        elif keyboard.is_pressed("esc"):
            log.info("Exiting")
            break

        # Only send move commands if speed changed
        if x_speed != 0 or y_speed != 0:
            log.info("Changing velocities to %s and %s.", x_speed, y_speed)
            x_axis.move_velocity(x_speed)
            y_axis.move_velocity(y_speed)
        else:
            # Stop axes if no key is pressed
            x_axis.stop(wait_until_idle=False)
            y_axis.stop(wait_until_idle=False)

        # Small delay to avoid flooding commands
        time.sleep(0.05)


def main() -> None:
    """Open Zaber device connections and initialize the program."""
    # Verify the expected devices exist.
    with Connection.open_serial_port(SERIAL_PORT) as connection:
        try:
            x_device = connection.get_device(X_AXIS[0])
            x_device.identify()
            x_axis = x_device.get_axis(X_AXIS[1])
        except MotionLibException:
            log.error("Failed to identify the X axis at address /%d %d", X_AXIS[0], X_AXIS[1])
            return

        try:
            if Y_AXIS[0] != X_AXIS[0]:
                y_device = connection.get_device(Y_AXIS[0])
                y_device.identify()
            else:
                y_device = x_device
            y_axis = y_device.get_axis(Y_AXIS[1])
        except MotionLibException:
            log.error("Failed to identify the Y axis at address /%d %d", Y_AXIS[0], Y_AXIS[1])
            return

        read_loop(x_axis, y_axis)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()