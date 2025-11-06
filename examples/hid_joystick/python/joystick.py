"""Simple Zaber stage control using a HID joystick under Windows.

Demonstrates using the Python input library to read a HID joystick, then
translate the stick deflections to stage velocities and button presses to
home or stop commands.

This example expects to find a two-axis device on the designated serial port.
Edit the constants below to change this.

Created 2023, Contributors: Soleil Lapierre
"""

import logging
import math
import time
import threading

from inputs import get_gamepad  # type: ignore

from zaber_motion import Units, MotionLibException
from zaber_motion.ascii import Connection, Axis

import matplotlib.pyplot as plt


log = logging.getLogger(__name__)

SERIAL_PORT = "COM8"

# Map joystick axes to Zaber (<device_addresse>, <axis_number>).
X_AXIS = (1, 1)
Y_AXIS = (2, 1)

# Constant for analog stick range from the inputs library.
MAX_DEFLECTION = 32768

# Tweakable parameters to increase responsiveness/speed:
SPEED_GAIN = 1          # multiply computed velocity (set >1 to increase speed)
RESPONSE_EXPONENT = 3       # 1 = linear, 2 = quadratic, 3 = cubic (current)
DEAD_ZONE = MAX_DEFLECTION / 5  # smaller value = more sensitive around center


def scale_deflection(deflection: float) -> float:
    """Map stick deflection to the range -1 to 1, with dead zone and configurable curve."""
    defl_abs = math.fabs(deflection)
    if defl_abs < 1:
        return 0

    sign = deflection / defl_abs
    scaled = (max(DEAD_ZONE, defl_abs) - DEAD_ZONE) / (MAX_DEFLECTION - DEAD_ZONE)
    log.info(str(scaled))
    return sign * math.pow(scaled, RESPONSE_EXPONENT)


def joystick_worker(x_axis: Axis, y_axis: Axis, pos: dict, lock: threading.Lock, stop_event: threading.Event) -> None:
    """Background thread: read joystick and control stages. Update pos dict for plotting."""
    # Pre-populate joystick state
    input_states = {
        "BTN_SELECT": 0,
        "ABS_X": 0,
        "ABS_Y": 0,
        "BTN_EAST": 0,
        "BTN_WEST": 0,
    }

    max_speed_x = x_axis.settings.get("maxspeed")
    # print(max_speed_x)
    x_axis.settings.set("maxspeed", 32768000000.0)
    print(max_speed_x)
    max_speed_y = y_axis.settings.get("maxspeed")
    # print(max_speed_y)
    y_axis.settings.set("maxspeed", 32768000000.0)
    print(max_speed_y)

    log.info("Joystick thread started. Use left stick to move, X to home, B to stop, Start to exit.")
    while not stop_event.is_set():
        try:
            events = get_gamepad()  # blocking, runs in worker thread so UI stays responsive
        except Exception:
            log.debug("get_gamepad error, retrying", exc_info=True)
            time.sleep(0.05)
            continue

        for event in events:
            if event.ev_type in ("Absolute", "Key"):
                input_states[event.code] = event.state
            else:
                continue

            # Exit request from joystick
            if input_states["BTN_SELECT"] == 1:
                log.info("Start button pressed: requesting stop.")
                stop_event.set()
                break

            try:
                # Homing
                if input_states["BTN_WEST"] == 1:
                    log.info("Homing (blocking start).")
                    x_axis.home(wait_until_idle=True)
                    y_axis.home(wait_until_idle=True)
                    # do not block waiting here so other inputs can interrupt homing
                    log.info("Device homed.")
                # Stop
                elif input_states["BTN_EAST"] == 1:
                    log.info("Stopping")
                    x_axis.stop(wait_until_idle=False)
                    y_axis.stop(wait_until_idle=False)
                else:
                    # compute scaled deflection in [-1,1]
                    x_raw = scale_deflection(input_states["ABS_X"])
                    y_raw = scale_deflection(input_states["ABS_Y"])

                    # apply gain and convert to device velocity, then clamp to device max
                    x_speed = x_raw * max_speed_x * SPEED_GAIN
                    y_speed = y_raw * max_speed_y * SPEED_GAIN

                    def clamp(v, limit):
                        return max(min(v, limit), -limit)

                    x_speed = clamp(x_speed, max_speed_x)
                    y_speed = clamp(y_speed, max_speed_y)

                    x_axis.move_velocity(x_speed)
                    y_axis.move_velocity(y_speed)

            except MotionLibException:
                log.error("Error sending move command", exc_info=True)

        # Update shared position for plotting (best-effort; ignore read errors)
        try:
            cur_x = x_axis.get_position(Units.LENGTH_MILLIMETRES)
            cur_y = y_axis.get_position(Units.LENGTH_MILLIMETRES)
            with lock:
                pos["x"] = cur_x
                pos["y"] = cur_y
        except Exception:
            log.debug("Could not read position for plotting", exc_info=True)

    log.info("Joystick thread exiting.")


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

        # Setup live matplotlib plot
        plt.ion()
        fig, ax = plt.subplots()

        try:
            init_x = x_axis.get_position(Units.LENGTH_MILLIMETRES)
            init_y = y_axis.get_position(Units.LENGTH_MILLIMETRES)
        except Exception:
            init_x, init_y = 0.0, 0.0

        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
        span = 10.0
        ax.set_xlim(init_x - span / 2, init_x + span / 2)
        ax.set_ylim(init_y - span / 2, init_y + span / 2)
        ax.set_aspect("equal", adjustable="box")
        point, = ax.plot([init_x], [init_y], "ro", label="Stage position")
        ax.legend()

        # Shared state between threads
        pos = {"x": init_x, "y": init_y}
        lock = threading.Lock()
        stop_event = threading.Event()

        # Start joystick worker thread
        worker = threading.Thread(target=joystick_worker, args=(x_axis, y_axis, pos, lock, stop_event), daemon=True)
        worker.start()

        try:
            # Main GUI loop: update plot and process GUI events
            while not stop_event.is_set():
                with lock:
                    cur_x = pos["x"]
                    cur_y = pos["y"]

                # set_data expects sequences; pass single-item lists for scalars
                point.set_data([cur_x], [cur_y])

                # adjust view if needed
                xlim = ax.get_xlim()
                ylim = ax.get_ylim()
                pad_x = (xlim[1] - xlim[0]) * 0.1
                pad_y = (ylim[1] - ylim[0]) * 0.1
                if not (xlim[0] + 1e-12 <= cur_x <= xlim[1] - 1e-12):
                    ax.set_xlim(min(cur_x - pad_x, xlim[0]), max(cur_x + pad_x, xlim[1]))
                if not (ylim[0] + 1e-12 <= cur_y <= ylim[1] - 1e-12):
                    ax.set_ylim(min(cur_y - pad_y, ylim[0]), max(cur_y + pad_y, ylim[1]))

                fig.canvas.draw_idle()
                # Process GUI events and yield control
                plt.pause(0.05)

        except KeyboardInterrupt:
            log.info("User requested exit (KeyboardInterrupt).")
            stop_event.set()
        finally:
            # Ensure worker stops (it may be blocked in get_gamepad until an event occurs)
            worker.join(timeout=0.5)
            try:
                plt.ioff()
                plt.close(fig)
            except Exception:
                pass


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
