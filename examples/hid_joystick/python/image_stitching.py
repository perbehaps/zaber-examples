"""Simple raster-scan and goto utility for two Zaber stages.

- Uses device/address mapping for X and Y stages.
- Performs a serpentine raster scan by default.
- Provides goto_position(x, y) for direct moves.
- Stops cleanly on KeyboardInterrupt.
"""
import logging
import time
from typing import Tuple

from zaber_motion import Units, MotionLibException
from zaber_motion.ascii import Connection, Axis

log = logging.getLogger(__name__)

# Connection / axis mapping (adjust to your hardware)
SERIAL_PORT = "COM8"
X_DEVICE = 1
X_AXIS_NUM = 1
Y_DEVICE = 2
Y_AXIS_NUM = 1

# Units used for absolute moves (assumes linear stages in millimetres)
STAGE_UNITS = Units.LENGTH_MILLIMETRES

# Safety / timing
INTER_MOVE_DELAY = 0.5  # seconds between issuing commands
POLL_INTERVAL = 0.05


def connect_axes(serial_port: str) -> Tuple[Axis, Axis]:
    """Open connection and return (x_axis, y_axis). Caller should ensure context."""
    conn = Connection.open_serial_port(serial_port)
    # return the connection object too by using context manager in caller; here we return axes and connection object
    connection = conn  # alias
    try:
        x_device = connection.get_device(X_DEVICE)
        x_device.identify()
        x_axis = x_device.get_axis(X_AXIS_NUM)
    except MotionLibException:
        connection.close()
        raise

    try:
        if Y_DEVICE != X_DEVICE:
            y_device = connection.get_device(Y_DEVICE)
            y_device.identify()
        else:
            y_device = x_device
        y_axis = y_device.get_axis(Y_AXIS_NUM)
    except MotionLibException:
        connection.close()
        raise

    return connection, x_axis, y_axis


def wait_for_axes_idle(x_axis: Axis, y_axis: Axis, poll: float = POLL_INTERVAL) -> None:
    """Block until both axes report idle."""
    while True:
        try:
            # print(x_axis.wait_until_idle(), y_axis.wait_until_idle())
            # if x_axis.wait_until_idle() and y_axis.wait_until_idle():
                # print(x_axis.wait_until_idle(), y_axis.wait_until_idle())
            return
        except MotionLibException:
            # If communication hiccup, sleep and retry
            log.debug("Error querying idle state; retrying.", exc_info=True)
        time.sleep(poll)


def goto_position(x: float, y: float, x_axis: Axis, y_axis: Axis, units: Units = STAGE_UNITS) -> None:
    """Move both axes to an absolute (x, y) position and wait until move completes.

    Moves are issued non-blocking for both axes, then the function waits until both axes are idle.
    """
    log.info("Goto absolute position X=%.6g, Y=%.6g %s", x, y, units.name)
    try:
        x_axis.move_absolute(x, units, wait_until_idle=False)
        # small delay between commands to avoid command overlap on bus
        time.sleep(INTER_MOVE_DELAY)
        y_axis.move_absolute(y, units, wait_until_idle=False)
        wait_for_axes_idle(x_axis, y_axis)
    except MotionLibException:
        log.error("Error during goto_position", exc_info=True)
        raise


def raster_scan(
    x_start: float,
    x_steps: int,
    x_step_size: float,
    y_start: float,
    y_steps: int,
    y_step_size: float,
    x_axis: Axis,
    y_axis: Axis,
    units: Units = STAGE_UNITS,
    serpentine: bool = True,
) -> None:
    """Perform a raster scan.

    Parameters:
    - x_start: X coordinate of the first column.
    - x_steps: number of columns (integer).
    - x_step_size: distance between columns.
    - y_start: Y coordinate of the first row.
    - y_steps: number of rows (integer).
    - y_step_size: distance between rows.
    - serpentine: if True, alternate X direction on each row to avoid long returns.
    """
    log.info("Starting raster scan: x_steps=%d, y_steps=%d, x_step=%.6g, y_step=%.6g %s",
             x_steps, y_steps, x_step_size, y_step_size, units.name)

    # Precompute X positions for a single row (left-to-right)
    x_positions = [x_start + i * x_step_size for i in range(x_steps)]

    try:
        for row in range(y_steps):
            y_pos = y_start + row * y_step_size
            # choose X order for this row
            if serpentine and (row % 2 == 1):
                iter_x = reversed(x_positions)
            else:
                iter_x = x_positions

            for x_pos in iter_x:
                log.info("Row %d / %d -> moving to (%.6g, %.6g)", row + 1, y_steps, x_pos, y_pos)
                goto_position(x_pos, y_pos, x_axis, y_axis, units)
                # place for capture / measurement call
                time.sleep(INTER_MOVE_DELAY)

    except KeyboardInterrupt:
        log.warning("Scan interrupted by user; stopping axes.")
        try:
            x_axis.stop(wait_until_idle=False)
            y_axis.stop(wait_until_idle=False)
        except Exception:
            log.debug("Error sending stop to axes", exc_info=True)
        raise


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    # Example scan parameters (adjust to your experiment)
    # x_start = x_axis.get_position(STAGE_UNITS)
    x_steps = 10
    x_step_size = 0.1  # mm
    # y_start = y_axis.get_position(STAGE_UNITS)
    y_steps = 10
    y_step_size = 0.1  # mm

    # Connect and run scan
    # Use context manager so connection closes cleanly
    with Connection.open_serial_port(SERIAL_PORT) as conn:
        try:
            # get axes
            x_device = conn.get_device(X_DEVICE)
            x_device.identify()
            x_axis = x_device.get_axis(X_AXIS_NUM)
            x_start = x_axis.get_position(STAGE_UNITS)

            if Y_DEVICE != X_DEVICE:
                y_device = conn.get_device(Y_DEVICE)
                y_device.identify()
            else:
                y_device = x_device
            y_axis = y_device.get_axis(Y_AXIS_NUM)
            y_start = y_axis.get_position(STAGE_UNITS)

            # Optional: print current positions
            try:
                current_x = x_axis.get_position(STAGE_UNITS)
                current_y = y_axis.get_position(STAGE_UNITS)
                log.info("Current position X=%.6g, Y=%.6g %s", current_x, current_y, STAGE_UNITS.name)
            except Exception:
                log.debug("Could not read current positions", exc_info=True)

            # Run raster scan
            raster_scan(
                x_start=x_start,
                x_steps=x_steps,
                x_step_size=x_step_size,
                y_start=y_start,
                y_steps=y_steps,
                y_step_size=y_step_size,
                x_axis=x_axis,
                y_axis=y_axis,
                units=STAGE_UNITS,
                serpentine=True,
            )

            log.info("Raster scan complete.")

            try:
                x_axis.stop(wait_until_idle=False)
                y_axis.stop(wait_until_idle=False)
            except Exception:
                log.debug("Error sending stop to axes", exc_info=True)
                
        except MotionLibException:
            log.error("Motion library error during setup or scan", exc_info=True)


if __name__ == "__main__":
    main()