import sys
import logging
import config

logger = logging.getLogger("HardwareController")
logging.basicConfig(level=logging.INFO)


class HardwareController:
    def __init__(self, fan_pin=config.FAN_PIN, light_pin=config.LIGHT_PIN,
                 gate_pin=config.GATE_PIN, gate_type=config.GATE_TYPE,
                 active_high=config.RELAY_ACTIVE_HIGH):
        self.fan_pin = fan_pin
        self.light_pin = light_pin
        self.gate_pin = gate_pin
        self.gate_type = gate_type.upper()
        self.active_high = active_high

        self.fan_state = False
        self.light_state = False
        self.gate_state = False

        self.is_rpi = False
        self.fan_device = None
        self.light_device = None
        self.gate_device = None

        self._setup_gpio()

    def _setup_gpio(self):
        try:
            import gpiozero
            from gpiozero import OutputDevice, AngularServo

            logger.info("Initializing Raspberry Pi GPIO (gpiozero)...")

            self.fan_device = OutputDevice(self.fan_pin, active_high=self.active_high, initial_value=False)
            self.light_device = OutputDevice(self.light_pin, active_high=self.active_high, initial_value=False)

            if self.gate_type == "SERVO":
                self.gate_device = AngularServo(
                    self.gate_pin,
                    min_angle=0,
                    max_angle=180,
                    min_pulse_width=0.0005,
                    max_pulse_width=0.0025
                )
                self.gate_device.angle = config.GATE_SERVO_CLOSE_ANGLE
            else:
                self.gate_device = OutputDevice(self.gate_pin, active_high=self.active_high, initial_value=False)

            self.is_rpi = True
            logger.info("Hardware GPIO initialized successfully.")
        except Exception as e:
            logger.warning(
                "GPIO initialization failed or running on non-RPi platform (%s). "
                "Hardware controller is running in SIMULATION/MOCK mode.", e
            )
            self.is_rpi = False

    def set_fan(self, state: bool) -> bool:
        if self.fan_state == state:
            return False

        self.fan_state = state
        if self.is_rpi and self.fan_device:
            if state:
                self.fan_device.on()
            else:
                self.fan_device.off()

        logger.info(">> [ACTUATOR] FAN turned %s", "ON" if state else "OFF")
        return True

    def set_light(self, state: bool) -> bool:
        if self.light_state == state:
            return False

        self.light_state = state
        if self.is_rpi and self.light_device:
            if state:
                self.light_device.on()
            else:
                self.light_device.off()

        logger.info(">> [ACTUATOR] LIGHT turned %s", "ON" if state else "OFF")
        return True

    def set_gate(self, open_state: bool) -> bool:
        if self.gate_state == open_state:
            return False

        self.gate_state = open_state
        if self.is_rpi and self.gate_device:
            if self.gate_type == "SERVO":
                target_angle = config.GATE_SERVO_OPEN_ANGLE if open_state else config.GATE_SERVO_CLOSE_ANGLE
                self.gate_device.angle = target_angle
            else:
                if open_state:
                    self.gate_device.on()
                else:
                    self.gate_device.off()

        logger.info(">> [ACTUATOR] GATE is now %s", "OPEN" if open_state else "CLOSED")
        return True

    def get_status(self):
        return {
            "fan": self.fan_state,
            "light": self.light_state,
            "gate": self.gate_state,
            "simulation_mode": not self.is_rpi
        }

    def cleanup(self):
        logger.info("Cleaning up GPIO devices...")
        if self.is_rpi:
            try:
                if self.fan_device:
                    self.fan_device.close()
                if self.light_device:
                    self.light_device.close()
                if self.gate_device:
                    self.gate_device.close()
            except Exception as e:
                logger.error("Error during GPIO cleanup: %s", e)
        logger.info("Hardware cleanup complete.")
