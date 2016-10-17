import RPi.GPIO as gpio
import time


if __name__ == '__main__':
    try:
        gpio.setmode(gpio.BCM)

        gpio.setup(18, gpio.OUT)
        gpio.output(18, gpio.HIGH)

        pwm = gpio.PWM(18, 100)

        pwm.start(0)
        for dc in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            pwm.ChangeDutyCycle(dc)
            time.sleep(1)
        pwm.stop()

    finally:
        gpio.cleanup()
