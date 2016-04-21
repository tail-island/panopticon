import RPi.GPIO as GPIO
import itertools
import time


pin_pairs = ((17, 27), (22, 23))


def setup_gpio_pins(trigs, echos):
    GPIO.setup(trigs, GPIO.OUT);
    GPIO.output(trigs, GPIO.LOW)
    
    GPIO.setup(echos, GPIO.IN)


def read_distance(trig, echo):
    GPIO.output(trig, True)
    time.sleep(0.00001)
    GPIO.output(trig, False)

    while not GPIO.input(echo):
        pass

    echo_starting_time = time.time()

    while GPIO.input(echo):
        pass

    return min((time.time() - echo_starting_time) / 2 * 340, 4.5)


def collect_data():
    try:
        GPIO.setmode(GPIO.BCM)

        setup_gpio_pins(*zip(*pin_pairs))

        while True:
            loop_starting_time = time.time()
            
            yield itertools.starmap(read_distance, pin_pairs)
            
            time.sleep(max(0.2 - (time.time() - loop_starting_time), 0))

    finally:
        GPIO.cleanup()


if __name__ == '__main__':
    for data in collect_data():
        for x in data:
            print(x, end='\t')
        print()
