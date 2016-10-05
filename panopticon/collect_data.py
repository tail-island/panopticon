import RPi.GPIO as gpio
import itertools
import time


def setup_gpio_pins(trigs, echos):
    gpio.setup(trigs, gpio.OUT);
    gpio.output(trigs, gpio.HIGH)

    gpio.setup(echos, gpio.IN)


def read_distance(trig, echo):
    gpio.output(trig, True)
    time.sleep(0.000015)
    gpio.output(trig, False)

    while not gpio.input(echo):
        pass

    echo_starting_time = time.time()

    while gpio.input(echo):
        pass

    return min((time.time() - echo_starting_time) / 2 * 340, 4.5)


def collect_data():
    try:
        gpio.setmode(gpio.BCM)
        pin_pairs = ((17, 27), (22, 23))
        
        setup_gpio_pins(*zip(*pin_pairs))
        while True:
            yield itertools.starmap(read_distance, pin_pairs)
            time.sleep(0.5)

    finally:
        gpio.cleanup()


if __name__ == '__main__':
    for data in collect_data():
        for x in data:
            print(x, end='\t')
        print()
