import RPi.GPIO as gpio

from itertools import startmap
from time import time, sleep


gpio.setmode(gpio.BCM)

pin_pairs = (( 4, 14),
             (15, 17),
             (27, 22),
             (23, 24),
             (10,  9),
             (25, 11),
             ( 8,  7),
             ( 5,  6),
             (12, 13),
             (19, 16))


def setup_gpio_pins(trigs, echos):
    gpio.setup(trigs, gpio.OUT);
    gpio.output(trigs, gpio.HIGH)

    gpio.setup(echos, gpio.IN)


def read_distance(trig, echo):
    gpio.output(trig, True)
    sleep(0.000015)
    gpio.output(trig, False)

    while not gpio.input(echo):
        pass

    echo_starting_time = time()

    while gpio.input(echo):
        pass

    return min((time() - echo_starting_time) / 2 * 340, 4.5)


def read_poses():
    try:
        setup_gpio_pins(*zip(*pin_pairs))
        
        while True:
            yield tuple(starmap(read_distance, pin_pairs))
            sleep(0.2)

    finally:
        gpio.cleanup()


if __name__ == '__main__':
    for pose in read_poses():
        for distance in pose:
            print('%.2' % distance, end='\t')
        print()
