import numpy as np


class Road:

    def __init__(self, length: float = 100.0):

        self.length = length

    def generate_straight_road(self):

        x = np.linspace(0, self.length, 500)

        y = np.zeros_like(x)

        return x, y