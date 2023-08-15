"""
This file contains the Forest Fire class that is used to simulate the evolving (spreading/reducing) of a forest fire.
"""

# Importing relevant modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from scipy.signal import convolve2d

# Creating the Forest Fire class to simulate the evolving of a forest fire
class ForestFire(object):
    """
    Class for simulating a forest fire. It contains the following key functionality:
    change_wind:
      Allocating a random new wind strength and wind direction.
    start:
      Start the forest fire with a single spark in the center of the forest.
    spread:
      Spread the forest fire according to a kernel reflecting wind direction and strength.
    reduce:
      Reduce the forest fire.
    """
    def __init__(self, area, wind_strength='random', wind_direction='random'):
        """
        Initialize the ForestFire class with its attributes.
        """
        self.size = np.round(np.sqrt(area))
        self.flames = np.zeros((int(self.size), int(self.size)))  # generate a matrix representing the forest fire
        # Wind strength determination[weak, medium, strong]
        if wind_strength == 'random':
          self.wind_strength = np.random.choice(['weak', 'medium', 'strong'])
        else:
          self.wind_strength = wind_strength
        # Wind direction determination [N, NE, E, SE, S, SW, W, NW]
        if wind_direction == 'random':
          self.wind_direction = np.random.choice(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'])
        else:
          self.wind_direction = wind_direction
        # Wind kernel dictionary
        self.kernel_wind_mapping = {
        # Light winds (3 x 3 kernels)
        'weakN': np.array([[1, 1, 1],
                          [0, 1, 0],
                          [0, 0, 0]]),
        'weakNE': np.array([[0, 0, 1],
                            [0, 1, 0],
                            [0, 0, 0]]),
        'weakE': np.array([[0, 0, 1],
                          [0, 1, 1],
                          [0, 0, 1]]),
        'weakSE': np.array([[0, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1]]),
        'weakS': np.array([[0, 0, 0],
                          [0, 1, 0],
                          [1, 1, 1]]),
        'weakSW': np.array([[0, 0, 0],
                            [0, 1, 0],
                            [1, 0, 0]]),
        'weakW': np.array([[1, 0, 0],
                          [1, 1, 0],
                          [1, 0, 0]]),
        'weakNW': np.array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 0]]),

        # Medium winds (5 x 5 kernels)
        'mediumN': np.array([[1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 0],
                            [0, 0, 1, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0]]),
        'mediumNE': np.array([[0, 0, 0, 1, 1],
                              [0, 0, 0, 1, 1],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]]),
        'mediumE': np.array([[0, 0, 0, 0, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 1, 1, 1],
                            [0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1]]),
        'mediumSE': np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 1, 1],
                              [0, 0, 0, 1, 1]]),
        'mediumS': np.array([[0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0],
                            [0, 0, 1, 0, 0],
                            [0, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1]]),
        'mediumSW': np.array([[0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0]]),
        'mediumW': np.array([[1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0],
                            [1, 1, 0, 0, 0],
                            [1, 0, 0, 0, 0]]),
        'mediumNW': np.array([[1, 1, 0, 0, 0],
                              [1, 1, 0, 0, 0],
                              [0, 0, 1, 0, 0],
                              [0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0]]),

        # Strong winds (7 x 7 kernels)
        'strongN': np.array([[1, 1, 1, 1, 1, 1, 1],
                            [0, 1, 1, 1, 1, 1, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0]]),
        'strongNE': np.array([[0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]]),
        'strongE': np.array([[0, 0, 0, 0, 0, 0, 1],
                            [0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 1, 1, 1, 1],
                            [0, 0, 0, 0, 1, 1, 1],
                            [0, 0, 0, 0, 0, 1, 1],
                            [0, 0, 0, 0, 0, 0, 1]]),
        'strongSE': np.array([[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 1, 1, 1],
                              [0, 0, 0, 0, 1, 1, 1]]),
        'strongS': np.array([[0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 0, 0, 0, 0],
                            [0, 0, 0, 1, 0, 0, 0],
                            [0, 0, 1, 1, 1, 0, 0],
                            [0, 1, 1, 1, 1, 1, 0],
                            [1, 1, 1, 1, 1, 1, 1]]),
        'strongSW': np.array([[0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0]]),
        'strongW': np.array([[1, 0, 0, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 1, 1, 0, 0, 0],
                            [1, 1, 1, 0, 0, 0, 0],
                            [1, 1, 0, 0, 0, 0, 0],
                            [1, 0, 0, 0, 0, 0, 0]]),
        'strongNW': np.array([[1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0],
                              [1, 1, 1, 0, 0, 0, 0],
                              [0, 0, 0, 1, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0],
                              [0, 0, 0, 0, 0, 0, 0]]),
        }
        # Create wind strength & direction key
        self.kernel_key = self.wind_strength + self.wind_direction
        self.gen_kernel(self.kernel_key)  # Generate the kernel used for simulating the spread of the forest fire

    def gen_kernel(self, kernel_key):
        """
        Select spreading kernel based on wind strength and wind directon from a dictionary.
        This kernel is used for the flames matrix convolution (spreading).
        """
        # Using mappings to get kernel definition
        self.kernel = self.kernel_wind_mapping[kernel_key]

    def change_wind(self):
        """
        Randomly change the wind strength and direction.
        """
        self.wind_strength = np.random.choice(np.delete(np.array(['weak', 'medium', 'strong']), np.where(np.array(['weak', 'medium', 'strong']) == self.wind_strength)))
        self.wind_direction = np.random.choice(np.delete(np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']), np.where(np.array(['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW']) == self.wind_strength)))
        self.kernel_key = self.wind_strength + self.wind_direction
        self.gen_kernel(self.kernel_key)  # Generate the new kernel used for simulating the changed spreading of the forest fire

    def start(self):
        """
        Simulate the start of a forest fire at the center of the forest area.
        """
        center = round(self.size/2)  # locate center of forest area
        self.flames[center][center] = 1  # initialize fire spark at the forest center

    def spread(self):
        """
        Simulate the spreading of the firest fire (discrete step).
        """
        adjacent_zeros = convolve2d(self.flames, self.kernel, mode='same', boundary='symm')  # use the kernel to simulate fire spreading
        self.flames[(self.flames == 0) & (adjacent_zeros > 0)] = 1  # update flames matrix in-place

    def reduce(self):
        """
        Simulate the reduction of the flames.
        """
        reduction_factor = 0.2  # reduction factor
        mask = np.random.rand(*self.flames.shape) < reduction_factor  # create a mask matrix
        self.flames[mask & (self.flames == 1)] = 0  # adjust the flame matrix in-place

    def show(self):
        """
        Visualize the development of the flames (red = flames & green & no flames)
        """
        colormap = colors.ListedColormap(["green","red"])  # assign the colormap
        plt.imshow(self.flames, cmap=colormap)  # create the plot
        plt.show()

### End of File ###