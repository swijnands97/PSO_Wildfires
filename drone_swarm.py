# Importing all relevant modules
import numpy as np
import pandas as pd
import random
import math
import matplotlib.pyplot as plt
from matplotlib import colors 
from core_parameters import *
import pickle

# Define function to load fire scenarious from CSV files
def load(scenario_name):
    with open(f'{scenario_name}.pkl', 'rb') as file:
        fire_scenario = pickle.load(file)
        return fire_scenario
      
# Load fire scenarious from CSV files
weak = load('weak')
medium = load('medium')
strong = load('strong')
wind_no_change = load('wind_no_change')
wind_change = load('wind_change')

# Creating the Forest Fire class to simulate the swarming of the drones
class DroneSwarm(object):
    """
    Class for simulating the swarm of drones. It contains a subclass 'Particle'. It has the following functionality:
    particle_swarm_optimization:
      Implements the Particle Swarm Optimization (PSO) algorithm. 
    run_optimization:
      Runs the particle swarm optimization and returns the best drone configuration's position and fitness.
    """
    def __init__(self, forest_size, previous_best_positions=None):
        self.matrix_size = forest_size
        self.particles = [self.Particle(self.matrix_size, previous_best_positions) for _ in range(num_particles)]
        self.global_best_position = self.particles[0].best_positions.copy()
        self.global_best_fitness = self.particles[0].best_fitness

    class Particle(object):
        """
        Class for simulating a single drone configuration (Particle). It contains the following functionality:
        move_within_bounds:
          Ensures that the drones' positions stay within the boundaries, implements collision avoidance, and
            limits the maximum movement of drones.
        evaluate_fitness:
          Calculates the fitness of a particle's position (percentage coverage of ones in binary fire matrix).
        """
        def __init__(self, forest_size, previous_best_positions=None):
            self.matrix_size = forest_size
            if previous_best_positions is None:
                # Initialize drones' positions near the center of the matrix
                center_x = self.matrix_size // 2
                center_y = self.matrix_size // 2
                x_vals = np.random.randint(center_x - self.matrix_size // 9, center_x + self.matrix_size // 9, num_drones)
                y_vals = np.random.randint(center_y - self.matrix_size // 9, center_y + self.matrix_size // 9, num_drones)
            else:
                # Use the previous best positions as initial positions for drones
                x_vals = previous_best_positions[::2]
                y_vals = previous_best_positions[1::2]
            self.positions = np.array([x for xy in zip(x_vals, y_vals) for x in xy])
            # Initialize the velocities within the allowed range
            max_velocity = max_drone_movement / math.sqrt(2)  # Limit initial velocity magnitude for diagonal movement
            self.velocities = np.array([random.uniform(-max_velocity, max_velocity) for _ in range(num_drones * 2)])
            self.best_positions = self.positions.copy()
            self.best_fitness = self.evaluate_fitness()

        def move_within_bounds(self):
            """
            Ensures that the drones' positions stay within the boundaries, implements collision avoidance, and
            limits the maximum movement of drones.
            """
            for i in range(0, num_drones * 2, 2):
                x, y = self.positions[i], self.positions[i + 1]
                initial_x, initial_y = self.best_positions[i], self.best_positions[i + 1]
                # Calculate the Euclidean distance between the current position and the initial position
                distance = math.sqrt((x - initial_x) ** 2 + (y - initial_y) ** 2)
                # If the distance exceeds the maximum allowed movement, adjust the position
                if distance > max_drone_movement:
                    dx = x - initial_x
                    dy = y - initial_y
                    scale_factor = max_drone_movement / distance
                    x = initial_x + dx * scale_factor
                    y = initial_y + dy * scale_factor
                # Ensure the new position stays within the matrix boundaries
                x_new = min(max(x, 0), self.matrix_size - 1)
                y_new = min(max(y, 0), self.matrix_size - 1)
                # Implement collision avoidance by checking if the new position is already occupied
                while (x_new, y_new) in zip(self.positions[::2], self.positions[1::2]):
                    # If there's a collision, randomly choose a new direction
                    dx = random.uniform(-1, 1)
                    dy = random.uniform(-1, 1)
                    x_new = x + dx
                    y_new = y + dy
                    # Ensure the new position stays within the matrix boundaries
                    x_new = min(max(x_new, 0), self.matrix_size - 1)
                    y_new = min(max(y_new, 0), self.matrix_size - 1)
                self.positions[i], self.positions[i + 1] = x_new, y_new

        def evaluate_fitness(self):
            """
            Calculates the fitness of a particle's position (percentage coverage of ones in binary fire matrix).
            """
            total_ones = np.sum(fire_matrix == 1)
            if total_ones > 0:
                covered_fire_locations = set()
                for i in range(0, num_drones * 2, 2):
                    x, y = int(self.positions[i]), int(self.positions[i + 1])
                    for dx in range(-coverage_range, coverage_range + 1):
                        for dy in range(-coverage_range, coverage_range + 1):
                            if dx * dx + dy * dy <= coverage_range * coverage_range:
                                new_x, new_y = x + dx, y + dy
                                if 0 <= new_x < self.matrix_size and 0 <= new_y < self.matrix_size:
                                    if fire_matrix[new_x, new_y] == 1:
                                        covered_fire_locations.add((new_x, new_y))
                return len(covered_fire_locations) / total_ones * 100  # Calculate the fitness as a percentage
            else:
                return 100

    def particle_swarm_optimization(self):
        """
        Implements the Particle Swarm Optimization (PSO) algorithm. 
        It iteratively updates the positions and velocities of particles, finding the best configuration of drones.
        """
        for _ in range(max_iterations):
            for particle in self.particles:
                # Update particle velocity and position using PSO equations
                particle.velocities = (
                    inertia_weight * particle.velocities +
                    c1 * random.random() * (particle.best_positions - particle.positions) +
                    c2 * random.random() * (self.global_best_position - particle.positions)
                )
                # Clip the velocities to the range [-max_drone_movement, max_drone_movement]
                particle.velocities = np.clip(particle.velocities, -max_drone_movement, max_drone_movement)
                # Update particle positions
                particle.positions = np.clip(particle.positions + particle.velocities, 0, self.matrix_size - 1)
                # Update particle best position and fitness if necessary
                current_fitness = particle.evaluate_fitness()
                if current_fitness > particle.best_fitness:
                    particle.best_fitness = current_fitness
                    particle.best_positions = particle.positions.copy()
                # Update global best position and fitness if necessary
                if current_fitness > self.global_best_fitness:
                    self.global_best_fitness = current_fitness
                    self.global_best_position = particle.positions.copy()
        # Limit drone movement within max_drone_movement from initial positions
        for particle in self.particles:
            particle.move_within_bounds()
        return self.global_best_position, self.global_best_fitness

    def run_optimization(self):
        """
        Runs the particle swarm optimization and returns the best drone configuration's position and fitness.
        """
        best_position, best_fitness = self.particle_swarm_optimization()
        return best_position, best_fitness

def plot_forest_fire_and_drones(fire_matrix, best_position):
    """
    Visualizes the forest fire and drone positions:(red = flames & green & no flames, black = drone) 
    """
    # Create a custom color map
    cmap = colors.ListedColormap(['green', 'red', 'black'])
    bounds = [0, 0.5, 1.5, 2.5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    plt.imshow(fire_matrix, cmap=cmap, norm=norm, interpolation='nearest')
    # Extract drone x and y positions
    drone_x = best_position[::2]  # Slice every other element starting from index 0 (x-coordinates)
    drone_y = best_position[1::2]  # Slice every other element starting from index 1 (y-coordinates)
    # Plot drones with black dots
    plt.scatter(drone_y, drone_x, marker='o', c='black', s=10)
    # Plot coverage range around each drone as a transparent circle
    for x, y in zip(drone_x, drone_y):
        circle = plt.Circle((y, x), coverage_range, color='blue', alpha=0.2)
        plt.gca().add_patch(circle)
    plt.title("Forest Fire and Drone Positions with Coverage Range")
    plt.xlabel("Y-coordinate")
    plt.ylabel("X-coordinate")
    plt.show()


############################## SIMULATION ANALYSIS ##############################
if __name__ == "__main__":
    
    # Simulation analysis control panel
    wind_strength = False
    changing_wind = True
    drone_count = False
    drone_coverage = False
    particles = False
    iterations = False
    inertia = False
    cognitive = False
    social = False
    
    #####################################################################################
    
    # Sensitivity Analysis: Wind Strength
    if wind_strength == True:
        # Initialze plot
        fig1, ax = plt.subplots(3)
        plt.suptitle('Fire Coverage for Increasing Wind Strength')
        ax[0].set_ylabel("Coverage [%]")
        ax[0].set_title('Weak Winds')
        ax[0].set_ylim([-5, 105])
        ax[1].set_ylabel("Coverage [%]")
        ax[1].set_title('Medium Winds')
        ax[1].set_ylim([-5, 105])
        ax[2].set_ylabel("Coverage [%]")
        ax[2].set_title('Strong Winds')
        ax[2].set_ylim([-5, 105])
        
        # Weak wind scenario 
        arrays = []
        for scenario in weak:
            fitness_values = []
            for i, fire_matrix in enumerate(scenario, start=1):
                previous_best_positions = None if i == 1 else best_position
                swarm = DroneSwarm(forest_size, previous_best_positions)
                best_position, best_fitness = swarm.run_optimization()
                previous_best_positions = best_position.copy()
                fitness_values.append(best_fitness)
            arrays.append(np.array(fitness_values))    
            ax[0].plot(fitness_values, lw=0.5, color='black')
        av_weak = sum(arrays)/len(arrays)    
        ax[0].plot(av_weak, lw=3, color='red')
        print(np.mean(av_weak))
       
        # Medium wind scenario 
        arrays = []
        for scenario in medium:
            fitness_values = []
            for i, fire_matrix in enumerate(scenario, start=1):
                previous_best_positions = None if i == 1 else best_position
                swarm = DroneSwarm(forest_size, previous_best_positions)
                best_position, best_fitness = swarm.run_optimization()
                previous_best_positions = best_position.copy()
                fitness_values.append(best_fitness)
            arrays.append(np.array(fitness_values))    
            ax[1].plot(fitness_values, lw=0.5, color='black')
        av_medium = sum(arrays)/len(arrays)    
        ax[1].plot(av_medium, lw=3, color='red')
        print(np.mean(av_medium))
       
        # Strong wind scenario 
        arrays = []
        for scenario in strong:
            fitness_values = []
            for i, fire_matrix in enumerate(scenario, start=1):
                previous_best_positions = None if i == 1 else best_position
                swarm = DroneSwarm(forest_size, previous_best_positions)
                best_position, best_fitness = swarm.run_optimization()
                previous_best_positions = best_position.copy()
                fitness_values.append(best_fitness)
            arrays.append(np.array(fitness_values))    
            ax[2].plot(fitness_values, lw=0.5, color='black')
        av_strong = sum(arrays)/len(arrays)    
        ax[2].plot(av_strong, lw=3, color='red')
        print(np.mean(av_strong))
        plt.show()
    
    #####################################################################################    
    
    # Sensitivity Analysis: Wind Change
    if changing_wind == True:
        # Initialze plot
        fig1, ax = plt.subplots(2)
        plt.suptitle('Fire Coverage for Changing Wind')
        ax[0].set_ylabel("Coverage [%]")
        ax[0].set_title('Without Wind Change')
        ax[0].set_ylim([-5, 105])
        ax[1].set_ylabel("Coverage [%]")
        ax[1].set_title('With Wind Change')
        ax[1].set_ylim([-5, 105])
        
        # Without wind change
        arrays = []
        for scenario in wind_no_change:
            fitness_values = []
            for i, fire_matrix in enumerate(scenario, start=1):
                previous_best_positions = None if i == 1 else best_position
                swarm = DroneSwarm(forest_size, previous_best_positions)
                best_position, best_fitness = swarm.run_optimization()
                previous_best_positions = best_position.copy()
                fitness_values.append(best_fitness)
            arrays.append(np.array(fitness_values))    
            ax[0].plot(fitness_values, lw=0.5, color='black')
        av_no_change = sum(arrays)/len(arrays)    
        ax[0].plot(av_no_change, lw=3, color='red')
        print(np.mean(av_no_change))
        
        # With wind change
        arrays = []
        for scenario in wind_change:
            fitness_values = []
            for i, fire_matrix in enumerate(scenario, start=1):
                previous_best_positions = None if i == 1 else best_position
                swarm = DroneSwarm(forest_size, previous_best_positions)
                best_position, best_fitness = swarm.run_optimization()
                previous_best_positions = best_position.copy()
                fitness_values.append(best_fitness)
            arrays.append(np.array(fitness_values))    
            ax[1].plot(fitness_values, lw=0.5, color='black')
        av_change = sum(arrays)/len(arrays)    
        ax[1].plot(av_change, lw=3, color='red')
        print(np.mean(av_change))
        plt.show()
    
    #####################################################################################
    
    # Sensitivity Analysis: Number of Drones
    if drone_count == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Number of Drones')
        ax.set_ylim([-5, 105])
        while num_drones < 16:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), lw=2, label=f'{num_drones} drones')  
            print(np.mean(sum(arrays)/len(arrays)))
            num_drones += 3
        plt.legend()
        plt.show()
    
    #####################################################################################
        
    # Sensitivity Analysis: Drone Coverage Range
    if drone_coverage == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Drone Coverage Range')
        ax.set_ylim([-5, 105])
        while coverage_range < 12:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), label=f'Range: {coverage_range}') 
            print(np.mean(sum(arrays)/len(arrays))) 
            coverage_range += 2
        plt.legend()
        plt.show()
        
    #####################################################################################
    
    # Sensitivity Analysis: Number of Particles
    if particles == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Number of Particles')
        ax.set_ylim([-5, 105])
        while num_particles < 35:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), label=f'{num_particles} particles')  
            print(np.mean(sum(arrays)/len(arrays))) 
            num_particles += 5
        plt.legend()
        plt.show()
    
    #####################################################################################
        
    # Sensitivity Analysis: Maximum Iterations
    if iterations == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Maximum Iterations')
        ax.set_ylim([-5, 105])
        while max_iterations < 30:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), lw=2, label=f'{max_iterations} iterations') 
            print(np.mean(sum(arrays)/len(arrays)))  
            max_iterations += 5
        plt.legend()
        plt.show()
    
    #####################################################################################    
    
    # Sensitivity Analysis: Inertia Weight (w)
    if inertia == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Inertia Weight (w)')
        ax.set_ylim([-5, 105])
        while inertia_weight < 1.0:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), lw=2, label=f'w: {inertia_weight}')  
            print(np.mean(sum(arrays)/len(arrays))) 
            inertia_weight += 0.1
        plt.legend()
        plt.show()
    
    #####################################################################################
        
    # Sensitivity Analysis: Cognitive Constant (c1)
    if cognitive == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Cognitive Constant (c1)')
        ax.set_ylim([-5, 105])
        while c1 < 2.05:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), label=f'c1: {c1}')  
            print(np.mean(sum(arrays)/len(arrays))) 
            c1 += 0.1
        plt.legend()
        plt.show()   
        
    #####################################################################################
        
    # Sensitivity Analysis: Social Constant (c2)
    if social == True:
        # Initialze plot
        fig1, ax = plt.subplots(1)
        ax.set_ylabel("Coverage [%]")
        ax.set_title('Coverage vs. Social Constant (c2)')
        ax.set_ylim([-5, 105])
        while c2 < 2.05:
            arrays = []
            for scenario in medium:
                fitness_values = []
                for i, fire_matrix in enumerate(scenario, start=1):
                    previous_best_positions = None if i == 1 else best_position
                    swarm = DroneSwarm(forest_size, previous_best_positions)
                    best_position, best_fitness = swarm.run_optimization()
                    previous_best_positions = best_position.copy()
                    fitness_values.append(best_fitness)
                arrays.append(np.array(fitness_values))    
            ax.plot(sum(arrays)/len(arrays), label=f'c2: {c2}')  
            print(np.mean(sum(arrays)/len(arrays))) 
            c2 += 0.1
        plt.legend()
        plt.show()   