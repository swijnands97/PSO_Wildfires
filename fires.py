"""
This file generates all forest fire scenarios based on wind strength, wind direction, and changes in wind.
5 types of wind scenarious are generated, and 50 randomized versions are created for each.

    1. Weak wind (no change, random direction)
    
    2. Medium wind (no change, random direction)
    
    3. Strong wind (no change, random direction)
    
    4. Wind without change (random strength, random direction)
    
    5. Wind with change (random strength, random direction)
"""

# Import forest_area and ForestFire class
from core_parameters import forest_area
from forest_fire import ForestFire

# Import relevant python package
import pickle

# Weak wind (no change, random direction)
weak = []
for f in range(25):
    fire = []
    f = ForestFire(forest_area, 'weak', 'random')
    f.start()
    fire.append(f.flames.copy())
    for i in range(30):
        f.spread()
        fire.append(f.flames.copy())
    for j in range(15):
        f.reduce()
        fire.append(f.flames.copy())
    weak.append(fire)
with open('weak.pkl', 'wb') as file:
    pickle.dump(weak, file)

    
# Medium wind (no change, random direction)
medium = []
for f in range(25):
    fire = []
    f = ForestFire(forest_area, 'medium', 'random')
    f.start()
    fire.append(f.flames.copy())
    for i in range(30):
        f.spread()
        fire.append(f.flames.copy())
    for j in range(15):
        f.reduce()
        fire.append(f.flames.copy())
    medium.append(fire)
with open('medium.pkl', 'wb') as file:
    pickle.dump(medium, file)


# Strong wind (no change, random direction)
strong = []
for f in range(25):
    fire = []
    f = ForestFire(forest_area, 'strong', 'random')
    f.start()
    fire.append(f.flames.copy())
    for i in range(30):
        f.spread()
        fire.append(f.flames.copy())
    for j in range(15):
        f.reduce()
        fire.append(f.flames.copy())
    strong.append(fire)
with open('strong.pkl', 'wb') as file:
    pickle.dump(strong, file)


# Wind without change (medium strength, random direction)
wind_no_change = []
for f in range(25):
    fire = []
    f = ForestFire(forest_area, 'medium', 'random')
    f.start()
    fire.append(f.flames.copy())
    for i in range(30):
        f.spread()
        fire.append(f.flames.copy())
    for j in range(15):
        f.reduce()
        fire.append(f.flames.copy())
    wind_no_change.append(fire)
with open('wind_no_change.pkl', 'wb') as file:
    pickle.dump(wind_no_change, file)

# Wind with change (medium strength, random direction)
wind_change = []
for f in range(25):
    fire = []
    f = ForestFire(forest_area, 'medium', 'random')
    f.start()
    fire.append(f.flames.copy())
    for i in range(20):
        f.spread()
        fire.append(f.flames.copy())
    # Change the wind direction    
    f.change_wind()
    for i in range(10):
        f.spread()
        fire.append(f.flames.copy())
    for j in range(15):
        f.reduce()
        fire.append(f.flames.copy())
    wind_change.append(fire)
with open('wind_change.pkl', 'wb') as file:
    pickle.dump(wind_change, file)

### End of File ###