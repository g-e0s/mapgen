## MapGen - a simple 2D map generator environment for RL algorithms
### Usage
```
import numpy as np
from mapgen import Dungeon

# create small map
env = Dungeon(16, 16, min_room_xy=7, max_room_xy=14)
obs = env.reset()

# 100 random steps
for _ in range(1000):
    action = np.random.choice(3)
    obs, reward, done, info = env.step(action)
    env.render("human")
    if done:
        break
```
### Observation layers
 - Unexplored cells
 - Free cells
 - Occupied cells
 - Visited cells