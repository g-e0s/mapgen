## MapGen - a simple 2D map generator environment for RL algorithms

### Installation

To install `MapGen`, create or activate target environment and do the following steps:
1. Clone repository to prefer place:
```shell
git clone https://github.com/g-e0s/mapgen.git
```
2. Go to repository root
```shell
cd mapgen
```
3. Install package with `pip`
```shell
pip install -e.
```

### Usage
```python
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