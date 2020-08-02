# CrowdRL
crowd simulation using PPO algorithm.

***

## Implemented environments
1) basic - Basic environment with only one agent and target.
2) circle1 - An environment where 8 agents are located at the vertices that divide the circle into 8 equal parts, and targeting the opposite vertex.
3) circle2 - An environment where 8 agents are located at the 4 vertices and 4 center of sides of a square, and targeting the opposite vertex.
4) crossing1 - An environment where two groups of agents cross a narrow path.
5) crossing2 - An environment where two groups of agents cross each other orthogonally.
6) obstacles - An environment where 5 agents move toward the target avoiding obstacles.

## Dependencies
scipy 1.4.1 (for cdist, pdist)
PyOpenGL 3.1.5
torch 1.5.1

## Train model
python train_model.py

