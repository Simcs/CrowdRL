# CrowdRL
crowd simulation using PPO algorithm.

***

## Environments
- ***basic*** - Basic environment with only one agent and target.
- ***circle1*** - An environment where 8 agents are located at the vertices that divide the circle into 8 equal parts, and targeting the opposite vertex.
- ***circle2*** - An environment where 8 agents are located at the 4 vertices and 4 center of sides of a square, and targeting the opposite vertex.
- ***crossing1*** - An environment where two groups of agents cross a narrow path.
- ***crossing2*** - An environment where two groups of agents cross each other orthogonally.
- ***obstacles*** - An environment where 5 agents move toward the target avoiding obstacles.

## Dependencies
scipy 1.4.1 (for cdist, pdist)  
PyOpenGL 3.1.5  
torch 1.5.1  

## Train model
```bash
python .\train_model.py --env=<environment_name> --path=<model_save_location> --model=<model_load_location>
```
### options
- ***-e***/***--env*** (optional) : The name of the environment to use.
- ***-p***/***--path*** (optional) : Path to save learning results(model).
- ***-m***/***--model*** (optional) : Path of model to import.

### Example
```bash
python .\train_model.py -m '.\checkpoints\basic\iteration-399,avg_reward--546.879.dat' -p .\checkpoints\200729\
```

## Test model
```bash
python .\test_model.py --model=<test_model_location> --env=<environment_name>
```
### options
- ***-m***/***--model*** (required) : Path of model to test.
- ***-e***/***--env*** (optional) : The name of the environment to use.
- ***-r***/***--render*** (optional) : Whether to render result or not. (Default=True)

### Example
```bash
 python .\test_model.py -e circle1 -m '.\checkpoints\200729\iteration-129,avg_reward--575.981.dat'
```
