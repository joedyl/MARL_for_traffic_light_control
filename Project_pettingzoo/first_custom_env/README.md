# **MARL for traffic light control and optimisation**

# Description
This is a custom environment built to simulate co-ordination between a few agents acting as traffic lights. It utilises Dockerfiles to run pettingzoo environments in dev containers and implement an algorithm to enable the agents to interact with the environment.

# Installation
Clone this repository. To make setup easy, vscode is the desired IDE. Within vscode, install the dev containers extension pack and the Docker extension. The docker extension will tell you to install docker desktop on your computer. This needs to be running at all times when you use the dev containers extension. To make things easier, install the remote explorer extension pack. This can interact with dev containers and the remote explorer tab that appears on the left hand side of your vs code will remember any dev containers you build. To build the container, type ctrl+shift+P and select Dev Containers: Open folder in container or press the + button in the Dev containers drop down within the remote explorer tab in your vscode. Select the folder that contains this repository and the container will build for you. It may take a number of minutes. 

# Usage
Once inside the container, change directory to the custom_env folder if you want to test an algorithm against the environment or see a sample of the environment in action. Either type python (_filename_).py in a terminal or press the run button in the top right corner of vs code on the python file you want to run. If you want to test the api of the environment, make sure your terminal is in the env folder (PATH = first_custom_env/custom_env/env)

# Current project state
So far, an environment has been finalised: graph_traffic_env. testing.py in the env folder provides the parallel api test, which is what is needed for a Multi Agent environment, as all agents act at the same time. The env is sub-classing the base AECEnv and wrapping with a parallel wrapper, because the AECEnv provides useful inbuilt functions that the ParallelEnv itself does not have.

The custom_env folder now has sample_behaviour.py, which is useful for testing the environment as it just uses basic sample actions of the environment. PPO (Code from: https://github.com/jjshoots) is a working algorithm that has been configured. train.py and render.py are for a MADDPG algorithm, (author credits in the py files) and are fully configured. These are from (https://www.agilerl.com/) initially and the code has been adapted from Pettingzoos documentation (https://pettingzoo.farama.org/tutorials/agilerl/MADDPG/) Both have wandb imported which you can set up for logging if you wish. You need to change the entity and the project name to be your own. .

The graph traffic env, as you will see, has 2 render modes. Graph, saves an image of the graph to a newly created folder in the current path. Human, prints out traffic light state and flow results to terminal.

Currently, the graph is configured to be a lattice, or grid shape if the number of agents is a square number, else it returns a random graph. Feel free to add other graph topologies to the generate_graph function and experiment with them.

# Future development

If anyone wishes to work on this project and wants to implement a new algorithm or environment, please put algorithm code in custom_env folder and the environment code in the env folder. 
If you would like to see a render mode in action with an algorithm, specify the render mode where the env is initialised in the algorithm files. Currently it is empty, which will run still, just with no rendered output.
Make sure to add any necessary requirements to requirements.txt and rebuild the docker container. The command to do this in vscode is the tools button next to the section describing your open dev container in the remote explorer tab.

# Authors
Joe Lyall. Assisted by Marco Perez Hernandez.

# License 
Project funded by UWE Bristol

