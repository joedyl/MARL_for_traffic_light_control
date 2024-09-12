## Using Rays RLlib to implement DQN on pistonball

This is attempting to train DQN agents on the Pettingzoo pistonball environment.

Code inspired from:
Author: Rohan (https://github.com/Rohan138)

# Installation and Usage

Clone this repository and ensure vscode is installed. You may also need to ensure docker desktop is installed and running and that you have installed WSL 2 on your host system. Within vscode, install the dev containers and docker extension and the remote development extension pack. 
Use ctrl shift P and open folder in container. If this doesnt work, press the + button when in the dev containers drop down option in the remote development tab on the left hand side of your vs code. Select open folder in container and choose this repos folder. The container should set up. When you see Dev Container: folder open in container on the bottom left of your window within the little light blue box, you are ready to go. To run any file, make sure youre in the src and type python _name_.py into the terminal.

# Current env state
The config for the pistonball environment is within the folder, so that it can be adjusted if necessary. The only current changes is continuous = False because DQN does not deal with continuous control well at all. Also, use the env, not the parallel env, when interacting with the pistonball environment, else a config problem will arise because of this

# Current project state
This project has been discontinued as it will not properly work on pistonball as it can only train agents when the AECEnv is in use. Ray does not react well to the parallel env if the ParallelPettingZooEnv wrapper from ray is on it. This means it is not being treated as a multi-agent system as each agent knows what each other agent has done at each step. Wandb is configured on this to be able to log returns if you so wish to pick this up and continue with it. 

# Goals for future

I would like to try to implement parameter sharing properly with this if possible. I think the multi-agent part in Ray does this (possibly, look into this) but the DQN config does not react well with the parallel env so that needs configuring. That is the overall goal. 


