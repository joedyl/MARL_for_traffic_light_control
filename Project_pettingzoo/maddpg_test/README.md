## Using AgileRL to help train MADDPG agents on Pistonball

Code inspired from Pettingzoos AgileRL tutorial: (https://pettingzoo.farama.org/tutorials/agilerl/MADDPG/)
Author credits given in py files

# Installation and Usage

Clone this repository into a folder you choose and ideally, ensure vscode is installed. You may also need to ensure docker desktop is installed and running and that you have installed WSL 2 on your host system. Within vscode, install the dev containers and docker extension and the remote development extension pack. 
Use ctrl shift P and open folder in container. If this doesnt work, press the + button when in the dev containers drop down option in the remote development tab on the left hand side of your vs code. Choose this repos folder. The container should set up. When you see Dev Container: folder open in container on the bottom left of your window within the little light blue box, you are ready to go. To run any file, make sure youre in the src and type python _filename_.py into the terminal.

# Current env state
The config for the pistonball environment is within the folder, so that it can be adjusted if necessary. The only current changes is continuous = False and n_pistons = 10 for fair tests in my experiments i used it for

# Current project state
This project has been discontinued as it has served its purpose. train.py and render.py work with pistonball environment and have wandb configured for logging. Make sure to change project and entity to your own details.