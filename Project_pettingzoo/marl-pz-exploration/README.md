# marl-pz-exploration

Boilerplate code for running a containerised marl environment based on petting zoo library. Code adjusted from original code by [https://github.com/jjshoots](https://github.com/jjshoots).

[Wandb](https://wandb.ai/site) also configured for logging and capturing training performance data.

pistonball env code added to be able to customise environment. Code from [https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/pistonball/pistonball.py](https://github.com/Farama-Foundation/PettingZoo/blob/master/pettingzoo/butterfly/pistonball/pistonball.py)

# Installation and Usage

Clone this repository and ensure vscode is installed. You may also need to ensure docker desktop is installed and running and that you have installed WSL 2 on your host system. Within vscode, install the dev containers and docker extension and the remote development extension pack. 
Use ctrl shift P and open folder in container. If this doesnt work, press the + button when in the dev containers drop down option in the remote development tab on the left hand side of your vs code. Select open folder in container and choose this repos folder. The container should set up. When you see Dev Container: folder open in container on the bottom left of your window within the little light blue box, you are ready to go. To run the PPO algorithm, run the test-pz-marl.py file or python test-pz-marl.py in the terminal from within the src folder of the repo. To change the environment variables, see pistonball.py. 


