import torch
from agilerl.algorithms.maddpg import MADDPG

import env.graph_traffic_env as custom
import wandb


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Configure the environment
    env = custom.parallel_env(render_mode="human")
    env.reset()


    state_dim = [env.observation_space(agent).shape for agent in env.possible_agents]
    one_hot = False


    action_dim = [env.action_space(agent).n for agent in env.agents]
    discrete_actions = True
    max_action = None
    min_action = None

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    n_agents = len(env.possible_agents)
    agent_ids = env.possible_agents

    # Instantiate an MADDPG object
    maddpg = MADDPG(
        state_dim,
        action_dim,
        one_hot,
        n_agents,
        agent_ids,
        max_action,
        min_action,
        discrete_actions,
        device=device,
    )

    # Load the saved algorithm into the MADDPG object
    path = "./models/MADDPG/MADDPG_trained_agent.pt"
    maddpg.loadCheckpoint(path)

    # Define test loop parameters
    episodes = 10  # Number of episodes to test agent on
    max_steps = 10  # Max number of steps to take in the environment in each episode

    rewards = []  # List to collect total episodic reward
    frames = []  # List to collect frames
    indi_agent_rewards = {
        agent_id: [] for agent_id in agent_ids
    }  # Dictionary to collect inidivdual agent rewards

    wandb.init(
    project="uwe-ai",   #replace project and entity with own
    entity= 'joe-lyall22',
    config= {"test_episodes": episodes,
            "steps": max_steps})
    
    # Test loop for inference
    for ep in range(episodes):
        state, info = env.reset()
        agent_reward = {agent_id: 0 for agent_id in agent_ids}
        score = 0
        for _ in range(max_steps):

            agent_mask = info["agent_mask"] if "agent_mask" in info.keys() else None
            env_defined_actions = (
                info["env_defined_actions"]
                if "env_defined_actions" in info.keys()
                else None
            )

            # Get next action from agent
            cont_actions, discrete_action = maddpg.getAction(
                state,
                epsilon=0,
                agent_mask=agent_mask,
                env_defined_actions=env_defined_actions,
            )
            if maddpg.discrete_actions:
                action = discrete_action
            else:
                action = cont_actions

            # Save the frame for this step and append to frames list
            env.render()

            # Take action in environment
            state, reward, termination, truncation, info = env.step(action)

            # Save agent's reward for this step in this episode
            for agent_id, r in reward.items():
                agent_reward[agent_id] += r

            # Determine total score for the episode and then append to rewards list
            score = sum(agent_reward.values())

            # Stop episode if any agents have terminated
            if any(truncation.values()) or any(termination.values()):
                break

        rewards.append(score)

        # Record agent specific episodic reward for each agent
        for agent_id in agent_ids:
            indi_agent_rewards[agent_id].append(agent_reward[agent_id])

        print("-" * 15, f"Episode: {ep}", "-" * 15)
        print("Episodic Reward: ", rewards[-1])
        for agent_id, reward_list in indi_agent_rewards.items():
            print(f"{agent_id} reward: {reward_list[-1]}")
        wandb.log({"test_returns": rewards[-1]})
    env.close()
    wandb.finish()