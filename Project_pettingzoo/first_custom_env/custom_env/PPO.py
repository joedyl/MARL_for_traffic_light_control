import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import env.graph_traffic_env as custom
import wandb


class Agent(nn.Module):
    def __init__(self, num_actions, input_dim):
        super().__init__()

        self.network = nn.Sequential(
            self._layer_init(nn.Linear(input_dim, 64)),
            nn.ReLU(),
            self._layer_init(nn.Linear(64, 64)),
            nn.ReLU(),
        )
        self.actor = self._layer_init(nn.Linear(64, num_actions), std=0.01)
        self.critic = self._layer_init(nn.Linear(64, 1))

    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer

    def get_value(self, x):
        return self.critic(self.network(x))

    def get_action_and_value(self, x, action=None):
        hidden = self.network(x)
        logits = self.actor(hidden)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action), probs.entropy(), self.critic(hidden)


def batchify_obs(obs, device, max_length):
    """Converts PZ style observations to batch of torch arrays with padding."""
    padded_obs = []
    for o in obs.values():
        padded_o = np.pad(o, (0, max(0, max_length - len(o))), 'constant', constant_values=0)
        padded_obs.append(padded_o)
    obs = np.stack(padded_obs, axis=0)
    obs = torch.tensor(obs, dtype=torch.float32).to(device)
    return obs


def batchify(x, device):
    """Converts PZ style returns to batch of torch arrays."""
    # convert to list of np arrays
    x = np.stack([x[a] for a in x], axis=0)
    # convert to torch
    x = torch.tensor(x).to(device)

    return x


def unbatchify(x, env):
    """Converts np array to PZ style arguments."""
    x = x.cpu().numpy()
    x = {a: x[i] for i, a in enumerate(env.possible_agents)}
    return x


if __name__ == "__main__":
    """ALGO PARAMS"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    LR = 0.001
    epsilon = 1e-5
    batch_size = 64
    max_cycles = 10
    total_episodes = 3000


    """ ENV SETUP """
    env = custom.parallel_env()
    num_agents = len(env.possible_agents)
    num_actions = env.action_space(env.possible_agents[0]).n

    input_dim = max([env.observation_space(agent).shape[0] for agent in env.possible_agents]) # Assuming MultiDiscrete observations

    """ LEARNER SETUP """
    agent = Agent(num_actions=num_actions, input_dim=input_dim).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=LR, eps=epsilon)

    """WAND LOGGING"""
    wandb.init(
    # # Set the project where this run will be logged
    project="uwe-ai",
    # # Track hyperparameters and run metadata
    entity='joe-lyall22', 
    config={
        "episodes": total_episodes,
        "steps": max_cycles,
        "exploration coef": ent_coef,
        "discount factor": gamma,
        "agents": num_agents,
        "learning rate": LR, 
        "optimiser epsilon": epsilon
     })


    # train for n number of episodes
    for episode in range(total_episodes):
        rb_obs = torch.zeros((max_cycles, num_agents, input_dim)).to(device)
        rb_actions = torch.zeros((max_cycles, num_agents)).to(device)
        rb_logprobs = torch.zeros((max_cycles, num_agents)).to(device)
        rb_rewards = torch.zeros((max_cycles, num_agents)).to(device)
        rb_terms = torch.zeros((max_cycles, num_agents)).to(device)
        rb_values = torch.zeros((max_cycles, num_agents)).to(device)
        total_episodic_return = 0
        clip_fracs = []

        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            next_obs, info = env.reset()
            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                obs = batchify_obs(next_obs, device, input_dim) 

                # get action from the agent
                actions, logprobs, _, values = agent.get_action_and_value(obs)

                # execute the environment and log data
                next_obs, rewards, terms, truncs, infos = env.step(
                    unbatchify(actions, env)
                )

                # add to episode storage
                rb_obs[step] = obs
                rb_rewards[step] = batchify(rewards, device)
                rb_terms[step] = batchify(terms, device)
                rb_actions[step] = actions
                rb_logprobs[step] = logprobs
                rb_values[step] = values.flatten()

                # compute episodic return
                total_episodic_return += rb_rewards[step].cpu().numpy()
                
                # if we reach termination or truncation, end
                if any(terms.values()) or any(truncs.values()):
                    end_step = step + 1
                    break
                else:
                    end_step = max_cycles

            rb_advantages = torch.zeros_like(rb_rewards).to(device)
            rb_returns = torch.zeros_like(rb_rewards).to(device)
            last_value = rb_values[end_step - 1] if end_step < max_cycles else torch.zeros_like(rb_values[0])
            last_advantage = 0

            for t in reversed(range(end_step)):
                if t == end_step - 1:
                    next_non_terminal = 1.0 - rb_terms[t]
                    next_value = last_value
                else:
                    next_non_terminal = 1.0 - rb_terms[t + 1]
                    next_value = rb_values[t + 1]
                delta = rb_rewards[t] + gamma * next_value * next_non_terminal - rb_values[t]
                last_advantage = delta + gamma * next_non_terminal * last_advantage
                rb_advantages[t] = last_advantage

            rb_returns = rb_advantages + rb_values

        # convert our episodes to batch of individual transitions
        b_obs = torch.flatten(rb_obs[:end_step], start_dim=0, end_dim=1)
        b_logprobs = torch.flatten(rb_logprobs[:end_step], start_dim=0, end_dim=1)
        b_actions = torch.flatten(rb_actions[:end_step], start_dim=0, end_dim=1)
        b_returns = torch.flatten(rb_returns[:end_step], start_dim=0, end_dim=1)
        b_values = torch.flatten(rb_values[:end_step], start_dim=0, end_dim=1)
        b_advantages = torch.flatten(rb_advantages[:end_step], start_dim=0, end_dim=1)

        # Optimizing the policy and value network
        b_index = np.arange(len(b_obs))
        for repeat in range(3):
            # shuffle the indices we use to access the data
            np.random.shuffle(b_index)
            for start in range(0, len(b_obs), batch_size):
                # select the indices we want to train on
                end = start + batch_size
                batch_index = b_index[start:end]

                _, newlogprob, entropy, value = agent.get_action_and_value(
                    b_obs[batch_index], b_actions.long()[batch_index]
                )
                logratio = newlogprob - b_logprobs[batch_index]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clip_fracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

                # normalize advantaegs
                advantages = b_advantages[batch_index]
                advantages = (advantages - advantages.mean()) / (
                    advantages.std() + 1e-8
                )

                # Policy loss
                pg_loss1 = -advantages * ratio
                pg_loss2 = -advantages * torch.clamp(
                    ratio, 1 - clip_coef, 1 + clip_coef
                )
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                v_loss = 0.5 * ((value.flatten() - b_returns[batch_index]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        
        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        print(f"Training episode {episode}")
        print(f"Episodic Return: {np.mean(total_episodic_return)}")
        print(f"Episode Length: {end_step}")
        print("")
        print(f"Value Loss: {v_loss.item()}")
        print(f"Policy Loss: {pg_loss.item()}")
        print(f"Old Approx KL: {old_approx_kl.item()}")
        print(f"Approx KL: {approx_kl.item()}")
        print(f"Clip Fraction: {np.mean(clip_fracs)}")
        print(f"Explained Variance: {explained_var}")
        print("\n-------------------------------------------\n")
        wandb.log({"Return": np.mean(total_episodic_return), "Value loss": v_loss.item()})

    wandb.finish()
        
    """ RENDER THE POLICY """
    env = custom.parallel_env()
    agent.eval()
    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            obs, infos = env.reset()
            obs = batchify_obs(obs, device, input_dim)
            terms = [False]
            truncs = [False]
            while not any(terms) and not any(truncs):
                actions, _, _, _ = agent.get_action_and_value(obs)

                obs, rewards, terms, truncs, infos = env.step(unbatchify(actions, env))
                obs = batchify_obs(obs, device, input_dim)
                terms = [terms[a] for a in terms]
                truncs = [truncs[a] for a in truncs]