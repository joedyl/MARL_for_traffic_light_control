import os

import ray
from gymnasium.spaces import Box, Discrete
from ray import tune, train
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.env import PettingZooEnv
from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.fcnet import FullyConnectedNetwork as TorchFC
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.torch_utils import FLOAT_MAX
from ray.tune.registry import register_env
from ray.air.integrations.wandb import WandbLoggerCallback
from ray.tune import Callback

import pistonball
# from pettingzoo.butterfly import pistonball_v6 - same as above import, above allows env modification

torch, nn = try_import_torch()


class TorchMaskedActions(DQNTorchModel):
    """PyTorch version of above ParametricActionsModel."""

    def __init__(
        self,
        obs_space: Box,
        action_space: Discrete,
        num_outputs,
        model_config,
        name,
        **kw,
    ):
        DQNTorchModel.__init__(
            self, obs_space, action_space, num_outputs, model_config, name, **kw
        )
        print("obs space shape: ", obs_space.shape)
        obs_len = obs_space.shape[0] - action_space.n
        flat_low = obs_space.low[:obs_len].reshape(-1)
        flat_high = obs_space.high[:obs_len].reshape(-1)

        orig_obs_space = Box(
            shape=(flat_low.size,), low=flat_low, high=flat_high  
        )
        self.action_embed_model = TorchFC(
            orig_obs_space,
            action_space,
            action_space.n,
            model_config,
            name + "_action_embed",
        )

    def forward(self, input_dict, state, seq_lens):
        total_actions = self.action_space.n
        obs_tensor = input_dict["obs"]
        print("shape of obs tensor: ", obs_tensor.shape)
        action_mask = obs_tensor[:, -total_actions:]  # Extract last 'n' elements as action mask
        observations = obs_tensor[:, :-total_actions]  # Rest is observations
        
        # Compute the predicted action embedding
        action_logits, _ = self.action_embed_model({"obs": observations})
        
        # Ensure dimensions are compatible
        # print("Shape of action_logits:", action_logits.shape)
        # print("Shape of action_mask:", action_mask.shape)

        action_mask = action_mask.max(dim=3)[0].max(dim=2)[0]
        # print("Shape of action_mask:", action_mask.shape)

        # turns probit action mask into logit action mask
        inf_mask = torch.clamp(torch.log(action_mask), -1e10, FLOAT_MAX)
        # print("Shape of inf_mask:", inf_mask.shape)
    
        # Ensure addition is dimensionally possible
        if action_logits.shape != inf_mask.shape:
            print("Mismatch in tensor shapes:", action_logits.shape, "vs", inf_mask.shape)
            # Additional handling may be necessary here, e.g., reshaping
            
        return action_logits + inf_mask, state
        
    def value_function(self):
        return self.action_embed_model.value_function()


if __name__ == "__main__":
    ray.init()

    alg_name = "DQN"
    ModelCatalog.register_custom_model("pa_model", TorchMaskedActions)
    # function that outputs the environment you wish to register.

    def env_creator():
        env = pistonball.env()
        return env

    env_name = "pistonball"
    register_env(env_name, lambda config: PettingZooEnv(env_creator()))

    test_env = PettingZooEnv(env_creator())
    obs_space = test_env.observation_space
    act_space = test_env.action_space

    class CheckpointLogger(Callback):
        def on_checkpoint(self, iteration, trials, trial, **info):
            print("Checkpoint saved:", trial.checkpoint.value)

    config = (
        DQNConfig()
        .environment(env=env_name)
        .rollouts(num_rollout_workers=1, rollout_fragment_length=30)
        .training(
            train_batch_size=200,
            hiddens=[],
            dueling=False,
            model={"custom_model": "pa_model"},
        )
        .multi_agent(
            policies={
                "player_0": (None, obs_space, act_space, {}),
                "player_1": (None, obs_space, act_space, {}),
                "player_2": (None, obs_space, act_space, {}),
                "player_3": (None, obs_space, act_space, {}),
                "player_4": (None, obs_space, act_space, {}),
                "player_5": (None, obs_space, act_space, {}),
                "player_6": (None, obs_space, act_space, {}),
                "player_7": (None, obs_space, act_space, {}),
                "player_8": (None, obs_space, act_space, {}),
                "player_9": (None, obs_space, act_space, {}),
            },
            policy_mapping_fn=(lambda agent_id, *args, **kwargs: agent_id),
        )
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(
            log_level="DEBUG"
        )  # TODO: change to ERROR to match pistonball example
        .framework(framework="torch")
        .exploration(
            exploration_config={
                # The Exploration class to use.
                "type": "EpsilonGreedy",
                # Config for the Exploration class' constructor:
                "initial_epsilon": 0.1,
                "final_epsilon": 0.0,
                "epsilon_timesteps": 100000,  # Timesteps over which to anneal epsilon.
            }
        )
    )

    tune.run(
        alg_name,
        name="DQN",
        stop={"training_iteration": 9, "timesteps_total": 1000000 if not os.environ.get("CI") else 10000},
        checkpoint_freq=3,
        config=config.to_dict(),
        callbacks = [CheckpointLogger(), WandbLoggerCallback(
            project="uwe-ai",
            api_key="enter_upon_login",
            log_config=True
        )]
    )