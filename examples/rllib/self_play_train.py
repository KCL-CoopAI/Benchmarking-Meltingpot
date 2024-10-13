# Copyright 2020 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Runs an example of a self-play training experiment."""

import os

from meltingpot import substrate
import ray
from ray import air
from ray import tune
from ray.rllib.algorithms import ppo,a3c,appo
from ray.rllib.policy import policy
from ray.air.integrations.wandb import WandbLoggerCallback
import utils
import torch
import argparse
import random
import numpy as np

def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    # torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")


def parse_args():
    parser = argparse.ArgumentParser("Stable-Baselines3 PPO with Parameter Sharing")
    parser.add_argument(
        "--env-name",
        type=str,
        default="coins",
        choices=['factory_commons__either_or', 'territory__inside_out', 'clean_up', 'chemistry__three_metabolic_cycles', 'chicken_in_the_matrix__repeated', 'chemistry__two_metabolic_cycles_with_distractors', 'territory__open', 'predator_prey__orchard', 'commons_harvest__open', 
                 'running_with_scissors_in_the_matrix__one_shot', 'pure_coordination_in_the_matrix__arena', 'predator_prey__open', 'boat_race__eight_races', 'stag_hunt_in_the_matrix__arena', 'collaborative_cooking__crowded', 'predator_prey__alley_hunt', 'commons_harvest__closed', 
                 'predator_prey__random_forest', 'pure_coordination_in_the_matrix__repeated', 'chicken_in_the_matrix__arena', 'gift_refinements', 'coop_mining', 'fruit_market__concentric_rivers', 'prisoners_dilemma_in_the_matrix__arena', 'rationalizable_coordination_in_the_matrix__repeated', 
                 'prisoners_dilemma_in_the_matrix__repeated', 'externality_mushrooms__dense', 'rationalizable_coordination_in_the_matrix__arena', 'bach_or_stravinsky_in_the_matrix__arena', 'bach_or_stravinsky_in_the_matrix__repeated', 'collaborative_cooking__asymmetric', 
                 'collaborative_cooking__cramped', 'paintball__king_of_the_hill', 'collaborative_cooking__forced', 'chemistry__two_metabolic_cycles', 'chemistry__three_metabolic_cycles_with_plentiful_distractors', 'paintball__capture_the_flag', 'commons_harvest__partnership', 
                 'hidden_agenda', 'collaborative_cooking__figure_eight', 'running_with_scissors_in_the_matrix__arena', 'collaborative_cooking__circuit', 'coins', 'stag_hunt_in_the_matrix__repeated', 'daycare', 'territory__rooms', 'running_with_scissors_in_the_matrix__repeated', 
                 'collaborative_cooking__ring', 'allelopathic_harvest__open'],
        help="The SSD environment to use",
    )
    parser.add_argument(
        "--num-agents",
        type=int,
        default=2,
        help="The number of agents",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=4,
        help="The number of cpus",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="The number of workers",
    )
    parser.add_argument(
        "--kl-threshold",
        type=float,
        default=0.01,
        help="The number of envs",
    )
    parser.add_argument(
        "--rollout-len",
        type=int,
        default=1000,
        help="length of training rollouts AND length at which env is reset",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=10000000,
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--total-iterations",
        type=int,
        default=7800, # approximately 0.5x1e9 timesteps
        help="Number of environment timesteps",
    )
    parser.add_argument(
        "--use-collective-reward",
        type=bool,
        default=False,
        help="Give each agent the collective reward across all agents",
    )
    parser.add_argument(
        "--inequity-averse-reward",
        type=bool,
        default=False,
        help="Use inequity averse rewards from 'Inequity aversion \
            improves cooperation in intertemporal social dilemmas'",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=5,
        help="Advantageous inequity aversion factor",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=0.05,
        help="Disadvantageous inequity aversion factor",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use_wandb", type=bool, default=False)
    parser.add_argument("--local_mode", type=bool, default=False)
    parser.add_argument("--use_lstm", type=bool, default=False)
    parser.add_argument("--user_name", type=str, default="shuqing-shi")
    parser.add_argument("--alg", type=str, default='PPO', choices=['PPO', 'A3C', 'APPO'])
    args = parser.parse_args()
    return args

def get_config(
    substrate_name: str = "coins",
    alg: str = 'PPO',
    num_rollout_workers: int = 2,
    rollout_fragment_length: int = 2000,
    train_batch_size: int = 131072,  # Doubled
    fcnet_hiddens=(128, 128),  # Increased depth and width
    post_fcnet_hiddens=(128, 128),  # Increased depth and width
    lstm_cell_size: int = 128,  # Increased
    sgd_minibatch_size: int = 4096,  # Doubled
    use_lstm: bool = True,  # Enabled LSTM
    num_sgd_iter: int = 60,  # Increased
    lr: float = 3e-4,  # Explicitly set learning rate
    vf_clip_param: float = 10.0,
    clip_param: float = 0.3,
):
  """Get the configuration for running an agent on a substrate using RLLib.

  We need the following 2 pieces to run the training:

  Args:
    substrate_name: The name of the MeltingPot substrate, coming from
      `substrate.AVAILABLE_SUBSTRATES`.
    num_rollout_workers: The number of workers for playing games.
    rollout_fragment_length: Unroll time for learning.
    train_batch_size: Batch size (batch * rollout_fragment_length)
    fcnet_hiddens: Fully connected layers.
    post_fcnet_hiddens: Layer sizes after the fully connected torso.
    lstm_cell_size: Size of the LSTM.
    sgd_minibatch_size: Size of the mini-batch for learning.

  Returns:
    The configuration for running the experiment.
  """
  # Gets the default training configuration
  if alg == 'PPO':
    config = ppo.PPOConfig()
  elif alg == 'A3C':
    config = a3c.A3CConfig()
  elif alg == 'APPO':
    config = appo.APPOConfig()
  # Number of arenas.
  config.num_rollout_workers = num_rollout_workers
  # This is to match our unroll lengths.
  config.rollout_fragment_length = 'auto'
  # Total (time x batch) timesteps on the learning update.
  config.train_batch_size = train_batch_size
  # Mini-batch size.
  config.sgd_minibatch_size = sgd_minibatch_size
  # Use the raw observations/actions as defined by the environment.
  config.preprocessor_pref = None
  # Use TensorFlow as the tensor framework.
  config = config.framework("torch")
  # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
  # config.num_gpus = int(os.environ.get("RLLIB_NUM_GPUS", "0"))
  config.num_gpus = 1
  config.log_level = "DEBUG"

  # 2. Set environment config. This will be passed to
  # the env_creator function via the register env lambda below.
  player_roles = substrate.get_config(substrate_name).default_player_roles
  config.env_config = {"substrate": substrate_name, "roles": player_roles}

  config.env = "meltingpot"

  # 4. Extract space dimensions
  test_env = utils.env_creator(config.env_config)

  # Setup PPO with policies, one per entry in default player roles.
  policies = {}
  player_to_agent = {}
  for i in range(len(player_roles)):
    rgb_shape = test_env.observation_space[f"player_{i}"]["RGB"].shape
    sprite_x = rgb_shape[0] // 8
    sprite_y = rgb_shape[1] // 8
    # sprite_x = rgb_shape[0] 
    # sprite_y = rgb_shape[1] 

    policies[f"agent_{i}"] = policy.PolicySpec(
        policy_class=None,  # use default policy
        observation_space=test_env.observation_space[f"player_{i}"],
        action_space=test_env.action_space[f"player_{i}"],
        config={
            "model": {
                "conv_filters": [[16, [8, 8], 8],
                                 [128, [sprite_x, sprite_y], 1]],
            },
        # config={
        #     "model": {
        #         "conv_filters": [[16, [3, 3], 1],
        #                          [64, [sprite_x * 8, sprite_y * 8], 1]],
        #     },
        })
    player_to_agent[f"player_{i}"] = f"agent_{i}"

  def policy_mapping_fn(agent_id, **kwargs):
    del kwargs
    return player_to_agent[agent_id]

  # 5. Configuration for multi-agent setup with one policy per role:
  config.multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)

  # 6. Set the agent architecture.
  # Definition of the model architecture.
  # The strides of the first convolutional layer were chosen to perfectly line
  # up with the sprites, which are 8x8.
  # The final layer must be chosen specifically so that its output is
  # [B, 1, 1, X]. See the explanation in
  # https://docs.ray.io/en/latest/rllib-models.html#built-in-models. It is
  # because rllib is unable to flatten to a vector otherwise.
  # The acb models used as baselines in the meltingpot paper were not run using
  # rllib, so they used a different configuration for the second convolutional
  # layer. It was 32 channels, [4, 4] kernel shape, and stride = 1.
  config.model["fcnet_hiddens"] = fcnet_hiddens
  config.model["fcnet_activation"] = "relu"
  config.model["conv_activation"] = "relu"
  config.model["post_fcnet_hiddens"] = post_fcnet_hiddens
  config.model["post_fcnet_activation"] = "relu"
  config.model["use_lstm"] = use_lstm
  config.model["lstm_use_prev_action"] = True
  config.model["lstm_use_prev_reward"] = False
  config.model["lstm_cell_size"] = lstm_cell_size
  

  return config


def train(config, alg, local_mode, use_wandb, num_cpus, num_iterations=1):
  """Trains a model.

  Args:
    config: model config
    num_iterations: number of iterations ot train for.

  Returns:
    Training results.
  """
  tune.register_env("meltingpot", utils.env_creator)
  ray.init(num_cpus=num_cpus, num_gpus=config.num_gpus,resources={"accelerator_type:A100":1}, local_mode=local_mode)
  stop = {
      "training_iteration": num_iterations,
  }
  alg = alg
  if use_wandb:
    return tune.Tuner(
      alg,
      param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1,
                               callbacks=[WandbLoggerCallback(project="MeltingPot-Benchmarking")],
        )).fit()
  else:
    return tune.Tuner(
      alg,
      param_space=config.to_dict(),
        run_config=air.RunConfig(stop=stop, verbose=1),
        ).fit()


def main(args):
  set_seed(args.seed)
  config = get_config(
      substrate_name=args.env_name,
      num_rollout_workers=args.num_workers,
      alg=args.alg,
      use_lstm=args.use_lstm,
  )
  results = train(config, alg=args.alg, local_mode=args.local_mode, use_wandb=args.use_wandb, num_cpus=args.num_cpus, num_iterations=args.total_iterations)
  print(results)
  assert results.num_errors == 0


if __name__ == "__main__":
  main(args=parse_args())
