# This code is based on the implementation of the C51 algorithm in the CleanRL library.
# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/c51/#c51py
import argparse
import os
import random
import time
from distutils.util import strtobool

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter
import torch.nn.init as init
from torch.optim.lr_scheduler import ExponentialLR
import custom_envs  # register custom envs
import pandas as pd
from custom_envs import BAwareObservation

import warnings
import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)


def parse_args():
    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-name", type=str, default=os.path.basename(__file__).rstrip(".py"),
        help="the name of this experiment")
    parser.add_argument("--seed", type=int, default=1,
        help="seed of the experiment")
    parser.add_argument("--torch-deterministic", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, `torch.backends.cudnn.deterministic=False`")
    parser.add_argument("--cuda", type=lambda x: bool(strtobool(x)), default=True, nargs="?", const=True,
        help="if toggled, cuda will be enabled by default")
    parser.add_argument("--track", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="if toggled, this experiment will be tracked with Weights and Biases")
    parser.add_argument("--wandb-project-name", type=str, default="cleanRL",
        help="the wandb's project name")
    parser.add_argument("--wandb-entity", type=str, default=None,
        help="the entity (team) of wandb's project")
    parser.add_argument("--capture-video", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to capture videos of the agent performances (check out `videos` folder)")
    parser.add_argument("--save-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to save model into the `{args.dir}/{run_name}` folder")
    parser.add_argument("--upload-model", type=lambda x: bool(strtobool(x)), default=False, nargs="?", const=True,
        help="whether to upload the saved model to huggingface")
    parser.add_argument("--hf-entity", type=str, default="",
        help="the user or org name of the model repository from the Hugging Face Hub")
    parser.add_argument("--dir", type=str, default="runs",
        help="the directory to store the model and log")

    # Algorithm specific arguments
    parser.add_argument("--env-id", type=str, default="CartPole-v1",
        help="the id of the environment")
    parser.add_argument("--total-timesteps", type=int, default=500000,
        help="total timesteps of the experiments")
    parser.add_argument("--learning-rate", type=float, default=3e-4,
        help="the learning rate of the optimizer")
    parser.add_argument("--num-envs", type=int, default=1,
        help="the number of parallel game environments")
    parser.add_argument("--n-quantiles", type=int, default=50,
        help="the number of quantiles")
    parser.add_argument("--buffer-size", type=int, default=10000,
        help="the replay memory buffer size")
    parser.add_argument("--gamma", type=float, default=1.0,
        help="the discount factor gamma")
    parser.add_argument("--target-network-frequency", type=int, default=500,
        help="the timesteps it takes to update the target network")
    parser.add_argument("--batch-size", type=int, default=256,
        help="the batch size of sample from the reply memory")
    parser.add_argument("--start-e", type=float, default=1,
        help="the starting epsilon for exploration")
    parser.add_argument("--end-e", type=float, default=0.01,
        help="the ending epsilon for exploration")
    parser.add_argument("--exploration-fraction", type=float, default=0.5,
        help="the fraction of `total-timesteps` it takes from start-e to go end-e")
    parser.add_argument("--learning-starts", type=int, default=10000,
        help="timestep to start learning")
    parser.add_argument("--train-frequency", type=int, default=10,
        help="the frequency of training")
    parser.add_argument("--scheduler-frequency", type=int, default=500,
        help="the timesteps it takes to reduce the learning rate")
    parser.add_argument("--b-frequency", type=int, default=500,
        help="the timesteps it takes to update the b")
    parser.add_argument("--alpha", type=float, default=1.0,
        help="the alpha parameter of cvar")
    parser.add_argument("--reward-normalizer", type=float, default=1.0,
        help="the reward normalizing factor of the extended state space")
    parser.add_argument("--extended-value", type=lambda x: bool(strtobool(x)), default=True, action=argparse.BooleanOptionalAction,
        help="whether to use the extended state space for the value function")
    args = parser.parse_args()

    return args


def make_env_b_aware(env_id, seed, idx, capture_video, run_name, gamma, b_0, normalizer):
    def thunk():
        if capture_video and idx == 0:
            if env_id.startswith("LunarLander"):
                env = BAwareObservation(gym.make(env_id, render_mode="rgb_array", enable_wind=True), gamma=gamma, b_0=b_0, normalizer=normalizer)
            else:
                env = BAwareObservation(gym.make(env_id, render_mode="rgb_array"), gamma=gamma, b_0=b_0, normalizer=normalizer)
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            if env_id.startswith("LunarLander"):
                env = BAwareObservation(gym.make(env_id, enable_wind=True), gamma, b_0=b_0, normalizer=normalizer)
            else:
                env = BAwareObservation(gym.make(env_id), gamma, b_0=b_0, normalizer=normalizer)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env.action_space.seed(seed)

        return env

    return thunk


def init_weights(layer):
    input_size = 3
    if type(layer) == nn.Linear:
        init.normal_(layer.weight, mean=0.0, std=1 / np.sqrt(input_size) / 2)
        layer.bias.data.fill_(0.0)


# ALGO LOGIC: initialize agent here:
class QNetwork(nn.Module):
    def __init__(self, env, n_quantiles=50, extended_value=True, reward_normalizer=1.0):
        super().__init__()
        self.env = env
        self.n_quantiles = n_quantiles
        self.extended_value = extended_value
        self.normalizer = reward_normalizer
        self.register_buffer("tau", (2 * torch.arange(n_quantiles) + 1) / (2.0 * n_quantiles))
        self.n = int(env.single_action_space.n)
        self.hidden_layer_size = 64
        input_size = np.array(env.single_observation_space.shape).prod()
        if not extended_value:
            input_size -= 1
        self.network = nn.Sequential(
            nn.Linear(input_size, self.hidden_layer_size),
            nn.SiLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.SiLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.SiLU(),
            nn.Linear(self.hidden_layer_size, self.hidden_layer_size),
            nn.SiLU(),
            nn.Linear(self.hidden_layer_size, self.n * n_quantiles),
            nn.Unflatten(1, (self.n, n_quantiles)),
        )
        self.network.apply(init_weights)

    def forward(self, x):
        x = x.float()
        return self.network(x)

    def get_action(self, x, action=None):
        # quantile function for each action
        if self.extended_value:
            quantiles = self.forward(x)  # (batch_size, n_actions, n_quantiles)
        else:
            quantiles = self.forward(x[:, :-1])
        if action is None:
            # Calculate Q-values for each action
            b_batch = self.normalizer * x[:, -1].reshape(-1, 1, 1).repeat(1, self.n, self.n_quantiles)
            bm_quantiles = b_batch - quantiles
            bm_quantiles = torch.max(torch.tensor(0), bm_quantiles)
            q_values = bm_quantiles.mean(dim=2)  # (batch_size, n_actions)
            action = torch.argmin(q_values, 1)
        return action, quantiles[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


# huber loss function
def huber(x, k=1.0):
    return torch.where(x.abs() < k, 0.5 * x.pow(2), k * (x.abs() - 0.5 * k))


if __name__ == "__main__":
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:

poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )
    args = parse_args()
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    run_name += f"__{args.n_quantiles}__{args.alpha}"
    writer = SummaryWriter(f"{args.dir}/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # This should be modified for each environment for good performance
    b_min, b_max = -10.0, 10.0
    # b_min, b_max = -1.0, 1.0
    b_linspace = np.linspace(b_min, b_max, 4001)
    b_0_df = pd.Series(name="b_0", dtype=np.float32)
    b_0 = 0.0
    b_0_df.loc[len(b_0_df)] = b_0

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [
            make_env_b_aware(args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma, b_0=b_0, normalizer=args.reward_normalizer)
            for i in range(args.num_envs)
        ]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_quantiles=args.n_quantiles, extended_value=args.extended_value, reward_normalizer=args.reward_normalizer).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)  # , eps=0.01 / args.batch_size)
    scheduler = ExponentialLR(optimizer, gamma=0.999)
    target_network = QNetwork(envs, n_quantiles=args.n_quantiles, extended_value=args.extended_value, reward_normalizer=args.reward_normalizer).to(
        device
    )
    target_network.load_state_dict(q_network.state_dict())

    rb = ReplayBuffer(
        buffer_size=args.buffer_size,
        observation_space=envs.single_observation_space,
        action_space=envs.single_action_space,
        device=device,
        n_envs=args.num_envs,
        handle_timeout_termination=False,
    )
    start_time = time.time()

    initial_state, _ = envs.reset(seed=args.seed)
    initial_state = torch.Tensor(initial_state).to(device)
    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset(seed=args.seed)
    discounted_return, time_step = 0, 0
    for global_step in tqdm.tqdm(range(args.total_timesteps), desc="Training"):
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)
        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, quantiles = q_network.get_action(torch.Tensor(obs).to(device))
            actions = actions.cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminated, truncated, infos = envs.step(actions)
        discounted_return += rewards[0] * args.gamma**time_step
        time_step += 1

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                # Skip the envs that are not done
                if "episode" not in info:
                    continue
                # print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                writer.add_scalar("charts/epsilon", epsilon, global_step)
                writer.add_scalar("charts/episodic_discounted_return", discounted_return, global_step)
                discounted_return, time_step = 0, 0

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, d in enumerate(truncated):
            if d:
                real_next_obs[idx] = infos["final_observation"][idx]
        rb.add(obs, real_next_obs, actions, rewards, terminated, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if global_step % args.train_frequency == 0:
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    _, next_quantiles = target_network.get_action(data.next_observations)
                    target_quantiles = data.rewards + args.gamma * next_quantiles * (1 - data.dones)

                _, old_quantiles = q_network.get_action(data.observations, data.actions.flatten())

                # (batch_size, 1, n_quantiles) - (batch_size, n_quantiles, 1) = (batch_size, n_quantiles, n_quantiles)
                diff = target_quantiles.unsqueeze(-1).transpose(-1, -2) - old_quantiles.unsqueeze(-1)
                tau_reshaped = target_network.tau.view(1, -1, 1).repeat(args.batch_size, 1, target_network.n_quantiles)
                loss = (huber(diff) * (tau_reshaped - (diff.detach() < 0).float()).abs()).mean(2).sum(1)
                loss = loss.mean()

                if global_step % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = old_quantiles.mean(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                    _, quantiles_0 = q_network.get_action(initial_state)
                    writer.add_scalar("rewards/srm", quantiles_0.mean(1).cpu().detach().numpy()[0], global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update the target network
            if global_step % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

            # update the learning rate
            if global_step % args.scheduler_frequency == 0:
                scheduler.step()

            # update the b
            if global_step % args.b_frequency == 0:
                initial_state_temp, _ = envs.reset(seed=args.seed)
                initial_states = np.array([[*initial_state_temp[0][:-1], b] for b in b_linspace], dtype=np.float32)
                _, quantiles = q_network.get_action(torch.Tensor(initial_states).to(device))
                b_batch = torch.tensor(args.reward_normalizer * b_linspace).reshape(-1, 1).repeat(1, args.n_quantiles).to(device)
                bm_quantiles = b_batch - quantiles
                bm_quantiles = torch.max(torch.tensor(0), bm_quantiles)
                q_values = bm_quantiles.mean(dim=1)  # (batch_size, n_actions)
                b_values = args.reward_normalizer * b_linspace - q_values.cpu().detach().numpy() / args.alpha
                b_0 = b_linspace[np.argmax(b_values)]
                b_0_df.loc[len(b_0_df)] = b_0
                writer.add_scalar("rewards/b", args.reward_normalizer * b_0, global_step)
                envs = gym.vector.SyncVectorEnv(
                    [
                        make_env_b_aware(
                            args.env_id, args.seed + i, i, args.capture_video, run_name, args.gamma, b_0=b_0, normalizer=args.reward_normalizer
                        )
                        for i in range(args.num_envs)
                    ]
                )
                obs, _ = envs.reset(seed=args.seed)

    if args.save_model:
        model_path = f"{args.dir}/{run_name}/{args.exp_name}.cleanrl_model"
        model_data = {
            "model_weights": q_network.state_dict(),
            "args": vars(args),
        }
        b_0_df.to_pickle(f"{args.dir}/{run_name}/b_0_df.pkl")
        torch.save(model_data, model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
