import numpy as np
import pandas as pd
import torch
import os
import gymnasium as gym
import tqdm
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from qrdqn import QNetwork as QNetworkQRDQN
from qrsrm import QNetwork as QNetworkQRSRM
from qrcvar import QNetwork as QNetworkQRCVaR
from qricvar import QNetwork as QNetworkQRiCVaR

from qrdqn import make_env
from qrcvar import make_env_b_aware
from qrsrm import make_env_sc_aware, calculate_mu_function, CVaR, dual_power, weighted_sum_of_cvar, exponential_risk_measures

from custom_envs import SCAwareObservation


def run_tc_simulation_qrsrm(q_network, env, Nsimulations=10, seed=6, device=None, mu=None, theta_0=None, gamma=None):
    envParams = env.params
    action_to_value = np.linspace(-envParams["max_u"], envParams["max_u"], envParams["Nda"])
    Ndt = envParams["Ndt"]
    n_quantiles = q_network.n_quantiles

    prices_tcs = []
    quantiles_tcs = []
    rewards_tcs = []
    actions_tcs = []

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = env.reset()
    obs = torch.tensor([obs]).to(device)
    actions, quantiles = q_network.get_action(obs, mu=mu, theta_0=theta_0)
    for i in tqdm.tqdm(range(1, Nsimulations + 1), smoothing=0.1, dynamic_ncols=True):
        quantiles_tc = np.zeros((Ndt, n_quantiles))
        prices_tc = np.zeros((Ndt + 1, env.observation_space.shape[0]))
        actions_tc = np.zeros((Ndt + 1,))
        rewards_tc = np.zeros((Ndt + 1,))
        obs, _ = env.reset()
        prices_tc[0, :] = obs
        obs = torch.tensor([obs]).to(device)
        for timestep in range(envParams["Ndt"]):
            actions, quantiles = q_network.get_action(obs, mu=mu, theta_0=theta_0)
            quantiles_tc[timestep, :] = quantiles.detach().cpu().numpy().squeeze()
            actions = actions[0].cpu().numpy()
            # TRY NOT TO MODIFY: execute the game and log data.
            obs, reward, _, _, _ = env.step(actions)
            actions_tc[timestep] = action_to_value[actions.astype(int)]
            rewards_tc[timestep] = reward
            prices_tc[timestep + 1, :] = obs
            obs = torch.tensor([obs]).to(device)
        prices_tcs.append(prices_tc)
        quantiles_tcs.append(quantiles_tc)
        actions_tcs.append(actions_tc)
        rewards_tcs.append(rewards_tc)

    return prices_tcs, quantiles_tcs, actions_tcs, rewards_tcs


def run_tc_simulation_from_dir(path, Nsimulations=None, sim_seed=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = ""
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, model))
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        if agent_name == "qrsrm":
            risk_measure = "CVaR"  # args["risk_measure"]
            alpha = args["alpha"]
            envs = gym.vector.SyncVectorEnv(
                [make_env_sc_aware(args["env_id"], sim_seed, i, False, run_name, args["gamma"]) for i in range(args["num_envs"])]
            )
            env = SCAwareObservation(gym.make(args["env_id"]), gamma=args["gamma"])

            q_network = QNetworkQRSRM(envs, n_quantiles=args["n_quantiles"]).to(device)
            q_network.load_state_dict(data["model_weights"])

            taus = np.linspace(0.0, 1.0, args["n_quantiles"] + 1)
            # taus_middle = (taus[:-1] + taus[1:]) / 2  # (n_quantiles,)
            if risk_measure == "CVaR":
                phi_values = CVaR(taus, alpha=args["alpha"])
            elif risk_measure == "Dual":
                phi_values = dual_power(taus, alpha=args["alpha"])
            elif risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
                alphas = np.array(args["alphas"].split(","), dtype=np.float32)
                weights = np.array(args["weights"].split(","), dtype=np.float32)
                phi_values = weighted_sum_of_cvar(taus, alphas=alphas, weights=weights)
            elif risk_measure == "Exp":
                phi_values = exponential_risk_measures(taus, alpha=args["alpha"])
            else:
                raise ValueError("The risk measure is not defined.")
            mu = torch.tensor(calculate_mu_function(phi_values), device=device)
            theta_0_df = pd.read_pickle(path + model + "/theta_0_df.pkl")
            theta_0 = torch.tensor(theta_0_df.loc[len(theta_0_df) - 1].values, device=device)

            prices_tcs, quantiles_tcs, actions_tcs, rewards_tcs = run_tc_simulation_qrsrm(
                q_network,
                env,
                Nsimulations=Nsimulations,
                seed=sim_seed,
                device=device,
                gamma=args["gamma"],
                mu=mu,
                theta_0=theta_0,
            )
        else:
            raise ValueError("Unknown model name: {}".format(model))
    return prices_tcs, quantiles_tcs, actions_tcs, rewards_tcs


def run_simulation_qrsrm(q_network, envs, Nsimulations=10000, seed=6, device=None, mu=None, theta_0=None, gamma=None):
    rewards = np.zeros((Nsimulations,))

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = envs.reset()
    obs = torch.tensor(obs).to(device)
    actions, _ = q_network.get_action(obs, mu=mu, theta_0=theta_0)
    for i in tqdm.tqdm(range(Nsimulations), smoothing=0.1, dynamic_ncols=True):
        obs, _ = envs.reset()
        obs = torch.tensor(obs).to(device)
        terminated, truncated = False, False
        timestep = 0
        while not terminated and not truncated:
            actions, _ = q_network.get_action(obs, mu=mu, theta_0=theta_0)
            actions = actions.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = envs.step(actions)
            obs = next_obs
            obs = torch.tensor(obs).to(device)
            rewards[i] += reward[0] * gamma**timestep
            timestep += 1

    return rewards


def run_simulation_qricvar(q_network, envs, Nsimulations=10000, seed=6, device=None, alpha=None, gamma=None):
    rewards = np.zeros((Nsimulations,))

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = envs.reset()
    obs = torch.tensor(obs).to(device)
    actions, _ = q_network.get_action(obs, alpha=alpha)
    for i in tqdm.tqdm(range(Nsimulations), smoothing=0.1, dynamic_ncols=True):
        obs, _ = envs.reset()
        obs = torch.tensor(obs).to(device)
        terminated, truncated = False, False
        timestep = 0
        while not terminated and not truncated:
            actions, _ = q_network.get_action(obs, alpha=alpha)
            actions = actions.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = envs.step(actions)
            obs = next_obs
            obs = torch.tensor(obs).to(device)
            rewards[i] += reward[0] * gamma**timestep
            timestep += 1

    return rewards


def run_simulation(q_network, envs, Nsimulations=None, seed=None, device=None, gamma=None):
    rewards = np.zeros((Nsimulations,))

    torch.manual_seed(seed)
    np.random.seed(seed)

    obs, _ = envs.reset()
    obs = torch.tensor(obs).to(device)
    actions, _ = q_network.get_action(obs)
    for i in tqdm.tqdm(range(Nsimulations), smoothing=0.1, dynamic_ncols=True):
        obs, _ = envs.reset()
        obs = torch.tensor(obs).to(device)
        terminated, truncated = False, False
        timestep = 0
        while not terminated and not truncated:
            actions, _ = q_network.get_action(obs)
            actions = actions.cpu().numpy()
            next_obs, reward, terminated, truncated, _ = envs.step(actions)
            obs = next_obs
            obs = torch.tensor(obs).to(device)
            rewards[i] += reward[0] * gamma**timestep
            timestep += 1

    return rewards


def run_simulation_from_dir(path, Nsimulations=None, eval_sim_seed=None):
    device = "cpu"  # torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = ""
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        args["num_envs"] = 1
        agent_seed = args["seed"]
        sim_seed = agent_seed if eval_sim_seed is None else eval_sim_seed
        print("Processing model {} --> {} with seed {}".format(ii + 1, model, sim_seed))
        if agent_name == "qrdqn":
            risk_measure = "Mean"
            alpha = 1.0
            weight = 1.0
            envs = gym.vector.SyncVectorEnv([make_env(args["env_id"], sim_seed, i, False, run_name) for i in range(args["num_envs"])])
            q_network = QNetworkQRDQN(envs, n_quantiles=args["n_quantiles"]).to(device)
            q_network.load_state_dict(data["model_weights"])

            rewards = run_simulation(q_network, envs, Nsimulations=Nsimulations, seed=sim_seed, device=device, gamma=args["gamma"])
        elif agent_name == "qrsrm":
            risk_measure = args["risk_measure"] if "risk_measure" in args.keys() else model.split("__")[6]
            alpha = args["alpha"]
            weight = 1.0
            envs = gym.vector.SyncVectorEnv(
                [
                    make_env_sc_aware(args["env_id"], sim_seed, i, False, run_name, args["gamma"], args["reward_normalizer"])
                    for i in range(args["num_envs"])
                ]
            )
            q_network = QNetworkQRSRM(
                envs, n_quantiles=args["n_quantiles"], extended_value=args["extended_value"], reward_normalizer=args["reward_normalizer"]
            ).to(device)
            q_network.load_state_dict(data["model_weights"])

            taus = np.linspace(0.0, 1.0, args["n_quantiles"] + 1)
            # taus_middle = (taus[:-1] + taus[1:]) / 2  # (n_quantiles,)
            if risk_measure == "CVaR":
                phi_values = CVaR(taus, alpha=args["alpha"])
            elif risk_measure == "Dual":
                phi_values = dual_power(taus, alpha=args["alpha"])
            elif risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
                weight = args["weights"]  # for naming purposes
                alphas = np.array(args["alphas"].split(","), dtype=np.float32)
                weights = np.array(args["weights"].split(","), dtype=np.float32)
                phi_values = weighted_sum_of_cvar(taus, alphas=alphas, weights=weights)
            elif risk_measure == "Exp":
                phi_values = exponential_risk_measures(taus, alpha=args["alpha"])
            else:
                raise ValueError("The risk measure is not defined.")
            mu = torch.tensor(calculate_mu_function(phi_values), device=device)
            theta_0_df = pd.read_pickle(path + model + "/theta_0_df.pkl")
            theta_0 = torch.tensor(theta_0_df.loc[len(theta_0_df) - 1].values, device=device)

            rewards = run_simulation_qrsrm(
                q_network,
                envs,
                Nsimulations=Nsimulations,
                seed=sim_seed,
                device=device,
                gamma=args["gamma"],
                mu=mu,
                theta_0=theta_0,
            )
        elif agent_name == "qricvar":
            risk_measure = "CVaR"
            alpha = args["alpha"]
            weight = 1.0
            envs = gym.vector.SyncVectorEnv([make_env(args["env_id"], sim_seed, i, False, run_name) for i in range(args["num_envs"])])
            q_network = QNetworkQRiCVaR(envs, n_quantiles=args["n_quantiles"]).to(device)
            q_network.load_state_dict(data["model_weights"])

            rewards = run_simulation_qricvar(
                q_network, envs, Nsimulations=Nsimulations, seed=sim_seed, device=device, gamma=args["gamma"], alpha=alpha
            )
        elif agent_name == "qrcvar" or agent_name == "qrcvar_soft":
            risk_measure = "CVaR"
            alpha = args["alpha"]
            weight = 1.0

            df = pd.read_pickle(path + model + "/b_0_df.pkl")
            b_0 = df.loc[len(df) - 1]

            envs = gym.vector.SyncVectorEnv(
                [
                    make_env_b_aware(args["env_id"], sim_seed, i, False, run_name, gamma=args["gamma"], b_0=b_0, normalizer=args["reward_normalizer"])
                    for i in range(args["num_envs"])
                ]
            )

            q_network = QNetworkQRCVaR(
                envs, n_quantiles=args["n_quantiles"], extended_value=args["extended_value"], reward_normalizer=args["reward_normalizer"]
            ).to(device)
            q_network.load_state_dict(data["model_weights"])

            rewards = run_simulation(q_network, envs, Nsimulations=Nsimulations, seed=sim_seed, device=device, gamma=args["gamma"])
        else:
            raise ValueError("Unknown model name: {}".format(model))
        df = pd.DataFrame(rewards, columns=["rewards"])
        df = df.assign(
            environment_name=args["env_id"],
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            weight=str(weight),
            n_quantile=args["n_quantiles"],
            sim_seed=sim_seed,
            agent_seed=args["seed"],
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def load_data_from_dir(path):
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, model))
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        if agent_name == "qrdqn":
            risk_measure = "Mean"
            alpha = 1.0
        elif agent_name == "qrsrm":
            risk_measure = args["risk_measure"]
            if risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
            else:
                alpha = args["alpha"]
        elif agent_name == "qricvar" or agent_name == "qrcvar":
            risk_measure = "CVaR"
            alpha = args["alpha"]
        else:
            raise ValueError("Unknown model name: {}".format(model))
        scalar_accumulator = EventAccumulator(str(path + model)).Reload().scalars
        keys = scalar_accumulator.Keys()
        assert "charts/episodic_discounted_return" in keys
        idx = keys.index("charts/episodic_discounted_return")
        df = pd.DataFrame(scalar_accumulator.Items(keys[idx]))
        df["wall_time"] = df["wall_time"] - df["wall_time"][0]
        df = df.assign(
            environment_name=args["env_id"],
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            n_quantile=args["n_quantiles"],
            agent_seed=args["seed"],
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def load_theta_diff_from_dir(path):
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, model))
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        if agent_name == "qrsrm":
            risk_measure = args["risk_measure"] if "risk_measure" in args.keys() else model.split("__")[6]
            if risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
            else:
                alpha = args["alpha"]
        else:
            print("Unknown model name: {}".format(model))
            continue
        scalar_accumulator = EventAccumulator(str(path + model)).Reload().scalars
        keys = scalar_accumulator.Keys()
        assert "losses/theta_0_diff" in keys
        idx = keys.index("losses/theta_0_diff")
        df = pd.DataFrame(scalar_accumulator.Items(keys[idx]))
        df["wall_time"] = df["wall_time"] - df["wall_time"][0]
        df = df.assign(
            environment_name=args["env_id"],
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            n_quantile=args["n_quantiles"],
            agent_seed=args["seed"],
        )
        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)
    return df_exp


def plot_policy(path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = ""
    sim_seed = 1
    hist2dim_pi_list = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, model))
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        if agent_name == "qrsrm":
            risk_measure = args["risk_measure"] if "risk_measure" in args.keys() else model.split("__")[6]
            alpha = args["alpha"]
            envs = gym.vector.SyncVectorEnv(
                [make_env_sc_aware(args["env_id"], sim_seed, i, False, run_name, args["gamma"]) for i in range(args["num_envs"])]
            )
            q_network = QNetworkQRSRM(envs, n_quantiles=args["n_quantiles"]).to(device)
            q_network.load_state_dict(data["model_weights"])

            taus = np.linspace(0.0, 1.0, args["n_quantiles"] + 1)
            # taus_middle = (taus[:-1] + taus[1:]) / 2  # (n_quantiles,)
            if risk_measure == "CVaR":
                phi_values = CVaR(taus, alpha=args["alpha"])
            elif risk_measure == "Dual":
                phi_values = dual_power(taus, alpha=args["alpha"])
            elif risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
                alphas = np.array(args["alphas"].split(","), dtype=np.float32)
                weights = np.array(args["weights"].split(","), dtype=np.float32)
                phi_values = weighted_sum_of_cvar(taus, alphas=alphas, weights=weights)
            elif risk_measure == "Exp":
                phi_values = exponential_risk_measures(taus, alpha=args["alpha"])
            else:
                raise ValueError("The risk measure is not defined.")
            mu = torch.tensor(calculate_mu_function(phi_values), device=device)
            theta_0_df = pd.read_pickle(path + model + "/theta_0_df.pkl")
            theta_0 = torch.tensor(theta_0_df.loc[len(theta_0_df) - 1].values, device=device)
        else:
            print("Unknown model name: {}".format(model))
            continue

        envParams = envs.get_attr("params")[0]
        s_space = np.linspace(envParams["theta"] - 0.5, envParams["theta"] + 0.5, 100)
        t_space = np.linspace(0, envParams["Ndt"] - 1, 100)
        # initialize 2D histogram
        hist2dim_pi = np.zeros([len(s_space), len(t_space)])

        for s_idx, s_val in enumerate(s_space):
            for t_idx, t_val in enumerate(t_space):
                # best action according to the policy
                action_temp, _ = q_network.get_action(
                    torch.tensor(np.array([[s_val, t_val, 0, args["gamma"] ** t_val]], dtype=np.float32)).to(device), mu=mu, theta_0=theta_0
                )
                action_temp = action_temp.detach().cpu().numpy().squeeze()
                hist2dim_pi[len(s_space) - s_idx - 1, t_idx] = np.int32(action_temp)
        hist2dim_pi_list.append(hist2dim_pi)

    return hist2dim_pi_list, s_space, t_space


def option_execution_policy(path, t_res=500, s_res=351):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = ""
    sim_seed = 1
    df_exps = []
    directories = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    for ii, model in enumerate(directories):
        print("Processing model {} --> {}".format(ii + 1, model))
        agent_name = model.split("__")[1]
        data = torch.load(path + model + f"/{agent_name}.cleanrl_model")
        args = data["args"]
        if agent_name == "qrsrm":
            risk_measure = args["risk_measure"] if "risk_measure" in args.keys() else model.split("__")[6]
            alpha = args["alpha"]
            envs = gym.vector.SyncVectorEnv(
                [make_env_sc_aware(args["env_id"], sim_seed, i, False, run_name, args["gamma"]) for i in range(args["num_envs"])]
            )
            q_network = QNetworkQRSRM(envs, n_quantiles=args["n_quantiles"]).to(device)
            q_network.load_state_dict(data["model_weights"])

            taus = np.linspace(0.0, 1.0, args["n_quantiles"] + 1)
            # taus_middle = (taus[:-1] + taus[1:]) / 2  # (n_quantiles,)
            if risk_measure == "CVaR":
                phi_values = CVaR(taus, alpha=args["alpha"])
            elif risk_measure == "Dual":
                phi_values = dual_power(taus, alpha=args["alpha"])
            elif risk_measure == "SRM" or risk_measure == "WSCVaR":
                alpha = args["alphas"]  # for naming purposes
                alphas = np.array(args["alphas"].split(","), dtype=np.float32)
                weights = np.array(args["weights"].split(","), dtype=np.float32)
                phi_values = weighted_sum_of_cvar(taus, alphas=alphas, weights=weights)
            elif risk_measure == "Exp":
                phi_values = exponential_risk_measures(taus, alpha=args["alpha"])
            else:
                raise ValueError("The risk measure is not defined.")
            mu = torch.tensor(calculate_mu_function(phi_values), device=device)
            theta_0_df = pd.read_pickle(path + model + "/theta_0_df.pkl")
            theta_0 = torch.tensor(theta_0_df.loc[len(theta_0_df) - 1].values, device=device)
        else:
            print("Unknown model name: {}".format(model))
            continue

        envParams = envs.get_attr("params")[0]
        s_space = np.linspace(envParams["theta"], envParams["theta"] - 0.4, s_res)
        t_space = np.linspace(0, envParams["Ndt"], t_res)
        # initialize 2D histogram
        execution_s = np.zeros((len(t_space),))

        for t_idx, t_val in enumerate(t_space):
            for s_idx, s_val in enumerate(s_space):
                # best action according to the policy
                action_temp, _ = q_network.get_action(
                    torch.tensor(np.array([[s_val, t_val, 0, args["gamma"] ** t_val]], dtype=np.float32)).to(device), mu=mu, theta_0=theta_0
                )
                action_temp = action_temp.detach().cpu().numpy().squeeze()
                if action_temp == 1:
                    execution_s[t_idx] = s_val
                    break
        df = pd.DataFrame(execution_s, columns=["execution_s"], index=t_space).reset_index()
        df = df.assign(
            environment_name=args["env_id"],
            agent=agent_name,
            risk_measure=risk_measure,
            alpha=str(alpha),
            n_quantile=args["n_quantiles"],
            agent_seed=args["seed"],
        )

        df_exps.append(df)
    df_exp = pd.concat(df_exps, sort=True).reset_index(drop=True)

    return df_exp


GAME_NAMES = [
    ("TradingEnv-v0", "Statistical Arbitrage"),
    ("AmericanOptionEnv-v0", "American Option Trading"),
    ("CliffWalkingEnv-v0", "Cliff Walking"),
    ("LunarLander-v2", "Lunar Lander"),
]
GAME_NAME_MAP = dict(GAME_NAMES)

AGENT_NAMES = [
    ("qrdqn", "QR-DQN"),
    ("qrsrm", "QR-SRM"),
    ("qrcvar", "QR-CVaR"),
    ("qrcvar_soft", "QR-CVaR Soft"),
    ("qricvar", "QR-iCVaR"),
]
AGENT_NAME_MAP = dict(AGENT_NAMES)


def environment_pretty(row):
    return GAME_NAME_MAP[row["environment_name"]]


def agent_pretty(row):
    if row["agent"] == "qrdqn":
        return f"{AGENT_NAME_MAP[row['agent']]}"
    elif row["agent"] == "qrsrm":
        if row["risk_measure"] == "CVaR":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "Dual":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\nu$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "SRM" or row["risk_measure"] == "WSCVaR":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
        elif row["risk_measure"] == "Exp":
            return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\lambda$=" + f"{row['alpha']})"
        else:
            raise ValueError("Unknown risk measure: {}".format(row["risk_measure"]))
    elif row["agent"] == "qrcvar" or row["agent"] == "qricvar" or row["agent"] == "qrcvar_soft":
        return f"{AGENT_NAME_MAP[row['agent']]}(" + r"$\alpha$=" + f"{row['alpha']})"
    else:
        raise ValueError("Unknown agent name: {}".format(row["agent"]))


def agent_pretty_quantile(row):
    return f"{AGENT_NAME_MAP[row['agent']]}(" + r"N=" + f"{row['n_quantile']})"


def add_columns(df):
    df["environment_pretty"] = df.apply(environment_pretty, axis=1)
    df["Model"] = df.apply(agent_pretty, axis=1)
    # add df["weight"]=1.0 if column does not exist
    if "weight" not in df.columns:
        df["weight"] = 1.0
    df["agent_id"] = (
        df["agent"] + "_" + df["alpha"].astype(str) + "_" + df["weight"].astype(str)
    )  # qrsrm with SRM as risk measure is identified by alphas not weights
    return df


def add_columns_quantile(df):
    df["environment_pretty"] = df.apply(environment_pretty, axis=1)
    df["Model"] = df.apply(agent_pretty_quantile, axis=1)
    df["agent_id"] = df["agent"] + "_" + df["n_quantile"].astype(str)  # qrsrm with SRM as risk measure is identified by alphas not weights
    return df


def make_agent_hue_kws(experiments):
    pairs = [(exp["agent_name"], exp["color"]) for exp in experiments]
    agent_names, colors = zip(*pairs)
    hue_kws = dict(color=colors)
    return list(agent_names), hue_kws


def moving_average(values, window_size):
    # numpy.convolve uses zero for initial missing values, so is not suitable.
    numerator = np.nancumsum(values)
    # The sum of the last window_size values.
    numerator[window_size:] = numerator[window_size:] - numerator[:-window_size]
    # numerator[:window_size] = np.nan
    denominator = np.ones(len(values)) * window_size
    denominator[:window_size] = np.arange(1, window_size + 1)
    smoothed = numerator / denominator
    assert values.shape == smoothed.shape
    return smoothed


def smooth(df, smoothing_window, index_columns, columns):
    dfg = df.groupby(index_columns)
    for col in columns:
        df[col] = dfg[col].transform(lambda s: moving_average(s.values, smoothing_window))
    return df


def smooth_dataframe(df):
    return smooth(
        df,
        smoothing_window=10,
        index_columns=["agent", "risk_measure", "alpha", "n_quantile", "environment_name", "agent_seed"],
        columns=[
            "value",
        ],
    )


def mean_value(group):
    # Perform some operation on the group
    result = group.mean()  # Replace this with your actual operation
    return result


def cvar1(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values < np.quantile(r_values, 1.0)])
    return result


def cvar2(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values < np.quantile(r_values, 0.5)])
    return result


def cvar3(group):
    # Perform some operation on the group
    r_values = group.values
    result = np.mean(r_values[r_values < np.quantile(r_values, 0.2)])
    return result


def expo4(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values1 = exponential_risk_measures(taus_middle, alpha=4.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values1) / nq
    return result


def expo12(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values1 = exponential_risk_measures(taus_middle, alpha=12.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values1) / nq
    return result


def dual2(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = dual_power(taus_middle, alpha=2.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def dual3(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = dual_power(taus_middle, alpha=3.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def dual4(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = dual_power(taus_middle, alpha=4.0)
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.0999, 0.5999, 0.9999], weights=[0.2, 0.3, 0.5])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm2(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.1999, 0.9999], weights=[0.8, 0.2])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm3(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.5, 0.5])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


def srm4(group):
    # Perform some operation on the group
    r_values = group.values
    nq = 10001
    taus = np.linspace(0.0, 1.0, nq)
    taus_middle = (taus[:-1] + taus[1:]) / 2
    phi_values2 = weighted_sum_of_cvar(taus_middle, alphas=[0.2, 1.0], weights=[0.99, 0.01])
    quantiles = np.quantile(r_values, taus_middle)
    result = np.matmul(quantiles, phi_values2) / nq
    return result


AGG_RISK_VALUES = [
    (r"$\operatorname{CVaR}_{1.0}$", cvar1),
    (r"$\operatorname{CVaR}_{0.5}$", cvar2),
    (r"$\operatorname{CVaR}_{0.2}$", cvar3),
    (r"$\operatorname{ERM}_{4.0}$", expo4),
    (r"$\operatorname{ERM}_{12.0}$", expo12),
    (r"$\operatorname{DPRM}_{2.0}$", dual2),
    (r"$\operatorname{DPRM}_{3.0}$", dual3),
    (r"$\operatorname{DPRM}_{4.0}$", dual4),
    (r"$\operatorname{WSCVaR}_{0.1,0.6,1.0}$", srm),
    # (r"$\operatorname{WSCVaR}_{0.2,1.0}_{0.8,0.2}$", srm2),
    (r"$\operatorname{WSCVaR}_{0.2,1.0}_{0.5,0.5}$", srm3),
    (r"$\operatorname{WSCVaR}_{0.2,1.0}_{0.99,0.01}$", srm4),
]
