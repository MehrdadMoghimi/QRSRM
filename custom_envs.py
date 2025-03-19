import gymnasium as gym
from gymnasium.spaces import Discrete, Box, flatten, unflatten, flatten_space, MultiDiscrete
import numpy as np


class TradingEnv(gym.Env):
    def __init__(self, params=None):
        super(TradingEnv, self).__init__()

        # default parameters for the model
        default_params = {
            "kappa": 2,  # kappa of the OU process
            "sigma": 1,  # 0.2,  # standard deviation of the OU process
            "theta": 1,  # mean-reversion level of the OU process
            "phi": 0.005,  # transaction costs
            "psi": 0.5,  # terminal penalty on the inventory
            "T": 1,  # trading horizon
            "Ndt": 10,  # number of periods
            "Nda": 21,  # number of actions
            "max_q": 5,  # maximum value for the inventory
            "max_u": 2,  # maximum value for the trades
            "random_reset": False,  # reset the inventory to a random value between -max_q and max_q if True, otherwise reset to 0
        }

        self.params = default_params if params is None else {**default_params, **params}

        # Action space: 21 actions will be mapped to (-params["max_u"], params["max_u"])
        self.action_space = Discrete(self.params["Nda"])
        self._action_to_value = np.linspace(-self.params["max_u"], self.params["max_u"], self.params["Nda"])

        # Observation space: state representing the stock price, the agent's inventory and the current time step
        self.observation_space = Box(
            low=np.array([self.params["theta"] - 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]), -self.params["max_q"], 0]),
            high=np.array(
                [self.params["theta"] + 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]), self.params["max_q"], self.params["Ndt"]]
            ),
            shape=(3,),
            dtype=np.float32,
        )

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        action_value = self._action_to_value[action]

        # reward is calculated with the current stock price and the current action
        # Only is time = T-1, the reward also includes the terminal penalty on the inventory
        reward = -self._stock_price * action_value - self.params["phi"] * np.power(action_value, 2)

        # price of the stock at next time step - OU process
        dt = self.params["T"] / self.params["Ndt"]
        eta = self.params["sigma"] * np.sqrt((1 - np.exp(-2 * self.params["kappa"] * dt)) / (2 * self.params["kappa"]))
        self._stock_price = (
            self.params["theta"] + (self._stock_price - self.params["theta"]) * np.exp(-self.params["kappa"] * dt) + eta * np.random.normal()
        )

        self._time_step += 1

        # inventory at next time step - add the trade to current inventory
        self._agent_inventory += action_value

        # Check if the next state is the last state
        if self._time_step == self.params["Ndt"]:
            # reward - profit with terminal penalty calculated with the new price of the stock and the new inventory
            reward += self._agent_inventory * self._stock_price - self.params["psi"] * np.power(self._agent_inventory, 2)
            terminated = True
        else:
            terminated = False

        observation = self._get_obs()
        info = self._get_info()

        # Return the expected five values: observation, reward, done, truncated, info
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # the agent's inventory is initialized to a random value between -max_q and max_q if random_reset is True
        if self.params["random_reset"]:
            self._agent_inventory = np.random.uniform(-self.params["max_q"], self.params["max_q"])
            # the stock price is initialized to a random value
            self._stock_price = np.random.normal(self.params["theta"], 4 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]))
            self._stock_price = np.min([np.max([self._stock_price, self.observation_space.low[0]]), self.observation_space.high[0]])
        else:
            self._stock_price = self.params["theta"]
            self._agent_inventory = 0

        # the current time step is set to 0
        self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        return np.array([self._stock_price, self._agent_inventory, self._time_step], dtype=np.float32)

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class AmericanOptionEnv(gym.Env):
    def __init__(self, params=None):
        super(AmericanOptionEnv, self).__init__()

        # default parameters for the model
        default_params = {
            "option": "put",  # call or put
            "price_process": "GBM",  # GBM or OU
            "mu": -0.3,  # drift of the GBM process
            "kappa": 2,  # kappa of the OU process
            "sigma": 0.3,  # standard deviation of the OU process or volatility of the GBM process
            "theta": 1,  # mean-reversion level of the OU process or initial price of the GBM process
            "T": 1,  # trading horizon
            "K": 1,  # strike price of the option
            "Ndt": 10,  # number of periods
            "Nda": 2,  # number of actions
            "random_reset": False,  # reset
        }

        self.params = default_params if params is None else {**default_params, **params}

        # Action space: 2 actions corresponding to hold and execute
        self.action_space = Discrete(self.params["Nda"])

        # Observation space: state representing the stock price, the agent's inventory and the current time step
        if self.params["price_process"] == "GBM":
            self.observation_space = Box(
                low=np.array([self.params["theta"] - 5 * self.params["sigma"], 0]),
                high=np.array([self.params["theta"] + 5 * self.params["sigma"], self.params["Ndt"]]),
                shape=(2,),
                dtype=np.float32,
            )
        elif self.params["price_process"] == "OU":
            self.observation_space = Box(
                low=np.array([self.params["theta"] - 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]), 0]),
                high=np.array([self.params["theta"] + 6 * self.params["sigma"] / np.sqrt(2 * self.params["kappa"]), self.params["Ndt"]]),
                shape=(2,),
                dtype=np.float32,
            )
        else:
            raise ValueError("Invalid price process")

    def step(self, action):
        assert self.action_space.contains(action), "Invalid action"

        self._time_step += 1

        # Check if the next state is the last state or if the agent executes the option
        if self._time_step == self.params["Ndt"] or action == 1:
            # reward
            if self.params["option"] == "call":
                reward = np.max([0, self._stock_price - self.params["K"]])
            elif self.params["option"] == "put":
                reward = np.max([0, self.params["K"] - self._stock_price])
            else:
                raise ValueError("Invalid option type")
            terminated = True
        else:
            reward = 0
            terminated = False

        dt = self.params["T"] / self.params["Ndt"]

        if self.params["price_process"] == "GBM":
            # price of the stock at next time step - GBM process
            self._stock_price *= np.exp(
                (self.params["mu"] - 0.5 * self.params["sigma"] ** 2) * dt + self.params["sigma"] * np.sqrt(dt) * np.random.normal()
            )
        elif self.params["price_process"] == "OU":
            # price of the stock at next time step - OU process
            eta = self.params["sigma"] * np.sqrt((1 - np.exp(-2 * self.params["kappa"] * dt)) / (2 * self.params["kappa"]))
            self._stock_price = (
                self.params["theta"] + (self._stock_price - self.params["theta"]) * np.exp(-self.params["kappa"] * dt) + eta * np.random.normal()
            )
        else:
            raise ValueError("Invalid price process")

        observation = self._get_obs()
        info = self._get_info()

        # Return the expected five values: observation, reward, done, truncated, info
        return observation, reward, terminated, False, info

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # the agent's inventory is initialized to a random value between -max_q and max_q if random_reset is True
        if self.params["random_reset"]:
            # the stock price is initialized to a random value
            self._stock_price = np.random.normal(self.params["theta"], self.params["sigma"])
            self._stock_price = np.min([np.max([self._stock_price, self.observation_space.low[0]]), self.observation_space.high[0]])
            self._time_step = np.random.randint(0, self.params["Ndt"] - 1)
        else:
            self._stock_price = self.params["theta"]
            # the current time step is set to 0
            self._time_step = 0

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def _get_obs(self):
        return np.array([self._stock_price, self._time_step], dtype=np.float32)

    def _get_info(self):
        return {}

    def render(self, mode="human"):
        pass

    def close(self):
        pass


class CliffWalkingEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_height=4, grid_width=8, stochasticity=0.5, max_steps=50):
        super(CliffWalkingEnv, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.stochasticity = stochasticity
        self.max_steps = max_steps
        self.unflatten_observation_space = MultiDiscrete([grid_width, grid_height])
        self.observation_space = flatten_space(self.unflatten_observation_space)
        self.observation_space.dtype = np.float32
        # Box(low=np.array([0, 0]), high=np.array([self.grid_width, self.grid_height]), shape=(2,), dtype=np.float32)
        self.action_space = Discrete(4)  # up, right, down, left

        self.start_state = np.array([0, 0], dtype=np.int32)
        self.goal_state = np.array([self.grid_width - 1, 0], dtype=np.int32)
        self.state = self.start_state
        self.time = 0

    def step(self, action):
        x, y = self.state[0], self.state[1]
        if np.random.uniform() < self.stochasticity:
            action = np.random.randint(0, 4)
        if action == 0:  # up
            y = min(self.grid_height - 1, y + 1)
        elif action == 1:  # right
            x = min(self.grid_width - 1, x + 1)
        elif action == 2:  # down
            y = max(0, y - 1)
        elif action == 3:  # left
            x = max(0, x - 1)

        self.state = np.array([x, y], dtype=np.int32)
        self.time += 1

        if np.all(self.state == self.goal_state):
            reward = 10
            done = True
        elif self.time >= self.max_steps:
            reward = 0
            done = True
        elif y == 0 and x > 0:
            reward = -1  # cliff
            done = False
        else:
            reward = 0
            done = False

        return self._get_obs(), reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        self.time = 0
        return self._get_obs(), {}

    def _get_obs(self):
        return flatten(self.unflatten_observation_space, self.state).astype(np.float32)

    def render(self, mode="human"):
        pass


class CliffWalkingOriginalEnv(gym.Env):
    metadata = {"render_modes": ["human"]}

    def __init__(self, grid_height=4, grid_width=8):
        super(CliffWalkingOriginalEnv, self).__init__()
        self.grid_height = grid_height
        self.grid_width = grid_width
        self.observation_space = Box(
            low=np.array([0, 0]),
            high=np.array([self.grid_width, self.grid_height]),
            shape=(2,),
            dtype=np.float32,
        )
        self.action_space = Discrete(4)  # up, right, down, left

        self.start_state = np.array([0, 0], dtype=np.float32)
        self.goal_state = np.array([self.grid_width - 1, 0], dtype=np.float32)
        self.state = self.start_state
        self.time = 0

    def step(self, action):
        x, y = self.state[0], self.state[1]
        if action == 0:  # up
            y = min(self.grid_height - 1, y + 1)
        elif action == 1:  # right
            x = min(self.grid_width - 1, x + 1)
        elif action == 2:  # down
            y = max(0, y - 1)
        elif action == 3:  # left
            x = max(0, x - 1)

        self.state = np.array([x, y])
        self.time += 1

        if np.all(self.state == self.goal_state):
            reward = 0
            done = True
        elif y == 0 and x > 0:
            reward = -100  # cliff
            done = True
        else:
            reward = -1
            done = False

        return self.state, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.state = self.start_state
        self.time = 0
        return self.state, {}

    def render(self, mode="human"):
        pass


class SCAwareObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, gamma: float = 1.0, normalizer: float = 1.0):
        """Initialize :class:`SCAwareObservation` that requires an environment with a flat :class:`Box` observation space.

        Args:
            env: The environment to apply the wrapper
            gamma: The discount factor
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        # print(env.observation_space, env.observation_space.dtype)
        assert isinstance(env.observation_space, Box)
        # assert env.observation_space.dtype == np.float32
        self.env = env
        self.gamma = gamma
        self.normalizer = normalizer
        low = np.append(self.observation_space.low, [-np.inf, 0.0])
        high = np.append(self.observation_space.high, [np.inf, 1])
        self.observation_space = Box(low, high, dtype=np.float64)

    def observation(self, observation):
        """Adds to the observation with the current s and c values.

        Args:
            observation: The observation to add the s and c values to

        Returns:
            The observation with the s and c values appended
        """
        return np.append(observation, [self.s, self.c])

    def reset(self, **kwargs):
        """Reset the environment setting the s to zero and c to 1.

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        obs, info = super().reset(**kwargs)
        self.s = 0
        self.c = 1

        return self.observation(obs), info

    def step(self, action):
        """Steps through the environment, incrementing the s and c values.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        obs, reward, done, truncation, info = self.env.step(action)
        self.s += self.c * reward / self.normalizer
        self.c *= self.gamma
        return self.observation(obs), reward, done, truncation, info


class BAwareObservation(gym.Wrapper, gym.utils.RecordConstructorArgs):
    def __init__(self, env: gym.Env, gamma: float = 1.0, b_0=None, normalizer: float = 1.0):
        """Initialize :class:`BAwareObservation` that requires an environment with a flat :class:`Box` observation space.

        Args:
            env: The environment to apply the wrapper
            gamma: The discount factor
        """
        gym.utils.RecordConstructorArgs.__init__(self)
        gym.Wrapper.__init__(self, env)
        assert isinstance(env.observation_space, Box)
        assert env.observation_space.dtype == np.float32
        assert b_0 is not None
        self.env = env
        self.gamma = gamma
        self.normalizer = normalizer
        self.b_0 = b_0
        low = np.append(self.observation_space.low, [-np.inf])
        high = np.append(self.observation_space.high, [np.inf])
        self.observation_space = Box(low, high, dtype=np.float64)

    def observation(self, observation):
        """Adds to the observation with the current b value.

        Args:
            observation: The observation to add the b value to

        Returns:
            The observation with the b value appended
        """
        return np.append(observation, [self.b])

    def reset(self, **kwargs):
        """Reset the environment setting the b to kwargs['b']

        Args:
            **kwargs: Kwargs to apply to env.reset()

        Returns:
            The reset environment
        """
        obs, info = super().reset(**kwargs)
        self.b = self.b_0

        return self.observation(obs), info

    def step(self, action):
        """Steps through the environment, incrementing the b value.

        Args:
            action: The action to take

        Returns:
            The environment's step using the action.
        """
        obs, reward, done, truncation, info = self.env.step(action)
        self.b = (self.b - (reward / self.normalizer)) / self.gamma
        return self.observation(obs), reward, done, truncation, info


gym.register(
    id="TradingEnv-v0",
    entry_point="custom_envs:TradingEnv",
)
gym.register(
    id="AmericanOptionEnv-v0",
    entry_point="custom_envs:AmericanOptionEnv",
)
gym.register(
    id="CliffWalkingEnv-v0",
    entry_point="custom_envs:CliffWalkingEnv",
)
gym.register(
    id="CliffWalkingOriginalEnv-v0",
    entry_point="custom_envs:CliffWalkingOriginalEnv",
)
