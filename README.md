# Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning

This repository contains implementations of various quantile regression-based reinforcement learning algorithms for optimizing different risk measures.

## Paper

For a detailed explanation of the algorithms and theoretical foundations, please refer to our paper:
[Beyond CVaR: Leveraging Static Spectral Risk Measures for Enhanced Decision-Making in Distributional Reinforcement Learning](https://arxiv.org/abs/2501.02087)

## Algorithms

The repository implements the following algorithms:

- **QR-DQN**: Standard Quantile Regression DQN that optimizes the expected value
- **QR-SRM**: Quantile Regression for static Spectral Risk Measures optimization
- **QR-CVaR**: Quantile Regression for static Conditional Value-at-Risk optimization
- **QR-iCVaR**: Quantile Regression for iterated CVaR optimization

## Risk Measures

The QR-SRM implementation supports several risk measures:

- **CVaR (Conditional Value at Risk)**: Focuses on the worst outcomes below a specified quantile
- **Weighted Sum of CVaR**: Combines multiple CVaR values with different confidence levels
- **Exponential Risk Measure**: Applies exponential weighting to the quantiles

## Environments

The repository includes custom environments:

- **TradingEnv**: A statistical arbitrage environment with mean-reverting price dynamics
- **AmericanOptionEnv**: An environment for American option trading
- **CliffWalkingEnv**: A modified version of the classic Cliff Walking problem


## Usage

To train an agent with a specific algorithm and risk measure:

```bash
python qrsrm.py --env-id TradingEnv-v0 --risk-measure CVaR --alpha 0.5 --n-quantiles 50 --gamma 0.99 --save-model
```

For the QR-CVaR algorithm:

```bash
python qrcvar.py --env-id TradingEnv-v0 --alpha 0.5 --n-quantiles 50 --gamma 0.99 --save-model
```

## Requirements

- PyTorch
- Gymnasium
- Stable-Baselines3
- NumPy
- Pandas
- TensorBoard


## Project Structure

- `qrsrm.py`: Implementation of the QR-SRM algorithm
- `qrdqn.py`: Implementation of the QR-DQN algorithm
- `qrcvar.py`: Implementation of the QR-CVaR algorithm
- `qricvar.py`: Implementation of the QR-iCVaR algorithm
- `custom_envs.py`: Custom environments and wrapper classes
- `utils.py`: Utility functions for evaluation and visualization

## References

This code is based on the implementation of C51 algorithm in the amazing CleanRL library: 
[C51 Implementation](https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/c51.py)