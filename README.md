# RL_Algorithms
Reinforcement learning algorithms with pytorch
## supported algorithms

### Online RL

Interact with the environment during training.

| algorithm                                                    | discrete control | continuous control |
| ------------------------------------------------------------ | ---------------- | ------------------ |
| [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) | ✔                | ⛔                  |
| [Double DQN (DDQN)](https://arxiv.org/abs/1509.06461)        | ✔                | ⛔                  |
| [Deep Deterministic Policy Gradients (DDPG)](https://arxiv.org/abs/1509.02971) | ⛔                | ✔                  |
| [Soft Actor-Critic (SAC)](https://arxiv.org/abs/1812.05905)  | ⛔                | ✔                  |

### Offline RL

Use the existing data set  for training, and there is no interaction with the environment during training.

| algorithm                                                    | discrete control | continuous control |
| ------------------------------------------------------------ | ---------------- | ------------------ |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | ⛔                | ✔                  |

