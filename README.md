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
| [Twin Delayed Deep Deterministic policy gradient(TD3)](https://arxiv.org/abs/1802.09477) | ⛔                | ✔                  |

### Offline RL

Use the existing data set  for training, and there is no interaction with the environment during training.

| algorithm                                                    | discrete control | continuous control |
| ------------------------------------------------------------ | ---------------- | ------------------ |
| [Conservative Q-Learning (CQL)](https://arxiv.org/abs/2006.04779) | ✔                | ✔                  |
| [Batch-Constrained deep Q-learning (BCQ)](https://arxiv.org/abs/1812.02900) |                  | ✔                  |
| [Policy in the Latent Action Space (PLAS)](https://arxiv.org/abs/2011.07213) | ⛔                | ✔                  |

## Requirements

```
|Python 3.7        |
|Pytorch 1.7.1	   |
|tensorboard 2.7.0 | To view the training curve in real time, 
|tqdm 4.62.3       | To show progress bar.
|numpy 1.21.3	   | 

|gym 0.19.0        | 
|box2d-py 2.3.8    | Include Box2d env, e.g,"BipedalWalker-v2" and "LunarLander-v2".
|atari-py 0.2.6    | Include Atari env, e.g, "Pong", "Breakout" and "SpaceInvaders".
|mujoco-py 2.0.2.8 | Include Mujoco env, e.g, "Hopper-v2", "Ant-v2" and "HalfCheetah-v2".

|d4rl 1.1          | Only used in Offline RL. Include offline dataset of Mujoco, CARLA and so on.
                     (Can be installed in "https://github.com/rail-berkeley/d4rl")
|d4rl-atari 0.1    | Only used in Offline RL. Include offline dataset of Atari.
                     (Can be installed in "https://github.com/takuseno/d4rl-atari")
```

## Quick start

### To train the agents on the environments

```shell
cd run

# train DQN
python dqn_gym.py --env=CartPole-v0 --train_id=dqn_test  

# train DDPG
python python ddpg_gym.py --env=Pendulum-v0 --train_id=ddpg_Pendulum-v0  

# train SAC
python sac_gym.py --env=Pendulum-v0 --train_id=sac_Pendulum-v0  
python sac_mujoco.py --env=Hopper-v2 --train_id=sac_Hopper-v2 --max_train_step=2000000 --auto

# train TD3
python td3_gym.py --env=Pendulum-v0 --train_id=td3_Pendulum-v0
python td3_mujoco.py --env=Hopper-v2 --train_id=td3_Hopper-v2  

# train BCQ
python bcq_mujoco.py --train_id=bcq_hopper-mudium-v2 --env=hopper-medium-v2  --device=cuda

# train PLAS
python plas_mujoco.py --train_id=plas_hopper-mudium-v2 --env=hopper-medium-v2 --device=cuda

# train CQL
python cql_mujoco.py --train_id=cql_hopper-mudium-v2 --env=hopper-medium-v2 --auto_alpha --device=cuda --entropy_backup --with_lagrange --lagrange_thresh=10.0
```

Some command line common parameters:

+ `--env`: the name of environment.(`--env=xxx`)
+ `--capacity`: the max size of replay buffer.(`--capacity=xxx`)
+ `--batch_size`: the size of batch that sampled from buffer.(`--batch_size=xxx`)
+ `--explore_step`: the steps of exploration before train.(`--explore_step=xxx`)
+ `--max_train_step`: the max train step.(`--max_train_step=xxx`)
+ `--log_interval`: the number of steps taken to record the model and the tensorboard.(`--log_interval=xxx`)
+ `--train_id`: path to save model and log tensorboard.(`--train_id=xxx`)
+ `--resume`: whether load the last saved model to train.(`--resume`)
+ `--device`: choose device (`--device=cpu` or `--device=cuda`).
+ `--eval`: whether evaluate the trained agent.(`--eval`)
+ `--seed`: the random seed of env or neural network(`--seed=xxx`)

The specific parameters for each algorithm can be viewed in the "xxx.py" files under the "run" folder. Of course I have also provided some default parameters.

**Note that your trained model and tensorboard files are stored in the "results/your train_id" folder.**

### Use tensorboard to view the training curve

```
cd run

tensorboard --logdir results
```

You can then view the training curve by typing "http://localhost:6006/" into your browser.

## Continue to train from last checkpoint

You just need to add `--resume` after your command line, such as:

```shell
python sac_mujoco.py --env=Hopper-v2 --train_id=sac_Hopper-v2 --max_train_step=2000000 --auto --resume
```

**Note that the "train_id" must be the same as your last training id.**

## Evaluate the trained agent

You can view the display of the trained agent via `--eval`, such as:

```shell
python sac_mujoco.py --env=Hopper-v2 --train_id=sac_Hopper-v2 --eval
```

**Note that the "train_id" must be the same as the id of the agent you want to see.**
