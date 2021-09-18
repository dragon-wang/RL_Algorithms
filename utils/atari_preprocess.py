import gym
from gym.wrappers import atari_preprocessing
from gym.wrappers import FrameStack
import numpy as np


def make_env(env_id,  # 环境id
             noop_max=30,  # 最大的no-op操作步数
             frame_skip=4,  # 跳帧步数
             screen_size=84,  # 帧的尺寸
             terminal_on_life_loss=True,  # 是否在一条命没后结束Episode
             grayscale_obs=True,  # True的话返回灰度图，否则返回RGB彩色图
             grayscale_newaxis=False,  # 将输出的灰度图由2维转换为1维
             scale_obs=True,  # 是否对obs标准化到[0,1]
             num_stack=4,  # 叠加帧的步数
             lz4_compress=False,  # 是否使用lz4压缩
             obs_LazyFramesToNumpy=True,  # 是否将输出的图像由LazyFrames转化为numpy
             ):

    assert gym.envs.registry.spec(env_id).entry_point == 'gym.envs.atari:AtariEnv', "env is not Atari"

    env = gym.make(env_id)
    env = atari_preprocessing.AtariPreprocessing(env=env,
                                                 noop_max=noop_max,
                                                 frame_skip=frame_skip,
                                                 screen_size=screen_size,
                                                 terminal_on_life_loss=terminal_on_life_loss,
                                                 grayscale_obs=grayscale_obs,
                                                 grayscale_newaxis=grayscale_newaxis,
                                                 scale_obs=scale_obs)
    env = FrameStack(env, num_stack=num_stack, lz4_compress=lz4_compress)
    if obs_LazyFramesToNumpy:
        env = ObsLazyFramesToNumpy(env)
    return env


class ObsLazyFramesToNumpy(gym.Wrapper):
    def __init__(self, env):
        super(ObsLazyFramesToNumpy, self).__init__(env)

    def reset(self, **kwargs):
        obs = self.env.reset()
        return np.array(obs)

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return np.array(next_obs), reward, done, info

