from abc import abstractmethod, ABC, ABCMeta
import torch
from utils import log_tools
from utils.train_tools import explore_before_train, evaluate
import numpy as np
import os

class PolicyBase(ABC):
    def __init__(self,
                 env,  # RL environment object
                 gamma, # The decay factor
                 eval_freq, # How often (time steps) the policy is evaluated. it will not evaluate the agent during train if eval_freq < 0.
                 max_train_step, # The max train step
                 train_id, # The name and path to save model and log tensorboard
                 log_interval, # The number of steps taken to record the model and the tensorboard
                 resume, # Whether load the last saved model and continue to train
                 device,  # The device. Choose cpu or cuda
                 ):
        self.env = env
        self.gamma = gamma
        self.eval_freq = eval_freq
        self.max_train_step = max_train_step
        self.train_id = train_id
        self.log_interval = log_interval
        self.resume = resume
        self.device = torch.device(device)

        self.result_dir = os.path.join(log_tools.ROOT_DIR, "run/results", self.train_id)
        self.checkpoint_path = os.path.join(self.result_dir, "checkpoint.pth")
    
    @abstractmethod
    def choose_action(self, obs, eval=False):
        """Select an action according to the observation

        Args:
            obs (_type_): The observation
            eval (bool): Whether used in evaluation
        """
        pass

    @abstractmethod
    def train(self):
        """The main body of rl algorithm
        """
        pass

    @abstractmethod
    def learn(self):
        """The main loop of training process
        """
        pass

    @abstractmethod
    def store_agent_checkpoint(self):
        """Save training data. (e.g. neural network parameters, optimizer parameters, training steps, ...)
        """
        pass

    @abstractmethod
    def load_agent_checkpoint(self):
        """Load training data
        """
        pass


class OffPolicyBase(PolicyBase):
    def __init__(self, 
                 replay_buffer,  # The replay buffer
                 explore_step,  # Steps to explore the environment before training
                 **kwargs  # The parameters of the parent class
                 ):
        super().__init__(**kwargs)

        self.replay_buffer = replay_buffer
        self.explore_step = explore_step

        self.train_step = 0
        self.episode_num = 0

    def choose_action(self, obs, eval=False):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def learn(self):
        # Make the directory to save the training results that consist of checkpoint files and tensorboard files
        log_tools.make_dir(self.result_dir)
        tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)
        
        explore_before_train(self.env, self.replay_buffer, self.explore_step)
        print("==============================start train===================================")
        obs = self.env.reset()

        episode_reward = 0
        episode_length = 0

        # The main loop of "choose action -> act action -> add buffer -> train policy"
        while self.train_step < self.max_train_step:
            action = self.choose_action(np.array(obs), eval=False)
            next_obs, reward, done, _ = self.env.step(action)
            episode_reward += reward
            self.replay_buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs
            episode_length += 1

            train_summaries = self.train()

            if done:
                self.episode_num += 1
                obs = self.env.reset()
            
                print(f"Episode Num: {self.episode_num} Episode Length: {episode_length} "
                      f"Episode Reward: {episode_reward:.2f} Time Step: {self.train_step}")
                tensorboard_writer.log_learn_data({"episode_length": episode_length,
                                                   "episode_reward": episode_reward}, self.train_step)
                episode_reward = 0
                episode_length = 0

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                tensorboard_writer.log_train_data(train_summaries, self.train_step)

            if self.eval_freq > 0 and self.train_step % self.eval_freq == 0:
                evaluate_summaries = evaluate(agent=self, episode_num=10)
                tensorboard_writer.log_eval_data(evaluate_summaries, self.train_step)

    def store_agent_checkpoint(self):
        raise NotImplementedError

    def load_agent_checkpoint(self):
        raise NotImplementedError


class OfflineBase(PolicyBase):
    def __init__(self, data_buffer, **kwargs):
        super().__init__(**kwargs)
        self.data_buffer = data_buffer
        self.train_step = 0

    def choose_action(self, obs, eval=True):
        """In offline settings, 
        since the agent does not interact with the environment during training, 
        this function is only used during evaluation.
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def learn(self):
        # Make the directory to save the training results that consist of checkpoint files and tensorboard files
        log_tools.make_dir(self.result_dir)
        tensorboard_writer = log_tools.TensorboardLogger(self.result_dir)

        if self.resume:
            self.load_agent_checkpoint()
        else:
            # delete tensorboard log file
            log_tools.del_all_files_in_dir(self.result_dir)

        while self.train_step < self.max_train_step:
            train_summaries = self.train()

            if self.train_step % self.log_interval == 0:
                self.store_agent_checkpoint()
                tensorboard_writer.log_train_data(train_summaries, self.train_step)

            if self.eval_freq > 0 and self.train_step % self.eval_freq == 0:
                evaluate_summaries = evaluate(agent=self, episode_num=10)
                tensorboard_writer.log_eval_data(evaluate_summaries, self.train_step)

    def store_agent_checkpoint(self):
        raise NotImplementedError

    def load_agent_checkpoint(self):
        raise NotImplementedError