from tensorboard.backend.event_processing import event_accumulator
import pandas as pd
import os
import seaborn as sns
from matplotlib import pyplot as plt


def smooth(df, column, weight=0.6):
    """
    Smooth a column of data in the DataFrame
    """
    scalars = df[column].to_numpy()
    last = scalars[0]
    smoothed_scalars = []
    for scalar in scalars:
        smoothed_scalar = last * weight + (1 - weight) * scalar  # Calculate smoothed value
        smoothed_scalars.append(smoothed_scalar)
        last = smoothed_scalar
    df[column] = smoothed_scalars


def get_pd(tensorboard_path, tag='evaluate_data/eval_episode_reward'):
    """
    Get pandas from one tensorboard file
    """
    event_data = event_accumulator.EventAccumulator(tensorboard_path)  # a python interface for loading Event data
    event_data.Reload()
    scalars = event_data.scalars.Items(tag)
    df = pd.DataFrame(scalars)[['step', 'value']]
    return df


def get_pd_from_parent_path(parents_path, tag='evaluate_data/eval_episode_reward'):
    """
    Get pandas from tensorboard files with common parent path
    """
    child_paths = os.listdir(parents_path)
    df = pd.DataFrame(columns=['step', 'value'])
    for child_path in child_paths:
        tens_path = os.path.join(parents_path, child_path)
        if os.path.isdir(tens_path):
            event_data = event_accumulator.EventAccumulator(tens_path)  # a python interface for loading Event data
            event_data.Reload()
            scalars = event_data.scalars.Items(tag)
            df = df.append(pd.DataFrame(scalars)[['step', 'value']], ignore_index=True)
    return df


def is_parent_path(parent_path):
    child_paths = os.listdir(parent_path)
    for child_path in child_paths:
        tens_path = os.path.join(parent_path, child_path)
        if os.path.isdir(tens_path):
            return True
    return False


def plot_from_paths(path_list, label_list, tag='evaluate_data/eval_episode_reward', smooth_weight=0.6):
    """
    Plot tensorboard file from paths from path_list and with label from label_list on one figure
    """
    for i in range(len(path_list)):
        if is_parent_path(path_list[i]):
            df_temp = get_pd_from_parent_path(path_list[i], tag=tag)
        else:
            df_temp = get_pd(path_list[i], tag=tag)
        if smooth_weight > 0:
            smooth(df_temp, "value", weight=smooth_weight)
            sns.lineplot(x="step", y="value", data=df_temp, label=label_list[i])
        else:
            sns.lineplot(x="step", y="value", data=df_temp, label=label_list[i])
    plt.legend(loc="upper left")
    plt.xlabel("time step", fontsize=13)
    plt.ylabel("average reward", fontsize=13)
    plt.show()


if __name__ == '__main__':
    path_list = ["E:/PycharmProjects/RL_Algorithms/run/results/bcq/Hopper-v0/medium-expert",
                 "E:/PycharmProjects/RL_Algorithms/run/results/bear/Hopper-v0/medium-expert",
                 "E:/PycharmProjects/RL_Algorithms/run/results/cql/Hopper-v0/medium-expert",
                 ]

    label_list = ["BCQ",
                  "BEAR",
                  "CQL",
                  ]

    plot_from_paths(path_list, label_list, smooth_weight=0.7)
