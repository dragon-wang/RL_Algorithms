import numpy as np
try:
    import d4rl
except ImportError:
    print('No module named "d4rl" , and you can install in https://github.com/rail-berkeley/d4rl')

try:
    import d4rl_atari
except ImportError:
    print('No module named "d4rl_atari" , and you can install in https://github.com/takuseno/d4rl-atari')


def get_d4rl_dataset(env, get_num=None) -> dict:
    """
    d4rl dataset: https://github.com/rail-berkeley/d4rl
    install: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
    :param get_num: how many data get form dataset
    """
    dataset = d4rl.qlearning_dataset(env)
    if get_num is None:
        data = dict(
            obs=dataset['observations'],
            acts=dataset['actions'],
            rews=dataset['rewards'],
            next_obs=dataset['next_observations'],
            done=dataset['terminals']
        )
    else:
        data_num = dataset['actions'].shape[0]
        ind = np.random.choice(data_num, size=get_num, replace=False)
        data = dict(
            obs=dataset['observations'][ind],
            acts=dataset['actions'][ind],
            rews=dataset['rewards'][ind],
            next_obs=dataset['next_observations'][ind],
            done=dataset['terminals'][ind]
        )

    return data


def get_d4rl_dataset_atari(env) -> dict:
    """
    d4rl atari dataset: https://github.com/takuseno/d4rl-atari
    install: pip install git+https://github.com/takuseno/d4rl-atari
    """
    dataset = env.get_dataset()
    data = dict(
        obs=dataset['observations'],
        acts=dataset['actions'],
        rews=dataset['rewards'],
        done=dataset['terminals']
    )

    return data
