import d4rl
import d4rl_atari


def get_d4rl_dataset(env) -> dict:
    """
    d4rl dataset: https://github.com/rail-berkeley/d4rl
    install: pip install git+https://github.com/rail-berkeley/d4rl@master#egg=d4rl
    """
    dataset = d4rl.qlearning_dataset(env)
    data = dict(
        obs=dataset['observations'],
        acts=dataset['actions'],
        rews=dataset['rewards'],
        next_obs=dataset['next_observations'],
        done=dataset['terminals']
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
