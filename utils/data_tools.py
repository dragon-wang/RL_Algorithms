import d4rl


def get_d4rl_dataset(env):
    dataset = d4rl.qlearning_dataset(env)
    data = dict(
        obs=dataset['observations'],
        acts=dataset['actions'],
        rews=dataset['rewards'],
        next_obs=dataset['next_observations'],
        done=dataset['terminals']
    )

    return data
