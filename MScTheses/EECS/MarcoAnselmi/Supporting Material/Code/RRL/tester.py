import torch
import numpy as np


def tester(policy, env):
    """
        Returns a generator to roll out each episode given a trained policy and
        environment to test on.
        Parameters:
            policy - The trained policy to test
            env - The environment to evaluate the policy on
            render - Specifies whether to render or not

        Return:
            A generator object rollout, or iterable, which will return the latest
            episodic length and return on each iteration of the generator.
        Note:
            If you're unfamiliar with Python generators, check this out:
                https://wiki.python.org/moin/Generators
            If you're unfamiliar with Python "yield", check this out:
                https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do
    """

    test_ret = []
    test_tmst = []
    successes = 0

    num_episodes = 0

    # Rollout until 1000 episods have been completed
    while num_episodes < 1000:
        num_episodes += 1
        obvs = env.reset()
        hidden = policy.initHidden()
        done = False

        # number of timesteps so far
        t = 0

        # Logging data
        ep_ret = 0  # episodic return

        while not done:
            t += 1

            # Render environment if specified, off by default
            env.render()

            # Query deterministic action from policy and run it
            obvs = torch.tensor(obvs, dtype=torch.float)
            obvs = obvs.unsqueeze(0).unsqueeze(0)
            action, hidden = policy(obvs, hidden)
            action = action.squeeze(0).squeeze(0)
            obvs, rwd, done, _ = env.step(action.detach().numpy())

            # Sum all episodic rewards as we go along
            ep_ret += rwd

        test_ret.append(ep_ret)
        test_tmst.append(t)
        if obvs[6] == 1.0 and obvs[7] == 1.0 and not env.lander.awake:
            successes += 1

        # Track episodic length
        print("Episode Duration: %d || Episode Reward: %.2f" % (t, ep_ret))

    mean_rew = np.mean(test_ret)
    max_rew = np.max(test_ret)
    min_rew = np.min(test_ret)
    med_rew = np.median(test_ret)
    mean_tmst = np.mean(test_tmst)
    succ_perc = (successes/1000) * 100

    print("Mean Reward: %.2f || Median Reward: %.2f || Minimum Reward: %.2f || Maximum Reward: %.2f || Mean Timesteps: %.2f || Success Percentage: %.2f"
          %(mean_rew, med_rew, min_rew, max_rew, mean_tmst, succ_perc))
