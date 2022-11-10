import torch


def rollout(policy, env):
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
    # Rollout until user kills process
    while True:
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

        # Track episodic length
        print("Episode Duration: %d || Episode Reward: %.2f" % (t, ep_ret))
