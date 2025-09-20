from stable_baselines3.common.callbacks import BaseCallback

class SaveTrainingRewardsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.rewards = []

    def _on_step(self) -> bool:
        # Get the reward value
        reward = self.locals["rewards"][-1]
        self.rewards.append(reward)
        return True
