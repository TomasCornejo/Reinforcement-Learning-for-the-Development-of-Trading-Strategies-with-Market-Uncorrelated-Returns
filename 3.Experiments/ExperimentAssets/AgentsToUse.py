import numpy as np
from stable_baselines3 import PPO, A2C, DDPG

# Define PPO Agent
class PPOAgent:
    def __init__(self, env, n_steps, seed=0):
        self.model_type = "SB3"
        self.model = PPO("MlpPolicy", env, n_steps=n_steps ,verbose=1, device='cuda', seed=seed)
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define A2C Agent
class A2CAgent:
    def __init__(self, env, n_steps, seed=0):
        self.model_type = "SB3"
        self.model = A2C("MlpPolicy", env, n_steps=n_steps, verbose=1, device='cuda', seed=seed)
    
    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define DDPG Agent
class DDPGAgent:
    def __init__(self, env, seed=0 , hp_params=None):
        self.model_type = "SB3"
        self.model = DDPG("MlpPolicy", env, verbose=0, device='cuda', seed=seed, **hp_params)

    def predict(self, obs):
        action, _ = self.model.predict(obs)
        return action
    
# Define Random Agent
class RandomAgent:
    def __init__(self, env):
        self.model_type = "Random"
        self.env = env
        self.tickers_len = len(env.get_attr('tickers')[0])
    
    def predict(self, obs):
        return np.random.uniform(-1, 1, self.tickers_len).reshape(1, self.tickers_len)

# Define SP500 Agent
class SP500Agent:
    def __init__(self, env):
        self.model_type = "SP500"
        self.env = env
        self.tickers = env.get_attr('tickers')[0]
        self.tickers_len = len(self.tickers)
        self.SP500idx = self.tickers.index('SP500')

        
        print("Tickers en init SP500Agent")
        print(self.env.get_attr('tickers')[0])
    
    def predict(self, obs):
        current_step = int(self.env.get_attr('episode_step')[0])
        # create a empty array of zeros
        actions = np.zeros(self.tickers_len).reshape(1, self.tickers_len)
        if current_step == 0:
            # Buy SP500
            actions[0, self.SP500idx] = 1
        return actions