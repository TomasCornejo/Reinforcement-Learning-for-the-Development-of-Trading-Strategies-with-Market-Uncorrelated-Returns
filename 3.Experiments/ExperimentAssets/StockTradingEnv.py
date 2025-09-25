import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class StockTradingEnv(gym.Env):
    metadata = {'render_modes': ['human']}
    
    def __init__(self, stock_data , benchmark_data, return_memory_size, reward_type, env_correlation_punishment, env_initial_balance):
        super(StockTradingEnv, self).__init__()

        # Define the benchmark
        self.benchmark_data = benchmark_data
        self.reward_type = reward_type
        self.env_correlation_punishment = env_correlation_punishment
        self.env_initial_balance = env_initial_balance
        
        # Initialize of memory
        self.return_memory_size = return_memory_size
        self.last_return_memory_benchmark = []
        self.last_return_memory_portfolio = []
        self.benchmark_portfolio_correlation = 0

        self.portfolio_return = []

        self.benchmark_portfolio_total_correlation = 0

        self.reward_returned = 0
        self.previous_reward = 0
        self.ratio_sharpe = 0

        self.transaction_cost = 0.0001

        # Remove any empty DataFrames from the stock data
        self.stock_data = {ticker: df for ticker, df in stock_data.items() if not df.empty}     
        self.tickers = list(self.stock_data.keys())
        print(f"Stock data has been loaded for {len(self.tickers)} stocks.")
        print(f"The tickers are: {self.tickers}")
        
        if not self.tickers:
            raise ValueError("All provided stock data is empty")
        
        # Calculate the size of one stock's data
        sample_df = next(iter(self.stock_data.values()))
        self.n_features = len(sample_df.columns)
        
        # Define action and observation space
        self.action_space = spaces.Box(low=-1, high=1, shape=(len(self.tickers),), dtype=np.float32)
        
        # Observation space: 
        #           len(ticker x (price data and features)) 
        #           + balance + net worth 
        #           + len (ticker x (shares held))
        #(NEW)      + last reward
        #(NEW)      + ratio between the last reward and the past reward(eval 3 o 5)
        #(NEW)      + last ratio of sharpe (calculus)
        #(NEW)      + last correlation between benchmark and portfolio
        #           + max net worth + current step
        self.obs_shape = self.n_features * len(self.tickers) + 2 + len(self.tickers) + 4 + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_shape,), dtype=np.float32)
        
        # Initialize account balance
        self.initial_balance = self.env_initial_balance
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        
        # Set the current step and the episode step
        self.current_step = 0
        self.episode_step = 0
        self.current_date = self.benchmark_data.index[self.current_step].date().strftime('%Y-%m-%d')
        
        # Initialize the current date for separate days
        self.is_last_tick_of_day = False

    def hard_reset(self):
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.episode_step = 0
        
        self.is_last_tick_of_day = False
        self.last_return_memory_benchmark = []
        self.last_return_memory_portfolio = []

        self.portfolio_return = []

        self.benchmark_portfolio_correlation = 0
        self.benchmark_portfolio_total_correlation = 0

        self.reward_returned = 0
        self.previous_reward = 0
        self.ratio_sharpe = 0
        
        #Especifico del hard reset
        self.current_step = 0
        self.current_date = self.benchmark_data.index[self.current_step].date().strftime('%Y-%m-%d')

        return self._next_observation(), {}

        
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.balance = self.initial_balance
        self.net_worth = self.initial_balance
        self.last_net_worth = self.initial_balance
        self.max_net_worth = self.initial_balance
        self.shares_held = {ticker: 0 for ticker in self.tickers}
        self.total_shares_sold = {ticker: 0 for ticker in self.tickers}
        self.total_sales_value = {ticker: 0 for ticker in self.tickers}
        self.episode_step = 0
        self.is_last_tick_of_day = False
        self.last_return_memory_benchmark = []
        self.last_return_memory_portfolio = []

        self.portfolio_return = []

        self.benchmark_portfolio_correlation = 0
        self.benchmark_portfolio_total_correlation = 0

        self.reward_returned = 0
        self.previous_reward = 0
        self.ratio_sharpe = 0

        return self._next_observation(), {}
    
    def _next_observation(self):
        frame = np.zeros(self.obs_shape)
        offset = 0
        for ticker in self.tickers:
            df = self.stock_data[ticker]
            if self.current_step < len(df):
                frame[offset:offset+self.n_features] = df.iloc[self.current_step].values
            elif len(df) > 0:
                frame[offset:offset+self.n_features] = df.iloc[-1].values
            offset += self.n_features
        frame[offset] = self.balance
        offset += 1
        frame[offset] = self.net_worth
        offset += 1
        frame[offset:offset+len(self.tickers)] = [self.shares_held[ticker] for ticker in self.tickers]
        offset += len(self.tickers)
        # Add the last reward
        frame[offset] = self.reward_returned
        offset += 1
        # Add the ratio between the last reward and the past reward
        if self.previous_reward != 0:
            frame[offset] = self.reward_returned / self.previous_reward
        else:
            frame[offset] = 0
        offset += 1
        # Add the last ratio of sharpe
        frame[offset] = self.ratio_sharpe
        offset += 1
        # Add the last correlation between benchmark and portfolio
        frame[offset] = self.benchmark_portfolio_correlation
        offset += 1
        # Add the max net worth and the current step
        frame[offset] = self.max_net_worth
        offset += 1
        frame[offset] = self.episode_step
        offset += 1
        if offset != self.obs_shape:
            raise ValueError(f"Offset {offset} does not match observation shape {self.obs_shape}")
        return frame

    def step(self, actions):
        self.previous_reward = self.reward_returned
        self.current_step += 1
        self.episode_step += 1
        self.current_date = self.benchmark_data.index[self.current_step].date().strftime('%Y-%m-%d')

        if self.is_last_tick_of_day:
            return self._next_observation(), 0, True, False, {}
        
        current_prices = {}
        for i, ticker in enumerate(self.tickers):
            current_prices[ticker] = self.stock_data[ticker].iloc[self.current_step]['Close']
            action = actions[i]

            # Buy or sell the stocks just if the price is different from 0
            if current_prices[ticker] != 0.0:
                if action > 0:  # Buy
                    shares_to_buy = int(self.balance * action / current_prices[ticker])
                    cost = shares_to_buy * current_prices[ticker]
                    if cost < self.balance:
                        self.balance = self.balance - (cost * (1 + self.transaction_cost))
                        self.shares_held[ticker] += shares_to_buy
                elif action < 0:  # Sell
                    shares_to_sell = int(self.shares_held[ticker] * abs(action))
                    sale = shares_to_sell * current_prices[ticker]
                    self.balance = self.balance + (sale * (1 - self.transaction_cost))
                    self.shares_held[ticker] -= shares_to_sell
                    self.total_shares_sold[ticker] += shares_to_sell
                    self.total_sales_value[ticker] += sale
            else:
                print(f"Price for {ticker} in {self.current_date} is 0.0")
        

        self.last_net_worth = self.net_worth
        self.net_worth = self.balance + sum(self.shares_held[ticker] * current_prices[ticker] for ticker in self.tickers)
        self.max_net_worth = max(self.net_worth, self.max_net_worth)
        

        # reward = self.net_worth - self.last_net_worth # Old Reward
        done = (self.net_worth <= 0 or self.is_last_tick_of_day)

        # Calculate the return of the benchmark (just from second step)
        if self.episode_step > 0:
            benchmark_return = self.benchmark_data.iloc[self.current_step]['Close'] / self.benchmark_data.iloc[self.current_step - 1]['Close'] - 1
            #Add the return to the benchmark to the memory but if the memory is full, remove the oldest return
            if len(self.last_return_memory_benchmark) >= self.return_memory_size:
                self.last_return_memory_benchmark.pop(0)
            self.last_return_memory_benchmark.append(benchmark_return)
            

            # Calculate the return of the portfolio
            portfolio_return = self.net_worth / self.last_net_worth - 1
            self.portfolio_return.append(portfolio_return)
            # Add the return to the portfolio to the memory but if the memory is full, remove the oldest return
            if len(self.last_return_memory_portfolio) >= self.return_memory_size:
                self.last_return_memory_portfolio.pop(0)
            self.last_return_memory_portfolio.append(portfolio_return)

        
        # Calculate the pearson correlation between the benchmark memory and the portfolio memory just if the memory is full
        if len(self.last_return_memory_benchmark) == self.return_memory_size and \
           len(self.last_return_memory_portfolio) == self.return_memory_size:
            self.benchmark_portfolio_correlation = pearsonccs([self.last_return_memory_benchmark, self.last_return_memory_portfolio])[0][1]   
        else:
            self.benchmark_portfolio_correlation = 0
        
        # Update the is_last_tick_of_day variable
        self.is_last_tick_of_day = self.benchmark_data.index[self.current_step].date().strftime('%Y-%m-%d') \
            != self.benchmark_data.index[self.current_step + 1].date().strftime('%Y-%m-%d')
        
        # Set Ratio of Sharpe
        if self.benchmark_portfolio_correlation != 0 and np.std(self.last_return_memory_portfolio) != 0:
            self.ratio_sharpe = (((np.prod(np.array(self.last_return_memory_portfolio)+1)-1)*self.initial_balance)/np.std(self.last_return_memory_portfolio)) 
        else:
            self.ratio_sharpe = (np.prod(np.array(self.last_return_memory_portfolio)+1)-1)*self.initial_balance

        # Set the reward
        if self.reward_type == 'sharpe':
            self.reward_returned = self.ratio_sharpe
        elif self.reward_type == 'sharpewithcorrelation':
            if self.benchmark_portfolio_correlation != 0 and np.std(self.last_return_memory_portfolio) != 0:
                self.reward_returned = self.ratio_sharpe * (1 - abs(self.benchmark_portfolio_correlation)*self.env_correlation_punishment)
            else:
                self.reward_returned = self.ratio_sharpe
        else:
            self.reward_returned = self.net_worth - self.last_net_worth
        
        obs = self._next_observation()
        return obs, self.reward_returned, done, False, {}
    
    def render(self, mode='human'):
        profit = self.net_worth - self.initial_balance
        print(f'Step: {self.current_step}')
        print(f'Benchmark-Portfolio Correlation: {self.benchmark_portfolio_correlation:.2f}')
        print(f'Balance: {self.balance:.2f}')
        for ticker in self.tickers:
            print(f'{ticker} Shares held: {self.shares_held[ticker]}')
        print(f'Net worth: {self.net_worth:.2f}')
        print(f'Profit: {profit:.2f}')

    def close(self):
        pass

def pearsonccs(samples):
    C = np.cov(samples)
    diag = np.diag(C)
    N = np.sqrt(np.outer(diag, diag))
    N[N == 0] = 1
    return C / N