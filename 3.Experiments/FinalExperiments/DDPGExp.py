# VARIABLES DE TRAINING
VAR_INITIAL_BALANCE = 100000
VAR_EPISODE_STEP_CONST = 64 
VAR_N_TOTAL_EPISODES = 900 # 200
VAR_TESTING_POINTS =  50 # 10
VAR_SEED_SHUFFLE = 3
VAR_SEED_AGENT = 3
VAR_ID_PATH = '000003'
VAR_PATH_TO_SAVE = f'ResultsData/RSCORR_TEST200/{VAR_ID_PATH}'
VAR_DESCRIPTION = 'Test con Reward= Ratio de Sharpe y correlacion como castigo'

VAR_DATA_PATH = '../Data/enriched_data'

# CONTEXT VARIABLES
CTX_REWARD_TYPE = 'sharpewithcorrelation' # 'sharpe', 'sharpewithcorrelation', 'netprofit'
CTX_SHUFFLE_DAYS = True
CTX_CORRELATION_PUNISHMENT = 1
CTX_RETURN_MEMORY_SIZE = 5


import pandas as pd
import numpy as np
import os
import pickle
import time
from datetime import datetime
from StockTradingEnv import StockTradingEnv
from AgentsToUse import DDPGAgent, RandomAgent, SP500Agent
from CustomCallback import SaveTrainingRewardsCallback
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.utils import get_device
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def trainDDPG(SHUFFLE_DAYS=CTX_SHUFFLE_DAYS,
         DATA_PATH=VAR_DATA_PATH,
         INITIAL_BALANCE=VAR_INITIAL_BALANCE, 
         EPISODE_STEP_CONST=VAR_EPISODE_STEP_CONST, 
         N_TOTAL_EPISODES=VAR_N_TOTAL_EPISODES, 
         TESTING_POINTS=VAR_TESTING_POINTS, 
         SEED_SHUFFLE=VAR_SEED_SHUFFLE,
         SEED_AGENT=VAR_SEED_AGENT, 
         ID_PATH=VAR_ID_PATH, 
         PATH_TO_SAVE=VAR_PATH_TO_SAVE, 
         DESCRIPTION=VAR_DESCRIPTION,  
         REWARD_TYPE=CTX_REWARD_TYPE, 
         CORRELATION_PUNISHMENT=CTX_CORRELATION_PUNISHMENT,
         RETURN_MEMORY_SIZE=CTX_RETURN_MEMORY_SIZE,
         HP_PARAMS={},
         SAVE_DATA=True):
    # Setting Metadata
    metadata = {}
    metadata['INITIAL_BALANCE'] = INITIAL_BALANCE
    metadata['EPISODE_STEP_CONST'] = EPISODE_STEP_CONST
    metadata['N_TOTAL_EPISODES'] = N_TOTAL_EPISODES
    metadata['TESTING_POINTS'] = TESTING_POINTS
    metadata['SEED_SHUFFLE'] = SEED_SHUFFLE
    metadata['SEED_AGENT'] = SEED_AGENT
    metadata['ID_PATH'] = ID_PATH
    metadata['PATH_TO_SAVE'] = PATH_TO_SAVE
    metadata['DESCRIPTION'] = DESCRIPTION
    metadata['RETURN_MEMORY_SIZE'] = RETURN_MEMORY_SIZE
    metadata['REWARD_TYPE'] = REWARD_TYPE
    metadata['CORRELATION_PUNISHMENT'] = CORRELATION_PUNISHMENT
    metadata['SHUFFLE_DAYS'] = SHUFFLE_DAYS
    metadata['HP_PARAMS'] = HP_PARAMS

    tickers = ['AAPL','ABCB','CSCO','DAL','EAST','ESPR','F',
           'GOOGL','JPM','META','MSFT','NVDA','PFE','PG','PSX',
           'SGLY','SP500','TSLA','UNH','USA','WMT','XOM']

    # ticker of benchmark
    benchmark = 'SP500'

    # Getting data
    training_data = get_data(tickers, data_path=DATA_PATH)

    if SHUFFLE_DAYS:
    # Shuffling data
        datasetToTrain = shuffle_dataset_dictionary(training_data, seed=SEED_SHUFFLE, benchmark=benchmark)
        benchmarkToTrain = datasetToTrain[benchmark]
    else:
        datasetToTrain = training_data
        benchmarkToTrain = training_data[benchmark]

    print("Ejecutando en :", get_device(device='cuda'))

    partition = int(N_TOTAL_EPISODES / TESTING_POINTS) - 1
    n_steps = EPISODE_STEP_CONST*partition

    #############################################################
    ################## TRAINING AND TESTING  ####################
    #############################################################

    # 1. Create the environment and train the agents
    env, ddpg_agent, random_agent , sp500_agent = create_env_and_train_agents(datasetToTrain, benchmarkToTrain, n_steps=n_steps, return_memory_size=RETURN_MEMORY_SIZE, reward_type=REWARD_TYPE, env_correlation_punishment=CORRELATION_PUNISHMENT, env_initial_balance=INITIAL_BALANCE, seed=SEED_AGENT, hp_params=HP_PARAMS)

    # 2. Test & visualize the agents
    agents = {
        'DDPG Agent': ddpg_agent,
        'Random Agent': random_agent,
        'SP500 Agent': sp500_agent
    }
    #Enable a metadata dictionary to save the duration time of the training in minutes
    start_training_time = time.time()
    metrics_resume, training_rewards_resume , trained_agents = getResumeMetrics(env, agents, episode_step_const=EPISODE_STEP_CONST, n_total_episodes=N_TOTAL_EPISODES, testing_points=TESTING_POINTS)
    end_training_time = time.time()
    metadata['TRAINING_TIME'] = (end_training_time - start_training_time) / 60

    #################################################################
    ################## TESTING WITH LAST AGENTS  ####################
    #################################################################
    
    # Step0: Lista de fechas de los testing points
    date_list = metrics_resume['SP500 Agent']['dates'] 
    date_list = [datetime.strptime(date, "%Y-%m-%d").date() for date in date_list]
    # Step1: Filter training data by the dates of the testing points
    testing_data = filter_data_for_testing(datasetToTrain, date_list)
    benchmark_testing_data = testing_data['SP500']
    # Step2: Create a new environment with the filtered data
    env_testing = DummyVecEnv([lambda: StockTradingEnv(testing_data, benchmark_testing_data, return_memory_size=RETURN_MEMORY_SIZE, reward_type=REWARD_TYPE,env_correlation_punishment=CORRELATION_PUNISHMENT, env_initial_balance=INITIAL_BALANCE)])
    # Step3: Test the best/last agents with the new environment
    start_testing_time = time.time()
    metrics_resume_testing, _ , _ = getResumeMetrics(env_testing, trained_agents, episode_step_const=EPISODE_STEP_CONST ,n_total_episodes=N_TOTAL_EPISODES, testing_points=TESTING_POINTS, just_test=True)
    end_testing_time = time.time()
    metadata['TESTING_TIME'] = (end_testing_time - start_testing_time) / 60


    #############################################################
    ################## EXPORTING RESULTS  #######################
    #############################################################

    #Creación de networths acumulados para cada agente
    for agent_name, agent_metrics in metrics_resume.items():
        net_worths_delta_list = agent_metrics['net_worths_delta']
        returns_list_multip = [1] + [(delta/INITIAL_BALANCE)+1 for delta in net_worths_delta_list]
        net_worths_delta_list = []
        for i in range(len(returns_list_multip)):
            net_worths_delta_list.append(np.prod(returns_list_multip[:i+1])*INITIAL_BALANCE)
        metrics_resume[agent_name]['net_worths_delta_acum'] = net_worths_delta_list

    #Creación de rewards Acumulada para cada agente
    for agent_name, agent_metrics in metrics_resume.items():
        rewards_list = agent_metrics['rewards']
        rewards_list_acum = [0] + [rewards_list[0]]
        for i in range(1,len(rewards_list)):
            rewards_list_acum.append(rewards_list_acum[-1] + rewards_list[i])
        metrics_resume[agent_name]['rewards_acum'] = rewards_list_acum

    #Creación de networths acumulados para cada último agente entrenado
    for agent_name, agent_metrics in metrics_resume_testing.items():
        net_worths_delta_list = agent_metrics['net_worths_delta']
        returns_list_multip = [1] + [(delta/INITIAL_BALANCE)+1 for delta in net_worths_delta_list]
        net_worths_delta_list = []
        for i in range(len(returns_list_multip)):
            net_worths_delta_list.append(np.prod(returns_list_multip[:i+1])*INITIAL_BALANCE)
        metrics_resume_testing[agent_name]['net_worths_delta_acum'] = net_worths_delta_list

    #Creación de rewards Acumulada para cada último agente entrenado
    for agent_name, agent_metrics in metrics_resume_testing.items():
        rewards_list = agent_metrics['rewards']
        rewards_list_acum = [0] + [rewards_list[0]]
        for i in range(1,len(rewards_list)):
            rewards_list_acum.append(rewards_list_acum[-1] + rewards_list[i])
        metrics_resume_testing[agent_name]['rewards_acum'] = rewards_list_acum
    
    if SAVE_DATA:
    #Create the path to export
        nowStr  = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        PATH_TO_EXPORT = f'{PATH_TO_SAVE}/{nowStr}_{SEED_AGENT}'
        exportMetricsResume(metrics_resume, metrics_resume_testing, training_rewards_resume, metadata, trained_agents , path_to_save =PATH_TO_EXPORT)

    # Return the mean rewards of the last agent trained
    testing_reward_avg = np.mean(metrics_resume_testing['DDPG Agent']['rewards_acum'])

    return testing_reward_avg


def get_data(tickers, data_path):
    training_data = {}
    for tick in tickers:
        training_data[tick] = pd.read_parquet(f'{data_path}/{tick}_training_data.pk')
    return training_data

def shuffle_dataframe(df, shuffled_days):
    date_to_group = {date: group for date, group in df.groupby(df.index.date)}
    shuffled_groups = [date_to_group[date] for date in shuffled_days]
    return pd.concat(shuffled_groups)

def shuffle_dataset_dictionary(dataset_dict, seed, benchmark='SP500'):
    outputDatasetDictionary = {}
    np.random.seed(seed)

    #Get unique dates for one
    unique_days = np.unique(dataset_dict[benchmark].index.date)

    #Shuffle the unique days
    shuffled_unique_days = np.random.permutation(unique_days)

    for ticker, dataframe in dataset_dict.items():
        outputDatasetDictionary[ticker] = shuffle_dataframe(dataframe, shuffled_unique_days)
    return outputDatasetDictionary

# Function to create the environment and train the agents
def create_env_and_train_agents(data, benchmark_data, n_steps, return_memory_size, reward_type, env_correlation_punishment, env_initial_balance, seed, hp_params=None):
    # Create the environment using DummyVecEnv with training data
    env = DummyVecEnv([lambda: StockTradingEnv(data, benchmark_data, return_memory_size, reward_type,env_correlation_punishment, env_initial_balance)])
    # Normalize the environment
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=5.)
    # Train DDPG Agent
    ddpg_agent = DDPGAgent(env, seed=seed, hp_params=hp_params)
    # Declaring Random Agent
    random_agent = RandomAgent(env)
    # Declaring SP500 Agent
    sp500_agent = SP500Agent(env)
    return env, ddpg_agent, random_agent, sp500_agent 

def train_test_agent(env, agent, episode_step_const, n_total_episodes=100, testing_points=10, just_test=False):
    """
    Test a single agent and track performance metrics, with an option to visualize the results.
    #TODO Corregir documentacion
    Parameters:
    - env: The trading environment.
    - agent: The agent to be tested.
    - stock_data: Data for the stocks in the environment.
    - n_tests: Number of tests to run (default: 1000).
    - visualize: Boolean flag to enable or disable visualization (default: False).

    Returns:
    - A dictionary containing steps, balances, net worths, and shares held.
    """
    # Initialize metrics tracking
    metrics = {
        'dates': [],
        'net_worths_delta': [],
        'shares_held': [],
        'pearson_correlations': [],
        'rewards': [],
        'portfolio_returns': [],
        'stds': [],
        'sharpe_ratios': [],
        'net_worths': []
    }

    # Reset the environment before starting the tests
    obs = np.array([env.env_method('hard_reset')[0][0].tolist()]) # env.reset()#

    # Get the training partition number of episodes
    partition = int(n_total_episodes / testing_points) - 1

    training_rewards = []

    # Set the episode loop
    for i_tpoint in range(testing_points):
        # First Evaluate the agents
        print(f"--------------Testing Point {i_tpoint+1} of {testing_points}-----------------------------------")
        # Set the initial episode values

        # Turn off training and reward normalization while testing
        env.training = False
        env.norm_reward = False
        
        done = False
        cum_reward = 0
        local_pearson_correlations = []
        local_portfolio_returns = []
        metrics['dates'].append(env.get_attr('current_date')[0])
        
    
        initial_testing_net_worth = env.get_attr('net_worth')[0]

        #TODO Take off Later
        #print("Fecha Inicial: ", env.get_attr('current_date')[0])
        #print("Current Step Inicial: ", env.get_attr('current_step')[0])

        net_worth_list = []
        episode_step_final_testing_list = []
        last_testing_shares_held_list = []
        
        # Run the testing episode
        while not done:
            action = agent.predict(obs)
            obs, reward, done, _info = env.step(action)
            local_pearson_correlations.append(env.get_attr('benchmark_portfolio_correlation')[0])
            episode_step_final_testing_list.append(env.get_attr('episode_step')[0])
            net_worth_list.append(env.get_attr('net_worth')[0])
            last_testing_shares_held_list.append(env.get_attr('shares_held')[0])
            local_portfolio_returns.append(env.get_attr('portfolio_return')[0])
            cum_reward += reward
            
            
        #print("Fecha Final: ", env.get_attr('current_date')[0])
        #print("Current Step Final: ", env.get_attr('current_step')[0])
        #print("Episode Step Final: ", episode_step_final_testing_list[-2])
        #print("Reward: ", cum_reward)

        net_worth_delta = net_worth_list[-2] - initial_testing_net_worth
        # Append metrics
        metrics['net_worths'].append(net_worth_list[-2])
        metrics['net_worths_delta'].append(net_worth_delta)
        metrics['pearson_correlations'].append(np.mean(local_pearson_correlations))
        metrics['rewards'].append(cum_reward[0])
        metrics['portfolio_returns'].append(np.mean(local_portfolio_returns[0][1:]))
        metrics['stds'].append(np.std(local_portfolio_returns[0][1:]))
        metrics['sharpe_ratios'].append((net_worth_delta / initial_testing_net_worth) / np.std(local_portfolio_returns[0][1:]))

        # Update shares held for each ticker
        # Hacer caso especial para SB3
        if agent.model_type == "SB3":
            metrics['shares_held'].append(last_testing_shares_held_list[-2])
        else:
            metrics['shares_held'].append(last_testing_shares_held_list[-1])

        if not(just_test):

            # Turn on training and reward normalization while training
            env.training = True
            env.norm_reward = True
            #After train the agent by partition of episodes
            #print("Fecha antes train: ", env.get_attr('current_date')[0])
            #print("Current Step antes train: ", env.get_attr('current_step')[0])
            if agent.model_type == "SB3":
                #Define Callbacks to train
                callback_list = CallbackList([SaveTrainingRewardsCallback()])
                #Train the agent
                agent.model.learn(episode_step_const*partition, callback=callback_list)
                #Add the rewards to the list of training rewards
                training_rewards.append(callback_list.callbacks[0].rewards)
            else:
                # El agente Random y SP500 no se entrenan
                for i in range(episode_step_const*partition):
                    action = agent.predict(obs)
                    obs, _reward, done, _info = env.step(action)
            #print("Fecha despues train: ", env.get_attr('current_date')[0]) 
            #print("Current Step despues train: ", env.get_attr('current_step')[0])
        
        episode_done = False
        # Run steps till the end of the day just if if is just_test = False
        if not(just_test):
            while not episode_done:
                action = agent.predict(obs)
                obs, _reward, episode_done, _info = env.step(action)
        obs = env.reset()
        
    return metrics , training_rewards , agent

def getResumeMetrics(env, agents, episode_step_const, n_total_episodes=50, testing_points=5, just_test=False):
    metrics_resume = {}
    training_rewards_resume = {}
    trained_agents = {}         
    for agent_name, agent in agents.items():
        #print(f"Testing {agent_name}...")
        metrics_resume[agent_name], training_rewards_resume[agent_name], trained_agents[agent_name] = train_test_agent(env, agent, episode_step_const, n_total_episodes, testing_points,just_test)
        #print(f"Done testing {agent_name}!")
    
    #print('-'*50)
    #print('All agents tested!')
    #print('-'*50)

    return metrics_resume, training_rewards_resume , trained_agents

def filter_data_for_testing(data, date_list):
    filtered_data = {}
    for ticker, df in data.items():
        normalized_index = df.index.normalize()
        index_dates = pd.Series(normalized_index.date, index=df.index)
        filtered_df = df.loc[index_dates.isin(date_list)]
        nonsense_timestamp = pd.Timestamp('9999-12-31 23:59:59-04:00')
        placeholder_data = {
            'Open': 1,
            'High': 1,
            'Low': 1,
            'Close': 1,
            'Volume': 1,
            'MACD': 1,
            'Signal': 1,
            'RSI': 1,
            'CCI': 1,
            'ADX': 1
        }
        placeholder_row = pd.DataFrame([placeholder_data], index=[nonsense_timestamp])
        filtered_data[ticker] = pd.concat([filtered_df, placeholder_row])
    return filtered_data

def exportMetricsResume(metrics_resume, metrics_resume_testing, training_rewards_resume, metadata, trained_agents , path_to_save):
# Saving Metrics Resume
# Create folder to save the results
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)

    with open(f'{path_to_save}/metrics_resume.pkl', 'wb') as f:
        pickle.dump(metrics_resume, f)

    with open(f'{path_to_save}/metrics_resume_testing.pkl', 'wb') as f:
        pickle.dump(metrics_resume_testing, f)

    with open(f'{path_to_save}/training_rewards_resume.pkl', 'wb') as f:
        pickle.dump(training_rewards_resume, f)

    with open(f'{path_to_save}/metadata.pkl', 'wb') as f:
        pickle.dump(metadata, f)

    for agent in trained_agents:
        if trained_agents[agent].model_type == "SB3":
            trained_agents[agent].model.save(f'{path_to_save}/{agent}_model')
    #print("Metrics Resume Saved!")


if __name__ == '__main__':
    trainDDPG()