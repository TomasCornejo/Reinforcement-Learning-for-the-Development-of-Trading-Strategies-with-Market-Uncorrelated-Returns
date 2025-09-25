from DDPGExp import trainDDPG
from CustomCallback import EarlyStoppingCallback
import optuna
import numpy as np
import os

VAR_INITIAL_BALANCE = 100000
VAR_EPISODE_STEP_CONST = 64 
VAR_N_TOTAL_EPISODES = 900 
VAR_TESTING_POINTS = 50
VAR_ID_PATH = '000003'
VAR_DESCRIPTION = 'Base con return memory size=3 + netprofit'
VAR_DATA_PATH = '../Data/enriched_data'
VAR_TRIALS = 100
VAR_STUDY_NAME = 'OptunaDDPGExpBase_RMemorySize3_NetProfit'
VAR_PATH_TO_SAVE = f'OptimizeIterationsResults/{VAR_STUDY_NAME}'

# CONTEXT VARIABLES
CTX_REWARD_TYPE = 'netprofit' #  'sharpewithcorrelation', 'netprofit'
CTX_SHUFFLE_DAYS = False
CTX_CORRELATION_PUNISHMENT = 1
CTX_RETURN_MEMORY_SIZE = 3

# CONTEXT VARIABLES BASE
# CTX_REWARD_TYPE = 'sharpewithcorrelation'    # 'sharpewithcorrelation', 'netprofit'
# CTX_SHUFFLE_DAYS = False
# CTX_CORRELATION_PUNISHMENT = 1
# CTX_RETURN_MEMORY_SIZE = 5


def objective(trial):
    # Define the hyperparameters
    hp_params = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True),
        "batch_size": trial.suggest_int("batch_size", 32, 512),
        "gamma": trial.suggest_float("gamma", 0.9, 0.999),
        "tau": trial.suggest_float("tau", 0.001, 0.1),
        "gradient_steps" : trial.suggest_categorical("gradient_steps", [-1, 1, 2]),
        "train_freq" : trial.suggest_categorical("train_freq", [1, 2, 3])
    }
    #Generate a numpy random seed for agent
    VAR_SEED_SHUFFLE = np.random.randint(0, 1000) 
    VAR_SEED_AGENT = np.random.randint(0, 1000)

    result = trainDDPG(
         SHUFFLE_DAYS=CTX_SHUFFLE_DAYS,
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
         HP_PARAMS=hp_params,
         SAVE_DATA=False
    )
    return result

if not os.path.exists(VAR_PATH_TO_SAVE):
    os.makedirs(VAR_PATH_TO_SAVE)


study = optuna.create_study(study_name=VAR_STUDY_NAME+VAR_ID_PATH , direction='maximize')
early_stopping_callback = EarlyStoppingCallback(early_stopping_rounds=10, direction='maximize')

study.optimize(objective, callbacks=[early_stopping_callback] , n_trials=VAR_TRIALS)

best_params = study.best_params
best_cum_reward_mean = study.best_value

print(f"Best params: {best_params}")
print(f"Best cum reward mean: {best_cum_reward_mean}")

study.trials_dataframe().to_csv(f'{VAR_PATH_TO_SAVE}/trialsData.csv')
