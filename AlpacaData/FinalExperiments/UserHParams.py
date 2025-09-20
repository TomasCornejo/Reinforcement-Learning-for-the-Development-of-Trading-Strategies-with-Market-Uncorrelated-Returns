from DDPGExp import trainDDPG
import numpy as np
import os

OPTIMALHPS =    {'learning_rate': 4.677239778446064e-05, 
                 'batch_size': 459, 
                 'gamma': 0.9404787744956136, 
                 'tau': 0.06929880014028883, 
                 'gradient_steps': 1, 
                 'train_freq': 1}

VAR_INITIAL_BALANCE = 100000
VAR_EPISODE_STEP_CONST = 64 
VAR_N_TOTAL_EPISODES = 900
VAR_TESTING_POINTS = 50
VAR_ID_PATH = '000003'
VAR_DESCRIPTION = 'Base con return memory size=3 + netprofit'
VAR_DATA_PATH = '../Data/enriched_data'
VAR_TRIALS = 100
VAR_STUDY_NAME = 'OptunaDDPGExpBase_RMemorySize3_NetProfit'
VAR_PATH_TO_SAVE = f'OptimizeIterationsResults/{VAR_STUDY_NAME}/{VAR_ID_PATH}'

# CONTEXT VARIABLES USING
CTX_REWARD_TYPE = 'netprofit' #  'sharpewithcorrelation', 'netprofit'
CTX_SHUFFLE_DAYS = False
CTX_CORRELATION_PUNISHMENT = 1
CTX_RETURN_MEMORY_SIZE = 3

# CONTEXT VARIABLES BASE
# CTX_REWARD_TYPE = 'sharpewithcorrelation'    # 'sharpewithcorrelation', 'netprofit'
# CTX_SHUFFLE_DAYS = False
# CTX_CORRELATION_PUNISHMENT = 1
# CTX_RETURN_MEMORY_SIZE = 5



def main():
    for i in range(VAR_TRIALS):
        #Generate a numpy random seed for agent
        VAR_SEED_SHUFFLE = np.random.randint(0, 1000) 
        VAR_SEED_AGENT = np.random.randint(0, 1000)

        VAR_ID_PATH = '00000' + str(i)
        VAR_PATH_TO_SAVE = f'OptimizeIterationsResults/{VAR_STUDY_NAME}/{VAR_ID_PATH}'

        #¿check if the path exists?
        if not os.path.exists(VAR_PATH_TO_SAVE):
            os.makedirs(VAR_PATH_TO_SAVE)

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
            HP_PARAMS=OPTIMALHPS,
            SAVE_DATA=True
        )
        print("Trial: ", i , "Result: ", result) 


if __name__ == "__main__":
    main()