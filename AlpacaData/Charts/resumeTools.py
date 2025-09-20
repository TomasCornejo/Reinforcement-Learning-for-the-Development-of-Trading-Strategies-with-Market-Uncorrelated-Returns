import pickle
import numpy as np
from glob import glob

def mean_resume_data(data_list_files):
    data_list = []
    for file in data_list_files:
        with open(file, 'rb') as f:
            data_list.append(pickle.load(f))
    full_data = {}
    avg_data = {}
    agents = list(data_list[0].keys())
    metrics = list(data_list[0][agents[0]].keys())
    metrics.remove('shares_held')
    metrics.remove('dates')

    # First pass to create structure
    for agent in agents:    
        full_data[agent] = {}
        avg_data[agent] = {}
        for metric in metrics:
            full_data[agent][metric] = []
    
    # Second pass to gather data
    for data in data_list:
        for agent in agents:
            for metric in metrics:
                full_data[agent][metric].append(data[agent][metric])
            
    # Third pass to compute mean
    for agent in agents:
        for metric in metrics:
            avg_data[agent][metric] = list(np.mean(full_data[agent][metric], axis=0))


    # Add Dates as a number array
    for agent in agents:
        avg_data[agent]['dates'] = [f'T {x+1}' for x in range(len(avg_data[agent]['net_worths_delta']))]
    
    return avg_data

def std_resume_data(data_list_files):
    data_list = []
    for file in data_list_files:
        with open(file, 'rb') as f:
            data_list.append(pickle.load(f))
    full_data = {}
    std_data = {}
    agents = list(data_list[0].keys())
    metrics = list(data_list[0][agents[0]].keys())
    metrics.remove('shares_held')
    metrics.remove('dates')

    # First pass to create structure
    for agent in agents:    
        full_data[agent] = {}
        std_data[agent] = {}
        for metric in metrics:
            full_data[agent][metric] = []
    
    # Second pass to gather data
    for data in data_list:
        for agent in agents:
            for metric in metrics:
                full_data[agent][metric].append(data[agent][metric])
    
    # Third pass to compute std
    for agent in agents:
        for metric in metrics:
            std_data[agent][metric] = list(np.std(full_data[agent][metric], axis=0))
    
     # Add Dates as a number array
    for agent in agents:
        std_data[agent]['dates'] = [f'T {x+1}' for x in range(len(std_data[agent]['net_worths_delta']))]
    
    return std_data


def mean_training_reward_data(data_list_files):
    data_list = []
    for file in data_list_files:
        with open(file, 'rb') as f:
            data_list.append(pickle.load(f))
    full_data = {}
    avg_data = {}
    agents = list(data_list[0].keys())

    # First pass to create structure
    for agent in agents:    
        full_data[agent] = []
    
    # Second pass to gather data
    for data in data_list:
        for agent in agents:
            full_data[agent].append(data[agent])
            
    # Third pass to compute mean
    for agent in agents:
        avg_data[agent] = list(np.mean(full_data[agent], axis=0))
    
    return avg_data

def std_training_reward_data(data_list_files):
    data_list = []
    for file in data_list_files:
        with open(file, 'rb') as f:
            data_list.append(pickle.load(f))
    full_data = {}
    std_data = {}
    agents = list(data_list[0].keys())

    # First pass to create structure
    for agent in agents:    
        full_data[agent] = []
    
    # Second pass to gather data
    for data in data_list:
        for agent in agents:
            full_data[agent].append(data[agent])
    
    # Third pass to compute std
    for agent in agents:
        std_data[agent] = list(np.std(full_data[agent], axis=0))
    
    return std_data


def getResumeDataTestingDDPGAgent(experiments_paths):
   resultDictionary = {}
   for expPath in experiments_paths:
      expName = expPath.split("_")[-1]
      resultDictionary[expName] = {}
      
      PATH_DATA_LIST_METRICS_RESUME_TESTING  = glob(expPath + "/00000*/*/metrics_resume_testing.pkl")
      metrics_resume_avg = mean_resume_data(PATH_DATA_LIST_METRICS_RESUME_TESTING)
      
      for k in metrics_resume_avg['DDPG Agent'].keys():
         if k != 'dates':
            resultDictionary[expName][k] = {}
            resultDictionary[expName][k] = np.mean(metrics_resume_avg['DDPG Agent'][k])
   
   return resultDictionary


def getResumeDataTestingSP500(experiments_paths):
   resultDictionary = {}
   for expPath in experiments_paths:
      expName = expPath.split("_")[-1]
      resultDictionary[expName] = {}
      
      PATH_DATA_LIST_METRICS_RESUME_TESTING  = glob(expPath + "/00000*/*/metrics_resume_testing.pkl")
      metrics_resume_avg = mean_resume_data(PATH_DATA_LIST_METRICS_RESUME_TESTING)
      
      for k in metrics_resume_avg['SP500 Agent'].keys():
         if k != 'dates':
            resultDictionary[expName][k] = {}
            resultDictionary[expName][k] = np.mean(metrics_resume_avg['SP500 Agent'][k])

   return resultDictionary

