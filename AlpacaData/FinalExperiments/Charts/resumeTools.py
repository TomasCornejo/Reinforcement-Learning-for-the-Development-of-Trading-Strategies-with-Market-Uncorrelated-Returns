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

def last_data_list(data_list_files):
    data_list = []
    for file in data_list_files:
        with open(file, 'rb') as f:
            data_list.append(pickle.load(f))

    resultData = {}
    agents = list(data_list[0].keys())
    metrics = list(data_list[0][agents[0]].keys())
    metrics.remove('shares_held')
    metrics.remove('dates')

    # First pass to create structure
    for agent in agents:    
        resultData[agent] = {}
        for metric in metrics:
            resultData[agent][metric] = []

    # Add the las value of each metric
    for data in data_list:
        for agent in agents:
            for metric in metrics:
                resultData[agent][metric].append(data[agent][metric][-1])
    
    return resultData

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

def getResumeDataTestingDDPGAgent(experiments_paths, agent_name=None):
    resultDictionary = {}
    if agent_name:
        agent_to_use = agent_name
    else:
        agent_to_use = 'DDPG Agent'

    for expPath in experiments_paths:
        expName = expPath.split("OptunaDDPGExpBase_")[-1]
        resultDictionary[expName+agent_to_use] = {}
        
        PATH_DATA_LIST_METRICS_RESUME_TESTING  = glob(expPath + "/00000*/*/metrics_resume_testing.pkl")
        metrics_resume_avg = mean_resume_data(PATH_DATA_LIST_METRICS_RESUME_TESTING)
        
        for k in metrics_resume_avg[agent_to_use].keys():
            if k != 'dates':
                resultDictionary[expName+agent_to_use][k] = {}
                resultDictionary[expName+agent_to_use][k] = np.mean(metrics_resume_avg[agent_to_use][k])
    return resultDictionary

def getResumeDataTrainingDDPGAgent(experiments_paths):
   resultDictionary = {}
   for expPath in experiments_paths:
      expName = expPath.split("OptunaDDPGExpBase_")[-1]
      resultDictionary[expName] = {}
      
      PATH_DATA_LIST_METRICS_RESUME_TRAINING  = glob(expPath + "/00000*/*/metrics_resume.pkl")
      metrics_resume_avg = mean_resume_data(PATH_DATA_LIST_METRICS_RESUME_TRAINING)
      
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

def getLastAgentsNetworth(path_files, metrica='net_worth_delta_acum'):
    #key = folder.split('\\OptunaDDPGExpBase_')[-1]
    metrics_resume_testing_avg = last_data_list(path_files)
    metricsDict = {}
    for agent in metrics_resume_testing_avg.keys():
        metricsDict[agent] = metrics_resume_testing_avg[agent][metrica]
    return metricsDict