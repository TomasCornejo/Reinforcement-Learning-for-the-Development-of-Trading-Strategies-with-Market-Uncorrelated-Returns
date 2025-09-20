import matplotlib.pyplot as plt
import pandas as pd
from .resumeTools import getLastAgentsNetworth

def plotTrainingRewards(agent_training_rewards_resume, agent_name , path_to_save = None):
    dataToPlot = []
    step_index_adder = 0
    cumulative_reward = 0
    for period_index, rewards in enumerate(agent_training_rewards_resume):
        for step_index, reward in enumerate(rewards):
            cumulative_reward += reward
            dataToPlot.append([step_index+step_index_adder, cumulative_reward, f'Period {period_index + 1}'])
        step_index_adder += len(rewards)

    df = pd.DataFrame(dataToPlot, columns=['Step', 'Reward', 'Training Period'])

    fig, ax = plt.subplots()

    # Plot each training period with a different color
    for period in df['Training Period'].unique():
        period_data = df[df['Training Period'] == period]
        ax.plot(period_data['Step'], period_data['Reward'], label=period)
    # Add title and labels
    ax.set_title('Serie de Recompensas de Entrenamiento - Agente DDPG')
    ax.set_xlabel('Paso', fontsize=14)
    ax.set_ylabel('Recompensa', fontsize=14)
    # Set the axis values to font size 12
    ax.tick_params(axis='both', which='major', labelsize=12)
    plt.grid(axis='y')
    # Add a legend
    #ax.legend()

    if path_to_save:
        fig.savefig(path_to_save + "base_agent" + '_training_rewards.png')
    plt.show()

def plotTrainingRewardsMultiAgents(agents_training_rewards_resume, path_to_save = None, label_list_dict = None):
    for agent in agents_training_rewards_resume:
        
        if len(agents_training_rewards_resume[agent]) > 0:
            plotTrainingRewards(agents_training_rewards_resume[agent], agent, path_to_save)

def plotPearsonCorrelationMultiAgents(metrics, path_to_save = None):
    plt.figure(figsize=(15, 5))
    for agent_name, agent_metrics in metrics.items():
        plt.plot(agent_metrics['dates'], agent_metrics['pearson_correlations'], label=agent_name)
    plt.xticks(rotation=45)
    plt.title('Pearson Correlations Per Episode')
    plt.xlabel('Date')
    plt.ylabel('Pearson Correlations')
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'pearson_correlations.png')
    plt.show()


def plotSharpeRatioMultiAgents(metrics, path_to_save = None, label_list_dict = None):
    plt.figure(figsize=(15, 5))
    for agent_name, agent_metrics in metrics.items():
        if label_list_dict:
            agent_name = label_list_dict[agent_name]
        plt.plot(agent_metrics['dates'], agent_metrics['sharpe_ratios'], label=agent_name)
    plt.xticks(rotation=45)
    if label_list_dict:
        plt.title(label_list_dict['title'], fontsize=14)
        plt.xlabel(label_list_dict['xlabel'], fontsize=14)
        plt.ylabel(label_list_dict['ylabel'], fontsize=14)
    else:
        plt.title('Sharpe Ratios Per Episode')
        plt.xlabel('Date')
        plt.ylabel('Sharpe Ratios')
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'sharpe_ratios.png')
    plt.show()

def plotTableMeanSharpeRatioMultiAgents(metrics, path_to_save = None):
    #Plot a table data with the sharpe ratios
    data = {}
    #Calculate the mean sharpe ratio for each agent
    for agent_name, agent_metrics in metrics.items():
        data[agent_name] = round(sum(agent_metrics['sharpe_ratios']) / len(agent_metrics['sharpe_ratios']), 2)
    
    df = pd.DataFrame(data.items(), columns=['Agent', 'Mean Sharpe Ratio'])
    #Order the data by the mean sharpe ratio desc
    df = df.sort_values(by='Mean Sharpe Ratio', ascending=False)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    ax.set_title('Mean Sharpe Ratios Per Agent')

    if path_to_save:
        fig.savefig(path_to_save + 'mean_sharpe_ratios.png')
    plt.show()

def plotTableMeanAndStandardDeviationSharpeRatioMultiAgents(metrics, path_to_save = None):
    #Plot a table data with the sharpe ratios
    data = {}
    #Calculate the mean and standard deviation for each agent
    for agent_name, agent_metrics in metrics.items():
        mean = round(sum(agent_metrics['sharpe_ratios']) / len(agent_metrics['sharpe_ratios']), 2)
        std = round(pd.Series(agent_metrics['sharpe_ratios']).std(), 2)
        data[agent_name] = (mean, std)
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Mean Sharpe Ratio', 'Standard Deviation Sharpe Ratio'])
    #Order the data by the mean sharpe ratio desc
    df = df.sort_values(by='Mean Sharpe Ratio', ascending=False)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    # Plot a table with the data with the index as the first column called 'Agent'
    df.index.name = 'Agent'
    df.reset_index(inplace=True)

    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    ax.set_title('Sharpe Ratios Per Agent')

    if path_to_save:
        fig.savefig(path_to_save + 'mean_std_sharpe_ratios.png')
    plt.show()

def plotTableFullMetricMultiAgents(metrics, metricToPlot , path_to_save = None):
    #Plot a table data with the metric
    data = {}
    #Calculate the mean ,standard deviation and confidence Inverval 95 (2 columns, 1 for lower and 1 for upper) for each agent
    for agent_name, agent_metrics in metrics.items():
        mean = round(sum(agent_metrics[metricToPlot]) / len(agent_metrics[metricToPlot]), 2)
        std = round(pd.Series(agent_metrics[metricToPlot]).std(), 2)
        ci = round(1.96 * (std / (len(agent_metrics[metricToPlot]) ** 0.5)), 2)
        data[agent_name] = (mean, std, mean - ci, mean + ci)
    
    df = pd.DataFrame.from_dict(data, orient='index', columns=['Mean', 'STD', 'Lower CI', 'Upper CI'])
    #Order the data by the mean metric desc
    df = df.sort_values(by=f'Mean', ascending=False)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    # Plot a table with the data with the index as the first column called 'Agent'
    df.index.name = 'Agent'
    df.reset_index(inplace=True)

    table = ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    ax.set_title(f'{metricToPlot} Per Agent')

    #Enlarge the font size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    if path_to_save:
        fig.savefig(path_to_save + f'fullmetrics_{metricToPlot}.png')
    plt.show()

## Per PATH!!!!
def plotTableFullLastWorthMultiAgents(path_files, path_to_save = None, metric ='net_worths_delta_acum', show = True, agent_name = None):
    net_worths_metrics = getLastAgentsNetworth(path_files, metric)
    df_net_worth = pd.DataFrame.from_dict(net_worths_metrics, orient='index').T
    # Calculate the mean and standard deviation of the net worths and put them in a new DataFrame
    df_mean_net_worth = df_net_worth.mean(axis=0)#.apply(lambda x: '{:.2f}'.format(x))
    df_std_net_worth = df_net_worth.std(axis=0)#.apply(lambda x: '{:.2f}'.format(x))
    df_len = len(df_net_worth)
    df_ci_net_worth = df_net_worth.std(axis=0).apply(lambda x: round(1.96 * (x / (df_len ** 0.5)), 2))
    df_resume_net_worth = pd.DataFrame({'mean': df_mean_net_worth.apply(lambda x: '{:.2f}'.format(x)), 
                                        'std': df_std_net_worth.apply(lambda x: '{:.2f}'.format(x)), 
                                        'Lower CI': (df_mean_net_worth - df_ci_net_worth).apply(lambda x: '{:.2f}'.format(x)), 
                                        'Upper CI': (df_mean_net_worth + df_ci_net_worth).apply(lambda x: '{:.2f}'.format(x))})
    print('net_worths')
    df_resume_net_worth

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis('off')
    # Plot a table with the data with the index as the first column called 'Agent'
    df_resume_net_worth.index.name = 'Agent'
    df_resume_net_worth.reset_index(inplace=True)

    table = ax.table(cellText=df_resume_net_worth.values, colLabels=df_resume_net_worth.columns, cellLoc='center', loc='center')
    ax.set_title(f'Last NetWorth Per Agent')

    #Enlarge the font size of the table
    table.auto_set_font_size(False)
    table.set_fontsize(8)

    if path_to_save:
        fig.savefig(path_to_save + f'lastNetWorth_perAgent.png')
    if show:
        plt.show()
    if agent_name is not None:
        return df_mean_net_worth[agent_name], df_std_net_worth[agent_name]
    return df_mean_net_worth , df_std_net_worth


def plotTableMeanPearsonCorrelationMultiAgents(metrics, path_to_save = None):
    #Plot a table data with the pearson correlations
    data = {}
    #Calculate the mean pearson correlation for each agent
    for agent_name, agent_metrics in metrics.items():
        data[agent_name] = round(sum(agent_metrics['pearson_correlations']) / len(agent_metrics['pearson_correlations']), 2)
    
    df = pd.DataFrame(data.items(), columns=['Agent', 'Mean Pearson Correlation'])
    #Order the data by the mean pearson correlation desc
    df = df.sort_values(by='Mean Pearson Correlation', ascending=False)

    fig, ax = plt.subplots(figsize=(5, 2))
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, cellLoc='center', loc='center')
    ax.set_title('Mean Pearson Correlations Per Agent')

    if path_to_save:
        fig.savefig(path_to_save + 'mean_pearson_correlations.png')
    plt.show()

def plotCumulativeRewardMultiAgents(metrics, path_to_save = None, label_list_dict = None):
    plt.figure(figsize=(15, 8))
    for agent_name, agent_metrics in metrics.items():
        if label_list_dict:
            agent_name = label_list_dict[agent_name]
        plt.plot(['Start'] + agent_metrics['dates'], agent_metrics['rewards_acum'], label=agent_name)
    #plt.xticks(rotation=45)
    plt.xticks([])

    if label_list_dict:
        plt.title(label_list_dict['title'], fontsize=14)
        plt.xlabel(label_list_dict['xlabel'], fontsize=14)
        plt.ylabel(label_list_dict['ylabel'], fontsize=14)
    else:
        plt.title('Cumulative Reward Per Test Episode')
        plt.xlabel('Test Episode', fontsize=14)
        plt.ylabel('Cumulative Reward', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12)
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'cumulative_rewards.png')
    plt.show()

def plotCumulativeNetWorthMultiAgents(metrics, path_to_save = None, label_list_dict = None):
    plt.figure(figsize=(15, 8))
    dataOutput = {}
    for agent_name, agent_metrics in metrics.items():
        if label_list_dict:
            agent_name = label_list_dict[agent_name]
        dataOutput[agent_name] = {}
        dataOutput[agent_name]['dates'] = agent_metrics['dates']
        dataOutput[agent_name]['patAcum'] = agent_metrics['net_worths_delta_acum']
        plt.plot(['Start'] + agent_metrics['dates'], agent_metrics['net_worths_delta_acum'], label=agent_name)
    #plt.xticks(rotation=45)
    plt.xticks([])

    if label_list_dict:
        plt.title(label_list_dict['title'], fontsize=14)
        plt.xlabel(label_list_dict['xlabel'], fontsize=14)
        plt.ylabel(label_list_dict['ylabel'], fontsize=14)
    else:
        plt.title('Cumulative Net Worth Per Test Episode', fontsize=14)
        plt.xlabel('Test Episode', fontsize=14)
        plt.ylabel('Cumulative Net Worth', fontsize=14)
    plt.tick_params(axis='y', which='major', labelsize=12)
    # increase the legend font size
    plt.legend(fontsize=12)
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'cumulative_net_worth.png')
    plt.show()
    return dataOutput

def endPlotNetWorthMultiAgents(metrics, path_to_save = None):
    plt.figure(figsize=(15, 8))
    for agent_name, agent_metrics in metrics.items():
        plt.plot(agent_metrics['dates'], agent_metrics['net_worths'], label=agent_name)
    plt.xticks(rotation=45)
    plt.title('Net Worth Per Episode')
    plt.xlabel('Date')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'cumulative_net_worth.png')
    plt.show()

#Create a scatter plot with net_worths and person_correlation
def plotScatterAandB(X , Y , finalResults, path_to_save = None, label_list_dict = None):
    fig, ax = plt.subplots()
    for expName in finalResults.keys():
        tag = expName
        if label_list_dict:
              tag = label_list_dict[expName]      
        ax.scatter(finalResults[expName][X], finalResults[expName][Y], label=tag)

    xForLabel = X.replace("_", " ").capitalize()
    yForLabel = Y.replace("_", " ").capitalize()
    if label_list_dict:
        ax.set_title(label_list_dict['title'], fontsize=14)
        ax.set_xlabel(label_list_dict['xlabel'], fontsize=14)
        ax.set_ylabel(label_list_dict['ylabel'], fontsize=14)
    else:
        ax.set_xlabel(xForLabel)
        ax.set_ylabel(yForLabel)
    ax.legend()
    if path_to_save:
        plt.savefig(path_to_save + 'plot_Scatter.png')
    plt.show()

# EXTRAS
def plotSharesHeldPlot(agents_metrics):
    agents_to_plot = list(agents_metrics.keys())
    for i in range(len(agents_to_plot)):
        #Create the dataframe for shares_held where each row is a ticker , each column is a step and the value is the shares held
        df = pd.DataFrame.from_dict(agents_metrics[agents_to_plot[i]]['shares_held']).T
        #Filter the rows where the sum of the shares held is greater than 0
        df = df.loc[(df.sum(axis=1) > 0)]
        #Now plot a line char where each line is a ticker and the x axis is the step
        df.T.plot(figsize=(12, 6))
        plt.title(f'Shares Held Over Time for {agents_to_plot[i]}')
        plt.xlabel('Episodes')
        plt.ylabel('Shares Held')
        plt.show()
