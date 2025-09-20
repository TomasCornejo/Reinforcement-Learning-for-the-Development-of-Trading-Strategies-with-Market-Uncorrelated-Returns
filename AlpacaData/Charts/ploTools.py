import matplotlib.pyplot as plt
import pandas as pd

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
    ax.set_title('Training Rewards Over Steps - ' + agent_name)
    ax.set_xlabel('Step')
    ax.set_ylabel('Reward')
    plt.grid(axis='y')
    # Add a legend
    #ax.legend()

    if path_to_save:
        fig.savefig(path_to_save + agent_name + '_training_rewards.png')
    plt.show()

def plotTrainingRewardsMultiAgents(agents_training_rewards_resume, path_to_save = None):
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


def plotSharpeRatioMultiAgents(metrics, path_to_save = None):
    plt.figure(figsize=(15, 5))
    for agent_name, agent_metrics in metrics.items():
        plt.plot(agent_metrics['dates'], agent_metrics['sharpe_ratios'], label=agent_name)
    plt.xticks(rotation=45)
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

def plotCumulativeRewardMultiAgents(metrics, path_to_save = None):
    plt.figure(figsize=(15, 5))
    for agent_name, agent_metrics in metrics.items():
        plt.plot(['Start'] + agent_metrics['dates'], agent_metrics['rewards_acum'], label=agent_name)
    plt.xticks(rotation=45)
    plt.title('Cumulative Reward Per Episode')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'cumulative_rewards.png')
    plt.show()

def plotCumulativeNetWorthMultiAgents(metrics, path_to_save = None):
    plt.figure(figsize=(15, 5))
    for agent_name, agent_metrics in metrics.items():
        plt.plot(['Start'] + agent_metrics['dates'], agent_metrics['net_worths_delta_acum'], label=agent_name)
    plt.xticks(rotation=45)
    plt.title('Cumulative Net Worth Per Episode')
    plt.xlabel('Date')
    plt.ylabel('Cumulative Net Worth')
    plt.legend()
    plt.grid(axis='y')

    if path_to_save:
        plt.savefig(path_to_save + 'cumulative_net_worth.png')
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
