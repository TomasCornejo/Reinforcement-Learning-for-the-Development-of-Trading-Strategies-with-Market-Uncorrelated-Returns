# Reinforcement Learning for the Development of Trading Strategies with Market-Uncorrelated Returns

## Abstract

A reinforcement learning approach is proposed and evaluated for managing capital in the U.S. stock market, with a specific focus on minimizing correlation with market portfolio returns. The approach aims to reduce the losses during black swan market conditions. The best-performing agent outperformed the intraday buy-and-hold strategy in terms of both Sharpe ratio (13% better), net worth (13.8% better) and just with 0.15 of Pearson correlation with the market index. This agent was further tested in two historical black swan scenarios (with 7% and 8% of index value loss, both months outside the train and test data), achieving a 35% and 41.1% reduction in losses. The results establish the framework to propose new agents uncorrelated with index values and other strategies or portfolio returns.

**Index Terms:** Deep reinforcement learning, Q-learning, Neural Networks, Automated stock trading

## Project Overview

This repository contains a comprehensive implementation of reinforcement learning algorithms for automated stock trading, specifically designed to create market-uncorrelated trading strategies. The project focuses on developing agents that can generate returns independent of market movements, particularly during volatile market conditions.

## Stock Universe

The research utilizes a diversified portfolio of 21 U.S. stocks across multiple sectors:

| Ticker | Company | Sector | Industry |
|--------|---------|--------|----------|
| AAPL | Apple Inc. | Technology | Consumer Electronics |
| ABCB | Ameris Bancorp | Financials | Regional Banking |
| CSCO | Cisco Systems, Inc. | Technology | Telecommunications Equipment |
| DAL | Delta Air Lines, Inc. | Transportation | Airlines |
| EAST | Eastside Distilling Inc | Consumer | Alcoholic Beverages |
| ESPR | Esperion Therapeutics Inc. | Healthcare | Biotechnology |
| F | Ford Motor Co. | Consumer | Automobiles & Components |
| GOOGL | Alphabet Inc. | Technology | Internet Services |
| JPM | JPMorgan Chase & Co. | Financials | Multinational Banking |
| META | Meta Platforms Inc. | Technology | Social Networks |
| MSFT | Microsoft Corporation | Technology | Software & Services |
| NVDA | NVIDIA Corporation | Technology | Semiconductors |
| PFE | Pfizer Inc. | Healthcare | Pharmaceuticals |
| PG | Procter & Gamble Co. | Consumer | Consumer Products |
| PSX | Phillips 66 | Energy | Oil & Gas |
| SGLY | Singularity Future Technology Ltd | Technology | Telecommunications Services |
| TSLA | Tesla Inc. | Consumer | Automobiles & Components |
| UNH | UnitedHealth Group Inc. | Healthcare | Health Services |
| USA | Liberty All-Star Equity Fund | Financials | Investment Funds |
| WMT | Walmart Inc. | Consumer | Retail |
| XOM | ExxonMobil Corporation | Energy | Oil & Gas |

**Benchmark:** SP500 Index

## Project Structure

### 1. Data Extraction (`1.DataExtraction/`)
- **`getStockData.ipynb`**: Data collection from Alpaca API
- **`hist_data_function.py`**: Historical data retrieval functions

### 2. Preprocessing (`2.Preprocessing/`)
- **`Preprocessing_Setup.ipynb`**: Data cleaning, technical indicators calculation, and train/test splits

### 3. Experiments (`3.Experiments/`)

#### 3.1 Experiment Assets (`ExperimentAssets/`)
- **`AgentsToUse.py`**: RL agent implementations (PPO, A2C, DDPG, Random, SP500)
- **`CustomCallback.py`**: Training callbacks for metrics collection
- **`RLMain.py`**: Main training pipeline
- **`StockTradingEnv.py`**: Custom Gymnasium trading environment

#### 3.2 Initial Experiments (`InitialExperiments/`)
- **`Initial_Comparison_between_RL_Algos.ipynb`**: Baseline comparison of RL algorithms
- **`Advanced_Daily_Episode_RL_Trading_Pipeline.ipynb`**: Daily episode structure implementation
- **`Advanced_Sharpe_Ratio_RL_Trading_Experiment.ipynb`**: Sharpe ratio reward function experiments
- **`Advanced_Sharpe_Ratio_Uncorrelated_Trading_Environment.ipynb`**: Market-uncorrelated trading environment
- **`Batch_Reproducibility_Experiments_Sharpe_Correlation.ipynb`**: Statistical validation experiments
- **`Comprehensive_RL_Trading_Experiment.ipynb`**: Full experimental pipeline
- **`Experimental_Results_Visualization_Pipeline.ipynb`**: Results visualization
- **`rewardVisualTest.ipynb`**: Reward function testing

#### 3.3 Final Experiments (`FinalExperiments/`)
- **`DDPGExp.py`**: DDPG-specific experiments
- **`OptunaDDPGExp.py`**: Hyperparameter optimization using Optuna
- **`StatisticalTest.ipynb`**: Statistical significance testing
- **`TestMejorAgenteEnDrawdowns.ipynb`**: Black swan scenario testing
- **`GetStudyData.ipynb`**: Data aggregation for analysis
- **`UserHParams.py`**: User-defined hyperparameters

### 4. Results Extraction (`4.ResultsExtraction/`)

#### 4.1 Data Collection
- **`AAA_GetTrainingData.ipynb`**: Training data aggregation
- **`AAA_GetTrainingMetrics.ipynb`**: Training metrics collection
- **`AAAA_GetTrainTestComparison.ipynb`**: Train/test performance comparison
- **`AAAAAAAA_GetHPparams.ipynb`**: Hyperparameter analysis

#### 4.2 Results Analysis
- **`ZZZ_RES1_GraficosTablasValoresBase.ipynb`**: Base model results
- **`ZZZ_RES2_GraficosTablasPivotTipoRecompensa.ipynb`**: Reward type analysis
- **`ZZZ_RES3_GraficosTablasPivotDiasAleatorizados.ipynb`**: Shuffled days analysis
- **`ZZZ_RES4_GraficosTablasPivotCastigoPorCorrelacion.ipynb`**: Correlation punishment analysis
- **`ZZZ_RES5_GraficosTablasPivotTamanoVentanaRetorno.ipynb`**: Return window size analysis
- **`ZZZ_RES6_GráficosComparativosGenerales.ipynb`**: General comparative analysis

#### 4.3 Statistical Analysis
- **`ZZZZ_meanSTD_Exp*.ipynb`**: Mean and standard deviation analysis for different experiments
- **`ZZZZ_p_value_*.ipynb`**: P-value calculations and statistical significance testing

#### 4.4 Visualization Tools (`Charts/`)
- **`ploTools.py`**: Plotting utilities
- **`resumeTools.py`**: Results summarization tools

#### 4.5 Paper Assets (`dataChartsForPaper/`)
- **`LineChart/`**: Line chart generation for research paper
- **`ScatterChart/`**: Scatter plot generation for research paper

## Key Features

### Trading Environment
- **Custom Gymnasium Environment**: Multi-stock trading with continuous action space
- **Daily Episode Structure**: Episodes end at day boundaries for realistic trading
- **Technical Indicators**: RSI, MACD, CCI, ADX for enhanced decision making
- **Correlation Tracking**: Real-time Pearson correlation with market benchmark
- **Transaction Costs**: Realistic trading costs included in reward calculation

### Reward Functions
- **Sharpe Ratio Reward**: Risk-adjusted returns as primary objective
- **Correlation Penalty**: Penalizes correlation with market benchmark
- **Net Profit Reward**: Alternative reward function for comparison
- **Return Memory**: Rolling window of returns for correlation calculation

### RL Algorithms
- **PPO (Proximal Policy Optimization)**: Policy gradient method
- **A2C (Advantage Actor-Critic)**: Actor-critic method
- **DDPG (Deep Deterministic Policy Gradient)**: Continuous control method
- **Baseline Agents**: Random trading and SP500 buy-and-hold strategies

### Experimental Design
- **Systematic Parameter Variation**: Correlation punishment, reward types, memory sizes
- **Statistical Validation**: Multiple runs with different random seeds
- **Black Swan Testing**: Performance during market crashes
- **Reproducibility**: Consistent experimental setup across all runs

## Results Summary

The research demonstrates that reinforcement learning agents can be trained to:
- **Outperform buy-and-hold strategies** by 13.8% in net worth
- **Achieve superior Sharpe ratios** (13% better than benchmark)
- **Maintain low correlation** with market index (0.15 Pearson correlation)
- **Reduce losses during market crashes** by 35-41% in black swan scenarios

## Usage

1. **Data Collection**: Run notebooks in `1.DataExtraction/` to collect stock data
2. **Preprocessing**: Execute `2.Preprocessing/Preprocessing_Setup.ipynb` to prepare data
3. **Experiments**: Use notebooks in `3.Experiments/` to run different experimental configurations
4. **Analysis**: Analyze results using notebooks in `4.ResultsExtraction/`

## Dependencies

- Python 3.8+
- PyTorch
- Stable-Baselines3
- Gymnasium
- Pandas
- NumPy
- Matplotlib
- Scipy
- Optuna (for hyperparameter optimization)

## License

This project is licensed under the terms specified in the LICENSE file.

## Citation

If you use this work in your research, please cite:

```
[Your citation format here]
```

## Contact

For questions or collaboration, please contact [your contact information].
