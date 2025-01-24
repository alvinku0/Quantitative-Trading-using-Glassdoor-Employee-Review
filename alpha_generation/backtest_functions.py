import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

def run_backtest(return_xnd, factor_xnd):
    portfolio_return = (return_xnd * factor_xnd.values).sum(axis = 1)
    print("Overall Annual Sharpe Ratio: " + str(np.sqrt(12) * portfolio_return.mean() / portfolio_return.std()))
    print("--------------")
    print("Yearly Sharpe Ratio")
    print(np.sqrt(12) * portfolio_return.groupby(portfolio_return.index.year).mean() / portfolio_return.groupby(portfolio_return.index.year).std())
    print("--------------")
    print("Yearly Return (%)")
    print(((portfolio_return+1).groupby(portfolio_return.index.year).prod() -1)*100)
    return portfolio_return

def smooth_weights(factor_xnd, window):
    return factor_xnd.rolling(window).mean()

def plot_backtest(portfolio_return):
    (portfolio_return+1).cumprod().plot()
    plt.title("Cumulative Return of strategy")
    plt.show()

def plot_weights(factor_xnd):
    factor_xnd.abs().max(axis = 1).plot(label = "Max weight")
    factor_xnd_binary = factor_xnd.isna().astype(int)
    plt.legend()
    # Replace 0s with 1s and 1s with 0s
    factor_xnd_binary = 1 - factor_xnd_binary
    factor_xnd_binary.sum(axis = 1).plot(secondary_y=True, label = "Number of stocks")
    plt.legend()
    plt.title("Max weight vs Number of stocks in portfolio")
    plt.show()

def neutralise(factor_xnd):
    scaled_factor_xnd = factor_xnd.sub(factor_xnd.mean(axis=1), axis=0)
    scaled_factor_xnd = scaled_factor_xnd.div(factor_xnd.abs().sum(axis=1), axis=0) # neutralise
    return scaled_factor_xnd

def portfolio_bins(factor_xnd, lower_threshold, upper_threshold):
    factor_xnd[(factor_xnd < upper_threshold) & (factor_xnd > lower_threshold)] = np.NaN
    return factor_xnd

def neutralise_by_group(factor_xnd, group_level):
    # revert piviot table format
    df = factor_xnd.reset_index().melt(id_vars='year_month', var_name='cusip', value_name='score')

    # temporary add 'year' column
    df['fyear'] = df['year_month'].dt.year

    # lookup GICS
    df_GICS = pd.read_parquet('C:/Users/phku0/Quant_Project/data_processing/mapping_cusip_GICS.parquet')
    df_GICS = df_GICS[['cusip', 'fyear', group_level]]
    df_merged = df.merge(df_GICS, on=['cusip', 'fyear'], how='left')

    # remove irrelevent column
    df_merged = df_merged[['year_month', 'cusip', 'score', group_level]]

    # calculate sector_mean_score
    df_sector_mean = df_merged.groupby(['year_month', group_level])['score'].mean().reset_index()
    df_sector_mean = df_sector_mean.rename(columns={'score': 'sector_mean_score'})
    df = df_merged.merge(df_sector_mean, on=['year_month', group_level], how='left')

    # calculate excess score relative to the sector (score - sector_mean_score)
    df['sector_excess_score'] = np.where(df['score'].notna() & df['sector_mean_score'].notna(),
                                        df['score'] - df['sector_mean_score'],
                                        np.nan)
    # abs of sector_excess_score
    df['abs_sector_excess_score'] = df['sector_excess_score'].abs()

    # calculate abs_market_sum
    df_abs_market_sum = df.groupby(['year_month'])['abs_sector_excess_score'].sum().reset_index().rename(columns={'abs_sector_excess_score': 'abs_market_sum_score'})
    df = df.merge(df_abs_market_sum, on=['year_month'], how='left')

    # neutralised_score = excess return of a stock in a month / absolute sum of market excess score in a month
    df['neutralised_score'] = df['sector_excess_score'] / df['abs_market_sum_score']

    new_factor_xnd = df[['year_month','cusip','neutralised_score']].pivot(index = 'year_month', columns = 'cusip', values = 'neutralised_score')
    return new_factor_xnd
