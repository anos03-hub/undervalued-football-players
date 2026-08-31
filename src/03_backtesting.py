#%%
import pandas as pd
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "ols_gb_model.parquet"
)

df = pd.read_parquet(DATA_PATH)
# %%
DATA_PATH1 = os.path.join(PROJECT_DIR, 'data', 'player_valuations.csv')
valuations = pd.read_csv(DATA_PATH1)
images_dir = os.path.join(PROJECT_DIR, 'data', 'images')
os.makedirs(images_dir, exist_ok=True)
# %%
# Convert valuation dates to datetime format
valuations['date'] = pd.to_datetime(valuations['date'])
# %%
# Check the available valuation date range
print(valuations['date'].min(), valuations['date'].max())

# %%
# Define the 6-month backtest target date and calculate each valuation's distance from it
target_date = pd.Timestamp('2025-12-31')
valuations['days_from_target'] = (
    valuations['date'] - target_date
).abs().dt.days
# %%
# Keep valuation records only for players included in the final modelling dataset
future_vals = valuations[
    valuations['player_id'].isin(df['player_id'])
].copy()
# %%
# Select the valuation closest to the target date for each player
future_vals = future_vals.sort_values('days_from_target')
future_vals = future_vals.drop_duplicates(
    subset = 'player_id',
    keep='first'
)

#Prepare the 6-month backtest dataset by selecting each player's closest future 
#valuation within a ±45-day window and calculating the subsequent market value change
# %%
future_vals['days_from_target'].describe()
# %%
future_vals = future_vals[
    future_vals['days_from_target'] <= 45
].copy()
# %%
future_vals = future_vals.rename(columns={
    'market_value_in_eur': 'future_value_6m',
    'date': 'future_date_6m'
})
# %%
backtest_df = pd.merge(
    df,
    future_vals[['player_id', 'future_date_6m', 'future_value_6m', 'days_from_target']],
    on='player_id',
    how='inner'
)
# %%
backtest_df['value_change_6m'] = (
    backtest_df['future_value_6m'] /
    backtest_df['market_value_in_eur'] - 1
)
# %%
backtest_df[
    ['Player', 'market_value_in_eur', 'future_value_6m', 'value_change_6m']
].head()



# %%
## Compare the 6-month market value outcomes of the 100 most undervalued and 100 most overvalued players identified by the OLS valuation gap
top100_ols = backtest_df.sort_values(
    'valuation_gap_pct',
    ascending = False
).head(100)
# %%
print(top100_ols['value_change_6m'].median())
print((top100_ols['value_change_6m'] > 0).mean())
# %%
# %%
top100_overvalued_ols = backtest_df.sort_values(
    'valuation_gap_pct',
    ascending = True
).head(100)
# %%
print(top100_overvalued_ols['value_change_6m'].median())
print((top100_overvalued_ols['value_change_6m'] > 0).mean())


# Compare 6-month market value outcomes across OLS and Gradient Boosting
# valuation-gap quintiles using median change and positive-change rates

# OLS quintiles
backtest_df['ols_quintile'] = pd.qcut(
    backtest_df['valuation_gap_pct'],
    q=5,
    labels=[
        'Most overvalued',
        'Q2',
        'Middle',
        'Q4',
        'Most undervalued'
    ]
)

print(
    backtest_df.groupby('ols_quintile')['value_change_6m'].median()
)

ols_positive_rate = (
    backtest_df.groupby('ols_quintile', observed=False)['value_change_6m']
    .apply(lambda x: (x > 0).mean())
)

print(ols_positive_rate)


# Gradient Boosting quintiles
backtest_df['gb_quintile'] = pd.qcut(
    backtest_df['gb_valuation_gap_pct'],
    q=5,
    labels=[
        'Most overvalued',
        'Q2',
        'Middle',
        'Q4',
        'Most undervalued'
    ]
)

print(
    backtest_df.groupby('gb_quintile', observed = False)['value_change_6m'].median()
)

gb_positive_rate = (
    backtest_df.groupby('gb_quintile')['value_change_6m']
    .apply(lambda x: (x > 0).mean())
)

print(gb_positive_rate)
# %%
# Test whether higher OLS valuation gaps are associated with stronger 6-month market value changes
from scipy.stats import spearmanr

rho, p_value = spearmanr(
    backtest_df['valuation_gap_pct'],
    backtest_df['value_change_6m']
)

print("Spearman rho:", rho)
print("p-value:", p_value)
# %%
from scipy.stats import mannwhitneyu
# Compare 6-month value changes between the most undervalued and most overvalued OLS quintiles,
#  and summarize median outcomes across all quintiles
most_undervalued = backtest_df[
    backtest_df['ols_quintile'] == 'Most undervalued'
]['value_change_6m']

most_overvalued = backtest_df[
    backtest_df['ols_quintile'] == 'Most overvalued'
]['value_change_6m']

stat, p_value = mannwhitneyu(
    most_undervalued,
    most_overvalued,
    alternative='greater'
)

print("U statistic:", stat)
print("p-value:", p_value)
# %%
# Calculate the median 6-month market value change for each OLS valuation-gap quintile
quintile_median = (
    backtest_df.groupby('ols_quintile', observed= False)['value_change_6m']
    .median()
    .reset_index()
)
print(quintile_median)
# %%
import matplotlib.pyplot as plt
# Calculate the share of players with a positive 6-month value change in each OLS quintile

plt.figure(figsize=(8,5))

plt.bar(
    quintile_median['ols_quintile'],
    quintile_median['value_change_6m'] * 100
)

plt.ylabel('Median 6-month value change (%)')
plt.xlabel('OLS valuation quintile')
plt.title('6-Month Market Value change by OLS Valuation Quintile')

plt.xticks(rotation = 20)
plt.tight_layout()
plt.savefig(
    os.path.join(images_dir, 'backtest_median_change.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
# %%
positive_rate = (
    backtest_df
    .groupby('ols_quintile', observed=False)['value_change_6m']
    .apply(lambda x: (x>0).mean())
    .reset_index(name='positive_rate')
)
print(positive_rate)
# %%
plt.figure(figsize=(8,5))

plt.bar(
    positive_rate['ols_quintile'],
    positive_rate['positive_rate'] * 100
)

plt.ylabel('Players with positive 6-month value change (%)')
plt.xlabel('OLS valuation quintile')
plt.title('Share of Players Increasing in Value by OLS Valuation Quintile')

plt.xticks(rotation = 20)
plt.tight_layout()
plt.savefig(
    os.path.join(images_dir, 'backtest_positive_rate.png'),
    dpi=300,
    bbox_inches='tight'
)
plt.show()
# %%
