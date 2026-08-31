# %%

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the final modelling results for scouting analysis and visualization

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

RESULTS_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "ols_gb_model.parquet"
)

df = pd.read_parquet(RESULTS_PATH)

print(df.shape)
print(df.columns.tolist())
# %%
# Display the top 20 undervalued players based on the final OLS ranking
final_ols_shortlist = (
    df[
        (df['market_value_in_eur'] >= 5_000_000) &
        (df['Age'] <= 28)
    ]
    .sort_values('valuation_gap_pct', ascending=False)
)
print(
    final_ols_shortlist[
        [
            'Rk',
            'Player',
            'Nation',
            'Pos',
            'Squad',
            'Comp',
            'Age',
            'predicted_value_oof',
            'market_value_in_eur',
            'valuation_gap_pct'
        ]
    ].head(20)
)
#%%
top_undervalued_def = final_ols_shortlist[
    final_ols_shortlist['Pos'] == 'DF'
]

print(
    top_undervalued_def[
        [
            'Rk',
            'Player',
            'Nation',
            'Pos',
            'Squad',
            'Comp',
            'Age',
            'predicted_value_oof',
            'market_value_in_eur',
            'valuation_gap_pct'
        ]
    ].head(10)
)
#%%
top_undervalued_mid = final_ols_shortlist[
    final_ols_shortlist['Pos'] == 'MF'
]

print(
    top_undervalued_mid[
        [           
            'Player',
            'Nation',
            'Pos_main',
            'Squad',
            'Comp',
            'Age',
            'predicted_value_oof',
            'market_value_in_eur',
            'valuation_gap_pct'
        ]
    ].head(10)
)

#%%
top_undervalued_fw = final_ols_shortlist[
    final_ols_shortlist['Pos'] == 'FW'
]

print(
    top_undervalued_fw[
        [
            'Player',
            'Nation',
            'Pos_main',
            'Squad',
            'Comp',
            'Age',
            'predicted_value_oof',
            'market_value_in_eur',
            'valuation_gap_pct'
        ]
    ].head(10)
)
# %%
final_ols_overvalued = (
    df[
        (df['market_value_in_eur'] >= 5_000_000) &
        (df['Age'] <= 28)
    ]
    .sort_values('valuation_gap_pct', ascending=True)
)

print(
    final_ols_overvalued[
    [
        'Player',
        'Nation',
        'Pos_main',
        'Squad',
        'Comp',
        'Age',
        'market_value_in_eur',
        'predicted_value_oof',
        'valuation_gap_pct'
    ]
    ].head(20)
)
# %%
# Plot the top 15 undervalued players based on the final OLS valuation gap

top15_plot = final_ols_shortlist.head(15).sort_values(
    'valuation_gap_pct',
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top15_plot['Player'],
    top15_plot['valuation_gap_pct']
)

plt.xlabel('Valuation gap (%)')
plt.ylabel('Player')
plt.title('Top 15 Undervalued Players - OLS Model')

plt.tight_layout()
plt.show()
# %%
print(
    top15_plot[
        [
            'Player',
            'Squad',
            'Pos_main',
            'Age',
            'market_value_in_eur',
            'predicted_value_oof',
            'valuation_gap_pct'
        ]
    ]
)
# %%
# Compare actual market values with model-implied OLS values for the top undervalued players

top15_values = final_ols_shortlist.head(15).copy()

top15_values['market_value_m'] = (
    top15_values['market_value_in_eur'] / 1_000_000
)

top15_values['predicted_value_m'] = (
    top15_values['predicted_value_oof'] / 1_000_000
)

top15_values = top15_values.sort_values(
    'predicted_value_m',
    ascending=True
)

plt.figure(figsize=(10, 7))

plt.barh(
    top15_values['Player'],
    top15_values['market_value_m'],
    label='Market value'
)

plt.barh(
    top15_values['Player'],
    top15_values['predicted_value_m'],
    alpha=0.6,
    label='Model-implied value'
)

plt.xlabel('Value (€m)')
plt.ylabel('Player')
plt.title('Market Value vs Model-Implied Value - Top 15 Undervalued Players')

plt.legend()
plt.tight_layout()

images_dir = os.path.join(PROJECT_DIR, 'data', 'images')
os.makedirs(images_dir, exist_ok=True)

plt.savefig(
    os.path.join(images_dir, 'market_vs_model_value.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.show()
plt.show()
# %%
# Compare OLS, baseline Gradient Boosting and tuned Gradient Boosting performance by position

model_comparison = pd.DataFrame({
    'Position': ['DF', 'MF', 'FW'],
    'OLS_R2': [0.713, 0.709, 0.725],
    'GB_R2': [0.675, 0.599, 0.657],
    'Tuned_GB_R2': [0.714, 0.651, 0.716]
})

model_comparison.plot(
    x='Position',
    kind='bar',
    figsize=(8, 5)
)


model_comparison.plot(
    x='Position',
    y=['OLS_R2', 'GB_R2', 'Tuned_GB_R2'],
    kind='bar',
    figsize=(8, 5)
)

plt.ylabel('Out-of-fold R²')
plt.xlabel('Position')
plt.title('Model Performance Comparison by Position')
plt.xticks(rotation=0)

# Model performance comparison
plt.savefig(
    os.path.join(images_dir, 'model_performance_comparison.png'),
    dpi=300,
    bbox_inches='tight'
)

plt.tight_layout()
plt.show()

#%%
# Save the final undervalued and overvalued scouting shortlists as CSV files

OUTPUT_DIR = os.path.join(
    PROJECT_DIR,
    'data',
    'processed'
)

final_ols_shortlist.head(20).to_csv(
    os.path.join(OUTPUT_DIR, 'final_undervalued_shortlist.csv'),
    index=False
)

final_ols_overvalued.head(20).to_csv(
    os.path.join(OUTPUT_DIR, 'final_overvalued_shortlist.csv'),
    index=False
)
# %%
