# %%
import pandas as pd
import os
import numpy as np


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "model_data.parquet"
)

df = pd.read_parquet(DATA_PATH)

print(df.shape)
# %%
# %%
# Split outfield players by primary position and apply the same minimum-minutes filter used in the main models

df['Pos_main'] = df['Pos'].str.split(',').str[0]

df = df[df['Pos_main'] != 'GK']

df_def = df[df['Pos_main'] == 'DF'].copy()
df_mid = df[df['Pos_main'] == 'MF'].copy()
df_fw = df[df['Pos_main'] == 'FW'].copy()

df_def = df_def[df_def['90s'] > 10].copy()
df_mid = df_mid[df_mid['90s'] > 10].copy()
df_fw = df_fw[df_fw['90s'] > 10].copy()

print(len(df_def), len(df_mid), len(df_fw))

# %%
# Convert counting statistics to per-90 rates so players with different playing time can be compared fairly

counting_stats = [
    'TklW', 'Int', 'Clr', 'Blocks_stats_defense', 'Recov',
    'PrgP', 'PrgC', 'PrgR', 'KP', 'PPA', '1/3',
    'SCA', 'npxG', 'xAG', 'Sh', 'SoT',
    'Att Pen', 'CPA', 'Mis', 'Dis'
]

for frame in [df_def, df_mid, df_fw]:
    for col in counting_stats:
        frame[col + '_90'] = frame[col] / frame['90s']

# %%
# Define position-specific candidate feature pools for the feature selection experiment
def_candidates = [
    'TklW_90',
    'Int_90',
    'Clr_90',
    'Blocks_stats_defense_90',
    'Recov_90',
    'Won%',
    'Tkl%',
    'Cmp%',
    'PrgP_90',
    'PrgC_90',
    'PrgR_90',
    'KP_90',
    'Mis_90',
    'Dis_90'
]

mid_candidates = [
    'PrgP_90',
    'PrgC_90',
    'PrgR_90',
    'KP_90',
    'PPA_90',
    '1/3_90',
    'SCA_90',
    'xAG_90',
    'npxG_90',
    'Recov_90',
    'TklW_90',
    'Int_90',
    'Cmp%',
    'Mis_90',
    'Dis_90'
]

fw_candidates = [
    'npxG_90',
    'npxG/Sh',
    'Sh_90',
    'SoT_90',
    'xAG_90',
    'KP_90',
    'SCA_90',
    'PrgC_90',
    'PrgR_90',
    'PPA_90',
    'Att Pen_90',
    'CPA_90',
    'Succ%',
    'Mis_90',
    'Dis_90'
]
# %%
# %%
# Create the same age and playing-time control variables used in the main positional models


for frame in [df_def, df_mid, df_fw]:
    frame['Age_c'] = frame['Age'] - frame['Age'].mean()
    frame['Age_c2'] = frame['Age_c'] ** 2
    frame['log_90s'] = np.log1p(frame['90s'])

# %%
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

# %%
# Define 5-fold cross-validation with shuffled, reproducible splits

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)
# %%
# Prepare defender candidate features and the log-transformed market value target

X_def_candidates = df_def[def_candidates].copy()
# %%
y_def = np.log(df_def['market_value_in_eur'])
# %%
# Use linear regression as the estimator for feature selection

model_def = LinearRegression()
# %%
# Select the eight defender features that produce the strongest cross-validated R²

sfs_def = SequentialFeatureSelector(
    model_def,
    n_features_to_select=8,
    direction='forward',
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
# %%
# Fit the forward sequential feature selector on defender candidate features

sfs_def.fit(X_def_candidates, y_def)
# %%
sfs_def.fit(X_def_candidates, y_def)
# %%
selected_def_features = X_def_candidates.columns[sfs_def.get_support()]
print(selected_def_features.tolist())
# %%
# Build the evaluation dataset using the selected football features,
# together with age, playing-time and competition controls
X_def_selected = df_def[list(selected_def_features)].copy()

controls = df_def[['Age_c', 'Age_c2', 'log_90s']].copy()
comp_def = pd.get_dummies(df_def['Comp'], drop_first=True)

X_def_selected = pd.concat(
    [controls, X_def_selected, comp_def],
    axis=1
).astype(float)
# %%
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error

pred_def_selected = cross_val_predict(
    LinearRegression(),
    X_def_selected,
    y_def,
    cv=cv
)

r2_def_selected = r2_score(
    y_def,
    pred_def_selected
)

pred_value_def_selected = np.exp(pred_def_selected)

mae_def_selected = mean_absolute_error(
    df_def['market_value_in_eur'],
    pred_value_def_selected
) / 1_000_000

print("Selected DF R2:", round(r2_def_selected, 3))
print("Selected DF MAE €m:", round(mae_def_selected, 2))
# %%
