
# %%
#IMPORTING PACKAGES - IMPORTING FILES
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict

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
#COUNTING PLAYERS BY POSITION
df['Pos_main'] = df['Pos'].str.split(',').str[0]

print(df['Pos_main'].value_counts())

# %%
#REMOVING GOALKEEPERS, CREATING DIFFERENT DF FOR EACH POSITION(DEF, MID, FORWARDS)

df = df[df['Pos_main'] != 'GK']
df_def = df[df['Pos_main'] == 'DF'].copy()
df_mid = df[df['Pos_main'] == 'MF'].copy()
df_fw = df[df['Pos_main'] == 'FW'].copy()

print(len(df_def), len(df_mid), len(df_fw))



# %%

#KEEPING ONLY PLAYERS WITH OVER 10 GAMES FOR THE SEASON 2024-2025
df_def = df_def[df_def['90s'] > 10].copy()
df_mid = df_mid[df_mid['90s'] > 10].copy()
df_fw = df_fw[df_fw['90s'] > 10].copy()
for frame in [df_def, df_mid, df_fw]:
    frame['log_90s'] = np.log1p(frame['90s'])
print(len(df_def), len(df_mid), len(df_fw))


# %%
for frame in [df_def, df_mid, df_fw]:
    for col in ['Tkl+Int', 'PrgP', 'PrgR', 'Carries', 'KP', 'xAG', 'npxG']:
        frame[col + '_90'] = frame[col] / frame['90s']

# %%
#CREATING NEW COLUMNS (CLR_90, RECOVERY_90, ATT_PEN_90 AND ALSO AGE COLUMN)
for frame in [df_def, df_mid, df_fw]:
    frame['Age2'] = frame['Age'] ** 2

df_def['Clr_90'] = df_def['Clr'] / df_def['90s']
df_mid['Recov_90'] = df_mid['Recov'] / df_mid['90s']
df_fw['Att_Pen_90'] = df_fw['Att Pen'] / df_fw['90s']

for frame in [df_def, df_mid, df_fw]:
    frame['Age_c'] = frame['Age'] - frame['Age'].mean()
    frame['Age_c2'] = frame['Age_c'] ** 2

#%%
# FINAL FEATURE SETS FOR EACH POSITION
def_features = [
    'Age_c',
    'Age_c2',
    'log_90s',
    'Tkl+Int_90',
    'Tkl%',
    'Won%',
    'PrgP_90',
    'PrgR_90',
    'Carries_90',
    'KP_90',
    'Clr_90',
    'Cmp%'
]
mid_features = [
    'Age_c', 'Age_c2', 'log_90s',
    'xAG_90', 'KP_90', 'PrgP_90',
    'PrgR_90', 'Carries_90',
    'Tkl+Int_90', 'npxG_90','Cmp%', 'Recov_90'
]

fw_features = [
    'Age_c', 'Age_c2', 'log_90s',
    'npxG_90', 'npxG/Sh', 'xAG_90',
    'KP_90', 'PrgR_90', 'PrgP_90','SCA90',
    'Att_Pen_90'
]

# %%
#CHECKING FOR NULL AND 0s IN DATA.
print(df_def[def_features].isna().mean().sort_values(ascending=False))
print(df_mid[mid_features].isna().mean().sort_values(ascending=False))
print(df_fw[fw_features].isna().mean().sort_values(ascending=False))



#BUILDING x_def, x_mid, x_fw

# %%
import statsmodels.api as sm
# Build the defender feature matrix and log-transformed target,
# add competition dummy variables, and include the intercept for OLS.
X_def = df_def[def_features].copy()
y_def = np.log(df_def['market_value_in_eur'])

comp_def = pd.get_dummies(df_def['Comp'], drop_first=True)

X_def = pd.concat([X_def, comp_def], axis=1)
X_def = X_def.astype(float)

X_def = sm.add_constant(X_def)

# %%
# Build the midfielder feature matrix and log-transformed target,
# add competition dummy variables, and include the intercept for OLS.
X_mid = df_mid[mid_features].copy()
y_mid = np.log(df_mid['market_value_in_eur'])

comp_mid = pd.get_dummies(df_mid['Comp'], drop_first=True)

X_mid = pd.concat([X_mid, comp_mid], axis=1).astype(float)
X_mid = sm.add_constant(X_mid)

model_mid = sm.OLS(y_mid, X_mid).fit()

print(model_mid.summary())


# %%
# Build the Forward feature matrix and log-transformed target,
# add competition dummy variables, and include the intercept for OLS.

X_fw = df_fw[fw_features].copy()
y_fw = np.log(df_fw['market_value_in_eur'])

comp_fw = pd.get_dummies(df_fw['Comp'], drop_first=True)

X_fw = pd.concat([X_fw, comp_fw], axis=1).astype(float)
X_fw = sm.add_constant(X_fw)

model_fw = sm.OLS(y_fw, X_fw).fit()

print(model_fw.summary())

#OLS models (def,mid,fw)
# %%
#Fit the defender model and getting the summary

model_def = sm.OLS(y_def, X_def).fit()

print(model_def.summary())

#%%
#Printing the Variance Inflation Factor for defender model
#Checking for Multicollinearity
from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_def = pd.DataFrame()
vif_def['feature'] = X_def.columns
vif_def['VIF'] = [
    variance_inflation_factor(X_def.values, i)
    for i in range(len(X_def.columns))
]

print(vif_def.sort_values('VIF', ascending=False))




# %%
# Build the midfielder feature matrix and log-transformed target,
# add competition dummy variables, and include the intercept for OLS.
X_mid = df_mid[mid_features].copy()
y_mid = np.log(df_mid['market_value_in_eur'])

comp_mid = pd.get_dummies(df_mid['Comp'], drop_first=True)

X_mid = pd.concat([X_mid, comp_mid], axis=1).astype(float)
X_mid = sm.add_constant(X_mid)

model_mid = sm.OLS(y_mid, X_mid).fit()

print(model_mid.summary())


# %%
#Printing the Variance Inflation Factor for Midfielder model
#Checking for Multicollinearity
vif_mid = pd.DataFrame()
vif_mid['feature'] = X_mid.columns
vif_mid['VIF'] = [
    variance_inflation_factor(X_mid.values, i)
    for i in range(len(X_mid.columns))
]

print(vif_mid.sort_values('VIF', ascending=False))




# %%
# Build the Forward feature matrix and log-transformed target,
# add competition dummy variables, and include the intercept for OLS.

X_fw = df_fw[fw_features].copy()
y_fw = np.log(df_fw['market_value_in_eur'])

comp_fw = pd.get_dummies(df_fw['Comp'], drop_first=True)

X_fw = pd.concat([X_fw, comp_fw], axis=1).astype(float)
X_fw = sm.add_constant(X_fw)

model_fw = sm.OLS(y_fw, X_fw).fit()

print(model_fw.summary())

# %%
#Printing the Variance Inflation Factor for Forward model
#Checking for Multicollinearity
vif_fw = pd.DataFrame()
vif_fw['feature'] = X_fw.columns
vif_fw['VIF'] = [
    variance_inflation_factor(X_fw.values, i)
    for i in range(len(X_fw.columns))
]

print(vif_fw.sort_values('VIF', ascending=False))


#5-fold OOF predictions for OLS
# %%
#Define the 5-fold cross-validation with shuffled, reproducible splits
cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

# %%
# Generate out-of-fold predictions for defenders using 5-fold cross-validation

X_def_cv = X_def.drop(columns='const')

df_def['predicted_log_oof'] = cross_val_predict(
    LinearRegression(),
    X_def_cv,
    y_def,
    cv=cv
)

# %%
# Generate out-of-fold predictions for Midfielders using 5-fold cross-validation

X_mid_cv = X_mid.drop(columns='const')

df_mid['predicted_log_oof'] = cross_val_predict(
    LinearRegression(),
    X_mid_cv,
    y_mid,
    cv=cv
)


# %%
# Generate out-of-fold predictions for Forwards using 5-fold cross-validation
X_fw_cv = X_fw.drop(columns='const')

df_fw['predicted_log_oof'] = cross_val_predict(
    LinearRegression(),
    X_fw_cv,
    y_fw,
    cv=cv
)


# %%
# Convert OOF log predictions back to euros and calculate each player's valuation gap
for frame in [df_def, df_mid, df_fw]:
    frame['predicted_value_oof'] = np.exp(frame['predicted_log_oof'])
    frame['valuation_gap_pct'] = (
        frame['predicted_value_oof'] / frame['market_value_in_eur'] - 1
    ) * 100



# %%
all_results = pd.concat(
    [df_def, df_mid, df_fw],
    ignore_index=True
)


# %%
# Identify the top 15 undervalued players aged 28 or younger with a market value of at least €5m
top_undervalued = all_results[
    (all_results['market_value_in_eur'] >= 5_000_000) &
    (all_results['Age'] <= 28)
].sort_values(
    'valuation_gap_pct',
    ascending=False
).head(15)
print(top_undervalued)



# %%
# Evaluate out-of-fold model performance for each position using R² and MAE
from sklearn.metrics import r2_score, mean_absolute_error

for name, frame in [('DF', df_def), ('MF', df_mid), ('FW', df_fw)]:
    r2 = r2_score(
        np.log(frame['market_value_in_eur']),
        frame['predicted_log_oof']
    )

    mae = mean_absolute_error(
        frame['market_value_in_eur'],
        frame['predicted_value_oof']
    ) / 1_000_000

    print(name, 'R2:', round(r2, 3), 'MAE €m:', round(mae, 2))


#%%
# Inspect the distribution of predicted-to-actual value ratios for each position
for name, frame in [('DF', df_def), ('MF', df_mid), ('FW', df_fw)]:
    ratio = frame['predicted_value_oof'] / frame['market_value_in_eur']
    print(name)
    print(ratio.quantile([0.05, 0.25, 0.5, 0.75, 0.95]))
    
    


#%%
# %%
extreme = all_results[
    (all_results['market_value_in_eur'] >= 5_000_000) &
    (all_results['Age'] <= 28)
].copy()

extreme['ratio'] = (
    extreme['predicted_value_oof']
    / extreme['market_value_in_eur']
)

print(
    extreme.sort_values('ratio', ascending=False)[[
        'Player', 'Age', 'Pos_main',
        'market_value_in_eur',
        'predicted_value_oof',
        'ratio'
    ]].head(20)
)


# %%
#Gradient boosting model building
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
gb_model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.05,
    max_depth=2,
    min_samples_leaf=10,
    random_state=42
)


# %%
#cross-validation prediction 
from sklearn.model_selection import cross_val_predict
df_def['gb_predicted_log_oof'] = cross_val_predict(
    gb_model,   #model
    X_def_cv, #features
    y_def,    #target  
    cv=cv     #cross-validation strategy  
)
# %%
from sklearn.metrics import r2_score, mean_absolute_error
#Computing r^2 scores and MAE with gradient boosting for Defenders
gb_r2_def = r2_score(
    y_def,
    df_def['gb_predicted_log_oof']
)

# Convert defender OOF log predictions back to euros and calculate MAE

gb_predicted_value_def = np.exp(
    df_def['gb_predicted_log_oof']
)

gb_mae_def = mean_absolute_error(
    df_def['market_value_in_eur'],
    gb_predicted_value_def
) / 1_000_000

print("GB Defender R2", round(gb_r2_def, 3))
print("GB Defenders MAE €m:", round(gb_mae_def, 2))


# Inspect extreme predicted-to-actual value ratios for defenders
ratio_gb = gb_predicted_value_def / df_def['market_value_in_eur']

print(ratio_gb.quantile([0.95, 0.99]))

# %%
# Generate out-of-fold Gradient Boosting predictions for midfielders
df_mid['gb_predicted_log_oof'] = cross_val_predict(
    gb_model,
    X_mid_cv,
    y_mid,
    cv=cv
)
#%%
# Generate out-of-fold Gradient Boosting predictions for forwards
df_fw['gb_predicted_log_oof'] = cross_val_predict(
    gb_model,
    X_fw_cv,
    y_fw,
    cv=cv
)
# %%
# Evaluate Gradient Boosting performance for midfielders using OOF R² and MAE
gb_r2_mid = r2_score(
    y_mid,
    df_mid['gb_predicted_log_oof']
)

gb_predicted_value_mid = np.exp(
    df_mid['gb_predicted_log_oof']
)

gb_mae_mid = mean_absolute_error(
    df_mid['market_value_in_eur'],
    gb_predicted_value_mid
) / 1_000_000

print("GB midfielder R2", round(gb_r2_mid, 3))
print("GB midfielder MAE €m:", round(gb_mae_mid, 2))
# %%
# Evaluate Gradient Boosting performance for Forwards using OOF R² and MAE
gb_r2_fw = r2_score(
    y_fw,
    df_fw['gb_predicted_log_oof']
)

gb_predicted_value_fw = np.exp(
    df_fw['gb_predicted_log_oof']
)

gb_mae_fw = mean_absolute_error(
    df_fw['market_value_in_eur'],
    gb_predicted_value_fw
) / 1_000_000

print("GB forwards R2", round(gb_r2_fw, 3))
print("GB forwards MAE €m:", round(gb_mae_fw, 2))

# %%
# Convert Gradient Boosting OOF log predictions to euros and calculate valuation gaps
for frame in [df_def, df_mid, df_fw]:
    frame['gb_predicted_value_oof'] = np.exp(frame['gb_predicted_log_oof'])

    frame['gb_valuation_gap_pct'] = (
        frame['gb_predicted_value_oof'] /
        frame['market_value_in_eur'] -1
    ) *100

# Combine Gradient Boosting results across all outfield positions

gb_results = pd.concat(
    [df_def, df_mid, df_fw],
    ignore_index=True
)
# %%
#Print undervalued players with Gradient Boosting Results, younger than 28yrs old and market value over 5M.
print(
    gb_results[
        (gb_results['market_value_in_eur'] >= 5_000_000) &
        (gb_results['Age'] <= 28)
    ][
        ['Player', 'Nation', 'Pos', 'Squad', 'Age', 'market_value_in_eur', 'Comp', 'gb_valuation_gap_pct']
    ].sort_values(
        by='gb_valuation_gap_pct',
        ascending=False
    ).head(15)
)



# %%
param_grid = {
    'n_estimators': [50, 100, 150],
    'learning_rate': [0.03, 0.05, 0.1],
    'max_depth': [1, 2, 3],
    'min_samples_leaf': [5, 10, 20]
}

grid_def = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)

grid_def.fit(X_def_cv, y_def)
print(grid_def.best_params_)
print(grid_def.best_score_)

best_gb_def = grid_def.best_estimator_
# %%
df_def['gb_tuned_predicted_log_oof'] = cross_val_predict(
    best_gb_def,
    X_def_cv,
    y_def,
    cv=cv
)
# %%
gb_tuned_r2_def = r2_score(
    y_def,
    df_def['gb_tuned_predicted_log_oof']
)

gb_tuned_pred_value_def = np.exp(
    df_def['gb_tuned_predicted_log_oof']
)

gb_tuned_mae_def = mean_absolute_error(
    df_def['market_value_in_eur'],
    gb_tuned_pred_value_def
) / 1_000_000

print("Tuned GB Def R2:", round(gb_tuned_r2_def, 3))
print("Tuned GB Def MAE €m:", round(gb_tuned_mae_def, 2))
# %%
grid_mid = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
grid_mid.fit(X_mid_cv, y_mid)
print(grid_mid.best_params_)
print(grid_mid.best_score_)
# %%

best_gb_mid = grid_mid.best_estimator_

df_mid['gb_tuned_predicted_log_oof'] = cross_val_predict(
    best_gb_mid,
    X_mid_cv,
    y_mid,
    cv=cv
)
# %%
gb_tuned_r2_mid = r2_score(
    y_mid,
    df_mid['gb_tuned_predicted_log_oof']
)

gb_tuned_pred_value_mid = np.exp(
    df_mid['gb_tuned_predicted_log_oof']
)

gb_tuned_mae_mid = mean_absolute_error(
    df_mid['market_value_in_eur'],
    gb_tuned_pred_value_mid
) / 1_000_000

print("Tuned GB mid R2:", round(gb_tuned_r2_mid, 3))
print("Tuned GB mid MAE €m:", round(gb_tuned_mae_mid, 2))
# %%
grid_fw = GridSearchCV(
    estimator=GradientBoostingRegressor(random_state=42),
    param_grid=param_grid,
    scoring='r2',
    cv=cv,
    n_jobs=-1
)
grid_fw.fit(X_fw_cv, y_fw)
print(grid_fw.best_params_)
print(grid_fw.best_score_)
# %%

best_gb_fw = grid_fw.best_estimator_

df_fw['gb_tuned_predicted_log_oof'] = cross_val_predict(
    best_gb_fw,
    X_fw_cv,
    y_fw,
    cv=cv
)
# %%
gb_tuned_r2_fw = r2_score(
    y_fw,
    df_fw['gb_tuned_predicted_log_oof']
)

gb_tuned_pred_value_fw = np.exp(
    df_fw['gb_tuned_predicted_log_oof']
)

gb_tuned_mae_fw = mean_absolute_error(
    df_fw['market_value_in_eur'],
    gb_tuned_pred_value_fw
) / 1_000_000

print("Tuned GB fw R2:", round(gb_tuned_r2_fw, 3))
print("Tuned GB fw MAE €m:", round(gb_tuned_mae_fw, 2))
# %%
# %%
# Compare OLS, baseline Gradient Boosting and tuned Gradient Boosting performance

model_comparison = pd.DataFrame({
    'Position': ['DF', 'MF', 'FW'],

    'OLS_R2': [
        r2_score(y_def, df_def['predicted_log_oof']),
        r2_score(y_mid, df_mid['predicted_log_oof']),
        r2_score(y_fw, df_fw['predicted_log_oof'])
    ],

    'OLS_MAE_m': [
        mean_absolute_error(
            df_def['market_value_in_eur'],
            df_def['predicted_value_oof']
        ) / 1_000_000,

        mean_absolute_error(
            df_mid['market_value_in_eur'],
            df_mid['predicted_value_oof']
        ) / 1_000_000,

        mean_absolute_error(
            df_fw['market_value_in_eur'],
            df_fw['predicted_value_oof']
        ) / 1_000_000
    ],

    'GB_R2': [
        gb_r2_def,
        gb_r2_mid,
        gb_r2_fw
    ],

    'GB_MAE_m': [
        gb_mae_def,
        gb_mae_mid,
        gb_mae_fw
    ],

    'Tuned_GB_R2': [
        gb_tuned_r2_def,
        gb_tuned_r2_mid,
        gb_tuned_r2_fw
    ],

    'Tuned_GB_MAE_m': [
        gb_tuned_mae_def,
        gb_tuned_mae_mid,
        gb_tuned_mae_fw
    ]
})

print(model_comparison.round(3))




# %%
# Define output paths and save the combined OLS/Gradient Boosting modelling results

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "ols_gb_model.parquet"
)

print(PROJECT_DIR)
print(OUTPUT_PATH)
gb_results.to_parquet(OUTPUT_PATH, index=False)
# Save the final top-undervalued shortlist as a separate processed output

top_undervalued.to_parquet(
    os.path.join(PROJECT_DIR, "data", "processed", "top_undervalued.parquet"),
    index=False
)