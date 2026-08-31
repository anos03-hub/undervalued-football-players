# %%
import pandas as pd
import numpy as np
import os
import sys
import unicodedata
from rapidfuzz import fuzz
import statsmodels.api as sm


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(BASE_DIR)

DATA_PATH = os.path.join(PROJECT_DIR, 'data', 'players_data-2024_2025.csv')
DATA_PATH2 = os.path.join(PROJECT_DIR, 'data', 'players-info.csv')
DATA_PATH3 = os.path.join(PROJECT_DIR, 'data', 'player_valuations.csv')

#Load dataframes and change date column to datetime (data df is fbref with advanced stats and column Players, players_info is transfermakt with names but not values, 
#players_evaluation -> values with players_id and not names)
data = pd.read_csv(DATA_PATH)
players_info = pd.read_csv(DATA_PATH2)
players_evaluation = pd.read_csv(DATA_PATH3)
players_evaluation['date'] = pd.to_datetime(players_evaluation['date'])



# %%
newest_value = players_evaluation[(players_evaluation['date']<='2025-06-30') & ( players_evaluation['date'] >= '2024-07-01')].sort_values('date').groupby('player_id').last().reset_index()[['player_id', 'date', 'market_value_in_eur']]
print(newest_value.head(10))
print(len(newest_value))
print(newest_value['player_id'].duplicated().sum())
print(newest_value[newest_value['player_id'] == 206050])

print(newest_value.info())              


# check how many players are both in data df and 
mask = data['Player'].isin(players_info['name'])
print(f'There are {mask.sum()} matches out of {len(mask)} ({mask.mean():.1%})')

#check the players that didnt match
print(data[~mask][['Player', 'Squad']].head(10))
# print(players_info[(players_info['name'].str.contains('Adri')) & (players_info['current_club_name'].str.contains('Betis'))])
# print(players_info[(players_info['name'].str.contains('Abe')) & (players_info['current_club_name'].str.contains('Osa'))])
# print(players_info[(players_info['name'].str.contains('Akda'))])
# %%
#function to normalize names so that the matching can be improved (normalize a name, lowercase and strip on both columns and creating a new column in both df)
def normalize_name(s):
    s = unicodedata.normalize('NFKD', s).encode('ascii', 'ignore').decode()  # πέτα διακριτικά
    s = s.lower()    # πεζά
    s = s.strip()    # καθάρισε άκρα
    return s
players_info['name_clean'] = players_info['name'].apply(normalize_name)
data['player_clean'] = data['Player'].apply(normalize_name)

mask_clean = data['player_clean'].isin(players_info['name_clean'])
print(f'There are {mask_clean.sum()} matches out of {len(mask_clean)} ({mask_clean.mean():.1%})')

data_masked = data[~mask_clean]
players_info['date_of_birth'] = pd.to_datetime(players_info['date_of_birth']) #convert date of birth columnd to datetime type
players_info['birth_year'] = players_info['date_of_birth'].dt.year #create new column year in players_info for every player
print(players_info['birth_year'])
from rapidfuzz import process, fuzz
name = 'adrian'
born = 1987

candidates = players_info[players_info['birth_year'] == born]
print(len(candidates))
results = process.extract(name, candidates['name_clean'], scorer=fuzz.token_set_ratio, limit=3)
print(results)

# %%
fuzzy_matches = []          # λίστα από dicts
unmatched = []              # για τον απολογισμό

from rapidfuzz.fuzz import token_set_ratio      
for idx, row in data[~mask_clean].iterrows():
    candidates = players_info[players_info['birth_year'] == row['Born']] #filter to keep only the players born in my year
    if len(candidates) == 0:
        unmatched.append({'name': row['player_clean'], 'reason': 'no_candidates'})
        continue
    top = process.extract(row['player_clean'], candidates['name_clean'], scorer = token_set_ratio, limit = 2)
    best_name, best_score, best_idx = top[0]
    if best_score < 80:
        unmatched.append({'name': row['player_clean'], 'reason': 'low_score'})
        continue
    elif len(top)>1 and top[1][1] == best_score:
        unmatched.append({'name': row['player_clean'], 'reason': 'ambiguous'})
        continue
    else:
        fuzzy_matches.append({
        'fbref_name': row['player_clean'],
        'player_id': players_info.loc[best_idx, 'player_id'],
        'score': best_score})

print(len(fuzzy_matches))
unmatched_df = pd.DataFrame(unmatched)
print(unmatched_df['reason'].value_counts())
print(unmatched_df[unmatched_df['reason']=='low_score']['name'].tolist())

lost = data[~mask_clean].merge(pd.DataFrame(unmatched), left_on='player_clean', right_on='name')
serious_losses = lost[lost['Min'] >= 900][['Player', 'Squad', 'Min', 'reason']]
print(serious_losses)

# %%
manual_matches = {
    'abel': 658115,        # Abel Bretones, Osasuna
    'adrian': 71271,       # Adrián San Miguel, Betis
    'max kilman': 503749,  # Maximilian Kilman, West Ham
    'michael amir murillo' : 354482,
    'julian chabot' : 303219,
    'alex grimaldo': 193082,
    'juan herzog' : 1146029,
    'alejandro jimenez': 741257,
    'kialonda' : 836441,
    'marvin' : 603940,
    'miguel' : 480883,
    "obite n'dicka": 194545,
}

manual_matches_df = pd.Series(manual_matches).reset_index().rename(columns={0: 'player_id' , 'index': 'fbref_name'})
print(manual_matches_df)

print(set(manual_matches_df['fbref_name']) - set(data[~mask_clean]['player_clean']))
# print(players_info.head(5))
# print(data[~mask_clean].head(5))


# %%
fuzzy_matches_df = pd.DataFrame(fuzzy_matches)

df_exact = pd.merge(data[mask_clean], players_info, left_on='player_clean', right_on='name_clean')
##print(df_exact)

df_name_id = df_exact[['name_clean' , 'player_id']].rename(columns={'name_clean':'fbref_name'})
print(df_name_id)

df_name_id['fbref_name'].duplicated().sum()



# %%
df_mergeYear = pd.merge(data[mask_clean], players_info , left_on = ['player_clean', 'Born'] , right_on = ['name_clean', 'birth_year']).rename(columns={'name_clean':'fbref_name'})


id_counts = df_mergeYear.groupby('fbref_name')['player_id'].nunique()
ambiguous_names = id_counts[id_counts>1].index
exact_df = df_mergeYear[~df_mergeYear['fbref_name'].isin(ambiguous_names)]
exact_df = exact_df.drop_duplicates(subset='fbref_name')

print(exact_df['fbref_name'].duplicated().sum())
exact_df = exact_df[['fbref_name', 'player_id']]
fuzzy_matches_df = fuzzy_matches_df[['fbref_name', 'player_id']]
players_df = pd.concat([exact_df, fuzzy_matches_df, manual_matches_df], ignore_index=True)

players_df = players_df[players_df['fbref_name'] != 'nicolas gonzalez']
players_df = players_df.drop_duplicates(subset='fbref_name')
players_df.head
players_df.info
players_df.dtypes

#%%
#Merging the different dataframes to one named model_df
model_df = pd.merge(data, players_df, left_on='player_clean', right_on='fbref_name')
model_df = pd.merge(model_df, newest_value, on='player_id')
# %%
#dropping duplicates, aka players that transfered in january to another team
model_df = model_df.sort_values('Min', ascending=False).drop_duplicates(subset='player_id', keep='first')
print(model_df.head(5))


# %%
PROJECT_DIR = os.getcwd()

OUTPUT_PATH = os.path.join(
    PROJECT_DIR,
    "data",
    "processed",
    "model_data.parquet"
)

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)

model_df.to_parquet(OUTPUT_PATH, index=False)
# %%
