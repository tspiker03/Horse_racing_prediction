import numpy as np
import pandas as pd
from datetime import datetime
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
from sklearn.metrics import (confusion_matrix, accuracy_score)
import scipy.stats as stats
import pickle

#pathname = input("Enter pathname for dataset:")

pathname = '/Users/susanwhite/PycharmProjects/Horse Racing/192021seaons.csv'

# read the data, store it in dataframe, get list of column names
df = pd.read_csv(str(pathname), low_memory=False)
df['place'] = pd.to_numeric(df['place'], errors='coerce')
df['draw'] = pd.to_numeric(df['draw'], errors='coerce')
df = df.dropna(subset=['place', 'draw'])



# life_win_percentage is the lifetime win percentage of the horse.  Number of wins divided by number or races.
def life_win_percentage(horse_id):
    """returns the lifetime winning percentage for a horse. """
    horse_df = df[df['horseid'] == horse_id]
    if 1 in list(horse_df['place']):
        percent = horse_df['place'].value_counts()[1]
        return percent
    else:
        return 0


def average_speed_rating(horse_id):
    """returns the average speed rating of the horse matching the given ID over the past 4 horses"""
    horse_df = df[df['horseid'] == horse_id]
    temp_df = horse_df[0:4]
    temp_df.sort_values(by='raceid', ascending=False)
    return temp_df['rating'].mean()


def sire_strike_rate(horse_id, racedate):
    """win percentage by offspring of the horse’s sire (father) prior to this race"""
    horse_df = df[df['horseid'] == horse_id]
    sire = list(horse_df['sire'])[0]
    if len(horse_df['horseid']) == 0:
        return 0
    else:
        sire_df = df[(df['racedate'] < racedate) & (df['sire'] == sire)]
        if 1 in list(sire_df['place']):
            return sire_df['place'].value_counts(normalize=True)[1]
        else:
            return 0


def jockey_strike_rate(horse_id, racedate):
    """Win percentage of jockey prior to that race"""
    horse_df = df[df['horseid'] == horse_id]
    jockey = list(horse_df['jockey'])[0]
    if len(horse_df['horseid']) == 0:
        return 0
    else:
        jockey_df = df[(df['racedate'] < racedate) & (df['jockey'] == jockey)]
        if 1 in list(jockey_df['place']):
            return jockey_df['place'].value_counts(normalize=True)[1]
        else:
            return 0


def trainer_strike_rate(horse_id, racedate):
    """Win percentage of trainer prior to that race"""
    horse_df = df[df['horseid'] == horse_id]
    trainer = list(horse_df['trainer'])[0]
    if len(horse_df['horseid']) == 0:
        return 0
    else:
        trainer_df = df[(df['racedate'] < racedate) & (df['trainer'] == trainer)]
        if 1 in list(trainer_df['place']):
            return trainer_df['place'].value_counts(normalize=True)[1]
        else:
            return 0


def draw_position(horse_id):
    """returns the post position for that particular race"""
    horse_df = df[df['horseid'] == horse_id]
    draw = list(horse_df['draw'])[0]
    if type(list(horse_df['draw'])[0]) == float:
        return draw
    else:
        return 0


def daysLTO(horse_id, racedate):
    """calculates the number of dats since last race"""
    horse_df = df[df['horseid'] == horse_id]
    if len(list(horse_df['racedate'])) <= 1:
        return 0
    date = list(horse_df['racedate'])[1]
    date1 = datetime.strptime(date, '%Y-%m-%d')
    date2 = datetime.strptime(racedate, '%Y-%m-%d')
    return abs((date2 - date1).days)

def place_last_race(horse_id):
    """returns place of horse in last race run"""
    horse_df = df[df['horseid'] == horse_id]
    if len(horse_df) >=2:
        return int(list(horse_df['place'])[1])
    else:
        return 0.0

def place_second_last_race(horse_id):
    """returns place of horse in second to last race run"""
    horse_df = df[df['horseid'] == horse_id]
    if len(horse_df) >=3:
        return int(list(horse_df['place'])[2])
    else:
        return 0.0

def average_rating_last_race(horse_id):
    """returns the average ratiing of the horses in the last race"""
    horse_df = df[df['horseid'] == horse_id]
    race_id = list(horse_df['raceid'])[0]
    last_race_df = df[df['raceid'] == race_id]
    average = float(last_race_df['rating'].mean())
    if type(average) == float:
        return average
    else:
        return 0

def speed_rating(horse_id):
    horse_df = df[df['horseid'] == horse_id]
    speed_rating = float(horse_df['rating'].iloc[0])
    if type(speed_rating) == float:
        return speed_rating
    else:
        return 0



###Body
# This little bit arranges the data into arrays by race.
# race_data is each individual race,
# races is an array of all the races in the training data


df['life_win_percentage'] = df.apply(lambda x:  life_win_percentage(x['horseid']), axis='columns')
df['average_speed_rating'] = df.apply(lambda x: average_speed_rating(x['horseid']), axis='columns')
df['sire_strike_rate'] = df.apply(lambda x: sire_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['jockey_strike_rate'] = df.apply(lambda x: jockey_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['trainer_strike_rate'] = df.apply(lambda x: trainer_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['daysLTO'] = df.apply(lambda x: daysLTO(x['horseid'],x['racedate']), axis='columns')
df['place_last_race'] = df.apply(lambda x: place_last_race(x['horseid']), axis='columns')
df['place_second_last_race'] = df.apply(lambda x: place_second_last_race(x['horseid']), axis='columns')
df['average_rating_last_race'] = df.apply(lambda x: average_rating_last_race(x['horseid']), axis='columns')
df['speed rating'] = df.apply(lambda x: speed_rating(x['horseid']), axis='columns')
df['real_odds'] = df.apply(lambda x: (1/x['winodds'])*100, axis='columns')
df['public_prob'] = df.apply(lambda x: (1/x['winodds']/((x['winodds']+1)/x['winodds'])), axis='columns')



df.dropna(inplace=True, subset=['place', 'draw', 'speed rating',
                                'daysLTO', 'sire_strike_rate', 'place_last_race', 'place_second_last_race',
                                 'jockey_strike_rate', 'life_win_percentage', 'real_odds'])

# define the model data, X and Y:
X = df[['draw', 'speed rating', 'daysLTO', 'sire_strike_rate', 'place_last_race',
        'place_second_last_race', 'jockey_strike_rate', 'life_win_percentage', 'real_odds']]
Xc = sm.add_constant(X)

# fit the model

df['place'] = df.place.astype('category')
Y = df['place']

model = OrderedModel(Y, X, distr='logit', ordered=True).fit()



#load test data
pathname = '/Users/susanwhite/PycharmProjects/Horse Racing/2022season.csv'


df = pd.read_csv(str(pathname), low_memory=False)
df['place'] = pd.to_numeric(df['place'], errors='coerce')
df['draw'] = pd.to_numeric(df['draw'], errors='coerce')
df = df.dropna(subset=['place', 'draw'])
df['life_win_percentage'] = df.apply(lambda x:  life_win_percentage(x['horseid']), axis='columns')
df['average_speed_rating'] = df.apply(lambda x: average_speed_rating(x['horseid']), axis='columns')
df['sire_strike_rate'] = df.apply(lambda x: sire_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['jockey_strike_rate'] = df.apply(lambda x: jockey_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['trainer_strike_rate'] = df.apply(lambda x: trainer_strike_rate(x['horseid'], x['racedate']), axis='columns')
df['daysLTO'] = df.apply(lambda x: daysLTO(x['horseid'],x['racedate']), axis='columns')
df['place_last_race'] = df.apply(lambda x: place_last_race(x['horseid']), axis='columns')
df['place_second_last_race'] = df.apply(lambda x: place_second_last_race(x['horseid']), axis='columns')
df['average_rating_last_race'] = df.apply(lambda x: average_rating_last_race(x['horseid']), axis='columns')
df['speed rating'] = df.apply(lambda x: speed_rating(x['horseid']), axis='columns')
df['real_odds'] = df.apply(lambda x: (1/x['winodds'])*100, axis='columns')
df['public_prob'] = df.apply(lambda x: (1/x['winodds']/((x['winodds']+1)/x['winodds'])), axis='columns')

df.dropna(inplace=True, subset=['place', 'draw', 'speed rating', 'average_speed_rating', 'real_odds',
                                'daysLTO', 'sire_strike_rate', 'place_last_race', 'place_second_last_race',
                                'average_rating_last_race', 'jockey_strike_rate', 'trainer_strike_rate',
                                'life_win_percentage'])

race_ids = df['raceid'].unique().tolist()
for id in race_ids:
    race_df = df[df['raceid'] == id]
    X_test = race_df[['draw', 'speed rating', 'daysLTO', 'sire_strike_rate', 'place_last_race',
        'place_second_last_race', 'jockey_strike_rate', 'life_win_percentage', 'real_odds']]
    Y_test = model.predict(X_test)

    results_df = race_df[['raceid', 'horseid', 'racedate', 'winodds', 'place']].copy()
    results_df['winner'] = Y_test[0]
    results_df['public_prob'] = df['public_prob']
    results_df.to_csv('/Users/susanwhite/PycharmProjects/Horse Racing/results.csv', mode='a', index=False)

#create an iterator with write permission

with open('model_pkl', 'wb') as files:
    pickle.dump(model, files)
