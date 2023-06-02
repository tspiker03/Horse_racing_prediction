import numpy as np
import pandas as pd
from sklearn.metrics import (confusion_matrix, accuracy_score)
import scipy.stats as stats
import matplotlib.pyplot as plt

pathname = '/Users/susanwhite/PycharmProjects/Horse Racing/results.csv'
df = pd.read_csv(str(pathname), low_memory=False)

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



def success_rate_model():
    '''this program calculates the success rate of the model and gives the results of each individual race'''
    race_ids = df['raceid'].unique().tolist()
    total_races = len(race_ids)
    success = 0
    for race_id in race_ids:
        race_df = df[df['raceid'] == race_id]
        pick = list(race_df[race_df.winner == race_df.winner.max()]['horseid'])[0]
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        answer = 'wrong'
        if winner == pick:
            success +=1
            answer = 'right'
        print("Race number", race_id, 'you picked', pick, 'to win')
        print('the winner of', race_id, 'was', winner,'.')
        print('you were', answer)
        success_rate = success/total_races
        print('your success rate is:', success_rate)

def profit():
    '''calculates the profit and loss with a 100$ bank roll on a 2 dollar bet'''
    bank = 1000
    bank_total = []
    bank_total.append(1000)
    race_ids = df['raceid'].unique().tolist()
    count = 1
    bet = bank*(.05)
    for race_id in race_ids:
        if race_id[0].isdigit():
            race_df = df[df['raceid'] == race_id]
            odds = list(race_df['winodds'])[0]
        else:
            continue
        odds = float(odds)
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        pick = list(race_df[race_df.winner == race_df.winner.max()]['horseid'])[0]
        if bank == 0:
            print('you are out of money after', count, 'races')
            break
        if winner == pick:
            bank = bank + bet*(float(odds))-bet*.22
            bank_total.append(bank)
            print('you now have', bank, 'dollars after', count, 'races')
        else:
            bank = bank - bet
            bank_total.append(bank)
            print('you now have', bank, 'dollars after', count, 'races')
        count += 1
    return bank_total

def kelly_profit():
    '''calculates profit based on Kelly Criteria'''
    bank = 100
    bank_total = []
    bank_total.append(100)
    race_ids = df['raceid'].unique().tolist()
    count = 1
    for race_id in race_ids:
        if race_id[0].isdigit():
            race_df = df[df['raceid'] == race_id]
            odds = float(list(race_df['winodds'])[0])
            kelly_fraction = float(race_df['winner'].max()) + ((float(race_df['winner'].max())+1)/odds)
        else:
            continue
        odds = float(odds)
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        pick = list(race_df[race_df.winner == race_df.winner.max()]['horseid'])[0]
        betsize = kelly_fraction*bank
        if betsize >= .5*bank:
            betsize = .5*bank
        print('you bet', betsize, 'based on Kelly Criteria' )
        if bank <= 2:
            print('you are out of money after', count, 'races')
            break
        if winner == pick:
            bank = bank + betsize * (float(odds))- (betsize*.22)
            bank_total.append(bank)
            print('you now have', bank, 'dollars after', count, 'races')
        else:
            bank = bank - betsize
            bank_total.append(bank)
            print('you now have', bank, 'dollars after', count, 'races')
        count += 1
    return bank_total

def success_rate_public():
    '''this program calculates the success rate of the model and gives the results of each individual race'''
    race_ids = df['raceid'].unique().tolist()
    total_races = len(race_ids)
    success = 0
    for race_id in race_ids:
        race_df = df[df['raceid'] == race_id]
        pick = list(race_df[race_df.public_prob == race_df.public_prob.max()]['horseid'])[0]
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        answer = 'wrong'
        if winner == pick:
            success +=1
            answer = 'right'
        print("Race number", race_id, 'the public picked', pick, 'to win')
        print('the winner of', race_id, 'was', winner,'.')
        print('they were', answer)
        success_rate = success/total_races
        print('their success rate is:', success_rate)

def public_profit():
    '''calculates the profit and loss with a 100$ bank roll on a 2 dollar bet'''
    bank = 1000
    bank_total = []
    bank_total.append(1000)
    race_ids = df['raceid'].unique().tolist()
    count = 1
    bet = bank*(.05)
    for race_id in race_ids:
        if race_id[0].isdigit():
            race_df = df[df['raceid'] == race_id]
            odds = list(race_df['winodds'])[0]
        else:
            continue
        odds = float(odds)
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        pick = list(race_df[race_df.public_prob == race_df.public_prob.max()]['horseid'])[0]
        if bank == 0:
            print('you are out of money after', count, 'races')
            break
        if winner == pick:
            bank = bank + bet*(float(odds))-bet*.22
            bank_total.append(bank)
            print('the public now has', bank, 'dollars after', count, 'races')
        else:
            bank = bank - bet
            bank_total.append(bank)
            print('the public now has', bank, 'dollars after', count, 'races')
        count += 1
    return bank_total


def public_versus_model():
    race_ids = df['raceid'].unique().tolist()
    model_right = 0
    model_right_public_wrong = 0
    public_right = 0
    public_right_model_wrong = 0
    model_right_pubic_wrong_list = []
    public_right_model_wrong_list = []
    for race_id in race_ids:
        race_df = df[df['raceid'] == race_id]
        public_pick = list(race_df[race_df.public_prob == race_df.public_prob.max()]['horseid'])[0]
        model_pick = list(race_df[race_df.winner == race_df.winner.max()]['horseid'])[0]
        winner = list(race_df[race_df.place == race_df.place.min()]['horseid'])[0]
        if model_pick == public_pick:
            if model_pick == winner:
                print('for race number', race_id, 'the model picked the same as the public.  you were both right')
                model_right += 1
                public_right += 1
            else:
                print('for race number', race_id, 'the model picked the same as the public.  you were both wrong')
        else:
            if model_pick == winner:
                print('for race numer', race_id, 'you picked the winner, the public was wrong.')
                model_right += 1
                model_right_public_wrong += 1
                model_right_pubic_wrong_list.append(race_id)
            elif public_pick == winner:
                print('for race number', race_id, 'the public picked the winner')
                public_right +=1
                public_right_model_wrong+=1
                public_right_model_wrong_list.append(race_id)
            else:
                print('for race number', race_id, 'neither of you got it right.  dumb ass')
    print('You were right', model_right, 'times.  The public was right', public_right,
          'times.  You were right and the public were wrong',model_right_public_wrong, 'times.',
        'The public was right and you were wrong', public_right_model_wrong, 'times.')
    return model_right_pubic_wrong_list, public_right_model_wrong_list


public_versus_model()


public_profit_list = public_profit()
profit_list = profit()
model_races, public_races = public_versus_model()
plt.plot(public_profit_list, label='public')
plt.plot(profit_list, label='model')
plt.xlabel("races")
plt.ylabel("dollars")
plt.title("Profit of Public Model vs. Our Model over 500 races")
plt.legend()
plt.show()



