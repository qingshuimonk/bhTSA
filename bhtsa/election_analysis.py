from nltk.twitter import Query, credsfromfile
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
from twitter_senti_analyzer import senti_score_daily
import datetime as dt
import os
import pickle

# settings
oauth = credsfromfile()
client = Query(**oauth)
twtNum = 1000
startTime = [2016, 9, 5]
days = 60

path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model')
f = open(os.path.join(path, 'NBClassifier.pickle'), 'r')
NBC = pickle.load(f)

origin = dt.date(startTime[0], startTime[1], startTime[2])
dates = []
for i in range(days):
    next_val = origin + dt.timedelta(days=i)
    dates.append(next_val)

path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'tweets')
if os.path.isfile(os.path.join(path, 'hilary_score.pickle')):
    f = open(os.path.join(path, 'hilary_score.pickle'), 'rb')
    hilary_score = pickle.load(f)
    f.close()
else:
    hilary_score = senti_score_daily('hilary clinton', client, NBC, twtNum, startTime, days, 1)
    f = open(os.path.join(path, 'hilary_score.pickle'), 'wb')
    pickle.dump(hilary_score, f)
    f.close()

if os.path.isfile(os.path.join(path, 'trump_score.pickle')):
    f = open(os.path.join(path, 'trump_score.pickle'), 'rb')
    trump_score = pickle.load(f)
    f.close()
else:
    trump_score = senti_score_daily('donald trump', client, NBC, twtNum, startTime, days, 1)
    f = open(os.path.join(path, 'trump_score.pickle'), 'wb')
    pickle.dump(trump_score, f)
    f.close()

hilary_mean = np.mean(hilary_score, axis=0)
hilary_upper = hilary_mean + np.std(hilary_score, axis=0)*0.1
hilary_lower = hilary_mean - np.std(hilary_score, axis=0)*0.1

trump_mean = np.mean(trump_score, axis=0)
trump_upper = trump_mean + np.std(trump_score, axis=0)*0.1
trump_lower = trump_mean - np.std(trump_score, axis=0)*0.1

x = hilary_score.reshape((-1, 1))
plt.hist(x, 100, facecolor='green', alpha=0.5, label='Hilary Clinton')
x = trump_score.reshape((-1, 1))
plt.hist(x, 100, facecolor='blue', alpha=0.5, label='Donald Trump')
plt.legend(loc='upper right')
plt.xlabel('Sentiment Score')
plt.ylabel('#Tweets')
plt.show()

# plot score distribution
fig, ax = plt.subplots()
plt.xticks(rotation=70)
plt.plot(dates, np.mean(hilary_score, axis=0), color='blue', linewidth=2, label='Hilary Clinton')
plt.plot(dates, np.mean(trump_score, axis=0), color='red', linewidth=2, label='Donald Trump')

ax.fill_between(dates, hilary_lower, hilary_upper, facecolor='yellow', alpha=0.5)
ax.fill_between(dates, trump_lower, trump_upper, facecolor='yellow', alpha=0.5)

ax.xaxis.set_major_formatter(DateFormatter('%m-%d'))
plt.xlabel('days')
plt.ylabel('score')
plt.title('Sentiment Score of Candidates')
plt.legend(loc="lower left")
plt.grid(True)
plt.show()
