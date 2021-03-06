from nltk.twitter import Query, credsfromfile, TweetViewer
import process_twt
from NBClassifier import NBClassifier
from SCClassifier import SCClassifier
from BGClassifier import BGClassifier
from nltk.corpus import twitter_samples, TwitterCorpusReader
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np

# settings
oauth = credsfromfile()
client = Query(**oauth)
twtNum = 10
client.register(TweetViewer(limit=twtNum))
tweets_gen = client.search_tweets(keywords='hearthstone', lang='en')
tweets = []
slangdict = process_twt.get_slang_dict()
twt_list = []
for t in tweets_gen:
    twt_list.append(process_twt.preprocess(t['text'], slangdict=slangdict))
twt_list = list(set(twt_list))

for t in twt_list[:twtNum]:
    print t

fileIds = twitter_samples.fileids()
root = twitter_samples.root

# read tweet data from corpus
negReader = TwitterCorpusReader(root, fileIds[0])
negTwt = []
posReader = TwitterCorpusReader(root, fileIds[1])
posTwt = []
for tweet in negReader.docs():
    negTwt.append((tweet['text']))
for tweet in posReader.docs():
    posTwt.append((tweet['text']))

path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model')
f = open(os.path.join(path, 'NBClassifier.pickle'), 'r')
NBC = pickle.load(f)

neg_sample = 'I hurt my leg this afternoon. Now I can do nothing but stay on bed'
pos_sample = 'Great! This classifier works!'

# NBC.test([neg_sample])

BGC = BGClassifier()
print 'Training BGClassifier...'
BGC.train(posTwt, negTwt)
print 'Done!'

print BGC.test([neg_sample])
print BGC.test([pos_sample])

plt.figure()
salience = []
for key in BGC.feature_salience:
    salience.append(BGC.feature_salience[key])
n, bins, patches = plt.hist(np.asarray(salience), 50, normed=1, facecolor='green', alpha=0.75)
plt.show()
