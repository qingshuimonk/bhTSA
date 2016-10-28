from nltk.twitter import Query, credsfromfile, TweetViewer
import process_twt
from NBClassifier import NBClassifier
from nltk.corpus import twitter_samples, TwitterCorpusReader
import os
import pickle

# # settings
# oauth = credsfromfile()
# client = Query(**oauth)
# twtNum = 10
# client.register(TweetViewer(limit=twtNum))
# tweets_gen = client.search_tweets(keywords='spurs', lang='en')
# tweets = []
# slangdict = process_twt.get_slangdictionary()
# twt_list = []
# for t in tweets_gen:
#     twt_list.append(process_twt.preprocess(t['text'], slangdict=slangdict))
# twt_list = list(set(twt_list))
#
# for t in twt_list[:twtNum]:
#     print t

fileIds = twitter_samples.fileids()
root = twitter_samples.root

# read tweet data from corpus
# negReader = TwitterCorpusReader(root, fileIds[0])
# negTwt = []
# posReader = TwitterCorpusReader(root, fileIds[1])
# posTwt = []
# for tweet in negReader.docs():
#     negTwt.append((tweet['text']))
# for tweet in posReader.docs():
#     posTwt.append((tweet['text']))
# NBC = NBClassifier()
# print 'Training NBClassifier...'
# NBC.train(posTwt, negTwt)
# print 'Done!'
# print NBC.informative_features()
# NBC.save()

path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model')
f = open(os.path.join(path, 'NBClassifier.pickle'), 'r')
NBC = pickle.load(f)
print NBC.informative_features()
