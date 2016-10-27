from nltk.twitter import Query, credsfromfile, TweetViewer
import process_twt
from nltk.corpus import twitter_samples, TwitterCorpusReader
from NBClassifier import NBClassifier


# settings
oauth = credsfromfile()
client = Query(**oauth)
twtNum = 10
client.register(TweetViewer(limit=twtNum))
tweets_gen = client.search_tweets(keywords='spurs', limit=twtNum, lang='en')
tweets = []
slangdict = process_twt.get_slangdictionary()
for t in tweets_gen:
    print process_twt.preprocess(t['text'],slangdict=slangdict)

stopwords = process_twt.get_stopwords()


fileIds = twitter_samples.fileids()
root = twitter_samples.root

negReader = TwitterCorpusReader(root, fileIds[0])
negTwt = []
posReader = TwitterCorpusReader(root, fileIds[1])
posTwt = []
for tweet in negReader.docs():
    negTwt.append((tweet['text']))
for tweet in posReader.docs():
    posTwt.append((tweet['text']))

NBC = NBClassifier()
NBC.train(posTwt, negTwt)
print NBC.informative_features()
