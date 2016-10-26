from nltk.twitter import Query, credsfromfile, TweetViewer
import process_twt

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
