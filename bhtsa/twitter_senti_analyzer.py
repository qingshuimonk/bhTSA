from nltk.twitter import TweetViewer
import pickle
import numpy as np
import datetime
import os


def senti_score_daily(keyword, client, classifier, twt_num, start_time, days, verb=0):
    score_all = np.zeros((twt_num, days))
    year = start_time[0]
    month = start_time[1]
    day = start_time[2]
    origin = datetime.date(year, month, day)

    for i in range(days):
        start_t = origin + datetime.timedelta(days=i)
        end_t = origin + datetime.timedelta(days=i+1)
        date1 = start_t.timetuple()[:6]
        date2 = end_t.timetuple()[:6]

        # if not exits get tweets from server and save them, otherwise just load them
        filename = keyword+'_'+str(start_t)+'_'+str(end_t)+'.pickle'
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'tweets')
        if os.path.isfile(os.path.join(path, filename)):
            f = open(os.path.join(path, filename), 'rb')
            tweets = pickle.load(f)
            f.close()
        else:
            client.register(TweetViewer(limit=twt_num, lower_date_limit=date1, upper_date_limit=date2))
            tweets_gen = client.search_tweets(keywords=keyword, limit=twt_num, lang='en')
            tweets = []
            for t in tweets_gen:
                tweets.append(t)
            f = open(os.path.join(path, filename), 'wb')
            pickle.dump(tweets, f)
            f.close()

        score = classifier.test(tweets)

        score_all[:, i] = score[:, 0]
        if verb == 1:
            print keyword+' : '+str(start_t)+' to '+str(end_t)+' score='+str(np.mean(score_all[:, i], axis=0))

    return score_all
