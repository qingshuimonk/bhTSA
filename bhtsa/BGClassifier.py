from __future__ import division
import nltk
from nltk.util import ngrams
from nltk.stem.lancaster import LancasterStemmer
import numpy as np
import pickle
from tqdm import tqdm
from process_twt import *


class BGClassifier(object):
    """
    A Naive Bayes Classifier for sentiment analysis

    Attributes:
        feature_list: A list containing informative words
        stop_words: A list containing stop words
        is_trained: An indicator of whether classifier is trained
        BGClassifier: Classifier
    """

    def __init__(self, name='BGClassifier', n=2, feature_th=0.8):
        self.feature_list = []
        self.feature_salience = []
        self.stop_words = get_stopwords()
        self.slang_dict = get_slang_dict()
        self.is_trained = False
        self.BGClassifier = []
        self.ngram = n
        self.feature_th = feature_th
        self.name = name

    def get_feature_vector(self, twt):
        st = LancasterStemmer()
        feature_vector = []
        # split tweet into words
        words = twt.split()
        for w in words:
            # strip punctuation
            w = w.strip('\'"?,.')
            # check if the word stats with an alphabet
            val = re.search(r"^[a-zA-Z][a-zA-Z0-9]*$", w)
            # ignore if it is a stop word
            if w in self.stop_words or val is None:
                continue
            else:
                feature_vector.append(st.stem(w.lower()))
        feature_vector = ngrams(feature_vector, self.ngram)
        return list(feature_vector)                 # feature_vector should be list instead of generator

    def extract_features(self, twt):
        # twt = ngrams(twt, self.ngram)
        twt_words = set(twt)
        features = {}
        for word in self.feature_list:
            features['contains(%s)' % ' '.join(word)] = (word in twt_words)
        return features

    def train(self, pos_twt, neg_twt):
        self.feature_list = []
        tweets = []
        pos_feature_set = []
        neg_feature_set = []
        gram_salience = {}
        pbar = tqdm(total=len(pos_twt)+len(neg_twt), desc='Get FeatureList')
        for row in pos_twt:
            sentiment = 'positive'
            processed_twt = preprocess(row, slangdict=self.slang_dict)
            feature_vector = self.get_feature_vector(processed_twt)
            pos_feature_set.extend(feature_vector)
            for gram in feature_vector:
                if gram not in gram_salience:
                    gram_salience[gram] = np.asarray([1, 0])
                else:
                    gram_salience[gram] += np.asarray([1, 0])
            tweets.append((feature_vector, sentiment))
            pbar.update(1)
        for row in neg_twt:
            sentiment = 'negative'
            processed_twt = preprocess(row, slangdict=self.slang_dict)
            feature_vector = self.get_feature_vector(processed_twt)
            neg_feature_set.extend(feature_vector)
            for gram in feature_vector:
                if gram not in gram_salience:
                    gram_salience[gram] = np.asarray([0, 1])
                else:
                    gram_salience[gram] += np.asarray([0, 1])
            tweets.append((feature_vector, sentiment))
            pbar.update(1)
        pbar.close()
        pos_len = len(pos_feature_set)
        neg_len = len(neg_feature_set)
        pbar = tqdm(total=len(gram_salience), desc='Get Salience')
        for gram in gram_salience:
            counts = gram_salience[gram]
            gram_salience[gram] = 1 - min([counts[0]/pos_len, counts[1]/neg_len]) / \
                                      max([counts[0]/pos_len, counts[1]/neg_len])
            if gram_salience[gram] > self.feature_th:
                self.feature_list.append(gram)
            pbar.update(1)
        pbar.close()

        # get top feature_num grams
        self.feature_salience = gram_salience

        # train classifier
        training_set = nltk.classify.util.apply_features(self.extract_features, tweets)
        self.BGClassifier = nltk.NaiveBayesClassifier.train(training_set)
        self.is_trained = True

    def test(self, twt):
        if self.is_trained:
            score = np.zeros((len(twt), 1))
            for cnt, row in enumerate(twt):
                row = preprocess(row)
                score[cnt] = (self.BGClassifier.prob_classify(self.extract_features(
                    self.get_feature_vector(row)))).prob('positive')
            return score

    def informative_features(self, num=10):
        if self.is_trained:
            return self.BGClassifier.show_most_informative_features(num)
        else:
            return ['Error: Classifier has not been trained']

    def save(self):
        path = os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model')
        if not os.path.exists(path):
            os.makedirs(path)
        f = open(os.path.join(path, self.name+'.pickle'), 'wb')
        pickle.dump(self, f)
        f.close()
