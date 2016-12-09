import csv
import os
import matplotlib.pyplot as plt
import numpy as np
from BGClassifier import BGClassifier

# load training data
negTwt = []
posTwt = []
with open(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'model',
                       'Sentiment Analysis Dataset.csv'), 'rb') as csvfile:
    senti_reader = csv.reader(csvfile, delimiter=',')
    for row in senti_reader:
        if row[0].isdigit():
            if int(row[1]) == 0:
                negTwt.append(row[3].strip())
            else:
                posTwt.append(row[3].strip())

# train classifiers
BGC = BGClassifier()
BGC.train(posTwt, negTwt)
plt.figure()
salience = []
for key in BGC.feature_salience:
    salience.append(BGC.feature_salience[key])
n, bins, patches = plt.hist(np.asarray(salience), bins=50, facecolor='green', alpha=0.75)
plt.show()
