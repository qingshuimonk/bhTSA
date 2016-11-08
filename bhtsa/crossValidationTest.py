from nltk.corpus import twitter_samples, TwitterCorpusReader
import numpy as np
import matplotlib.pyplot as plt
from NBClassifier import NBClassifier
from SCClassifier import SCClassifier
from sklearn.metrics import roc_curve, auc
import os
import pickle

# settings
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

posInd = np.random.permutation(len(posTwt))
negInd = np.random.permutation(len(negTwt))

X_1 = np.array([])
X_2 = np.array([])
Y = np.array([])
NB_auc = np.zeros((5, 1))
BH_auc = np.zeros((5, 1))

for fold in range(5):
    # this is ugly, but works for now
    train_idx_pos, test_idx_pos = posInd[:0.8 * len(posTwt)], posInd[0.8 * len(posTwt):]
    train_idx_neg, test_idx_neg = negInd[:0.8 * len(negTwt)], negInd[0.8 * len(negTwt):]
    train_pos = [posTwt[i] for i in train_idx_pos]
    train_neg = [negTwt[i] for i in train_idx_neg]
    test = [posTwt[i] for i in test_idx_pos] + [negTwt[i] for i in test_idx_neg]
    target = np.concatenate((np.ones((len(test_idx_pos), 1)), np.zeros((len(test_idx_neg), 1))), axis=0)
    NBC = NBClassifier()
    NBC.train(train_pos, train_neg)
    SCC = SCClassifier(nbc=NBC)
    SCC.train(train_pos, train_neg)

    score_1 = NBC.test(test)
    score_2 = SCC.test(test)

    X_1 = np.append(X_1, score_1)
    X_2 = np.append(X_2, score_2)
    Y = np.append(Y, target)

    fpr = dict()
    tpr = dict()
    fpr, tpr, _ = roc_curve(target, score_1)
    NB_auc[fold] = auc(fpr, tpr)

    fpr = dict()
    tpr = dict()
    fpr, tpr, _ = roc_curve(target, score_2)
    BH_auc[fold] = auc(fpr, tpr)

f = open(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'result', 'roc_NBvsSC_NB_no_slang.pickle'), 'wb')
pickle.dump((NB_auc, BH_auc, X_1, X_2, Y), f)
f.close()

# get ROC
fpr_1 = dict()
tpr_1 = dict()
fpr_2 = dict()
tpr_2 = dict()
roc_auc_1 = dict()
roc_auc_2 = dict()
fpr_1, tpr_1, _ = roc_curve(Y, X_1)
fpr_2, tpr_2, _ = roc_curve(Y, X_2)
roc_auc_1 = auc(fpr_1, tpr_1)
roc_auc_2 = auc(fpr_2, tpr_2)

plt.figure()
lw = 2
plt.plot(fpr_1, tpr_1, color='blue', linewidth=lw, label='NaiveBayes (area = %0.2f)' % roc_auc_1)
plt.plot(fpr_2, tpr_2, color='green', linewidth=lw, label='BHClassifier (area = %0.2f)' % roc_auc_2)
# plt.plot(fpr[0], tpr[0])
plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Comparison')
plt.legend(loc="lower right")
plt.show()
