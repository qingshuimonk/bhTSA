import re
import os


def reduce_redundancy(twt):
    pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
    return pattern.sub(r"\1\1", twt)


def preprocess(twt, tolower=True, slangdict={}):
    # replace slangs with original words
    for word in twt.split(' '):
        if word.strip() in slangdict:
            twt = twt.replace(word.strip(), slangdict[word.strip()])
    if tolower:
        twt = twt.lower()
    twt = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', twt)
    twt = re.sub('@[^\s]+', 'AT_USER', twt)
    twt = re.sub('[\s]+', ' ', twt)
    twt = re.sub(r'#([^\s]+)', r'\1', twt)
    twt = twt.strip('\'"')
    twt = twt.strip()
    twt = reduce_redundancy(twt)
    return twt


def get_stopwords():
    stopwords = []

    fp = open(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'stopwords.txt'), 'r')
    line = fp.readline()
    while line:
        word = line.strip()
        stopwords.append(word)
        line = fp.readline()
    fp.close()
    return stopwords


def get_slang_dict():
    fp = open(os.path.join(os.path.dirname(__file__), os.pardir, 'data', 'slangdict.txt'), 'r')
    line = fp.readline()
    slangdict = {}
    while line:
        words = line.split('-')
        if len(words) == 2:
            slangdict.update({words[0].strip(): words[1].strip()})
        line = fp.readline()
    fp.close()
    return slangdict
