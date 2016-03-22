import os
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import cPickle


def getFeature():
    with open(os.path.join('spam_filter_train.txt'), 'r') as f:
        trainData = f.readlines()
    with open(os.path.join('spam_filter_test.txt'), 'r') as f:
        testData = f.readlines()
    data = trainData + testData
    trainNum, testNum = len(trainData), len(testData)
    del trainData
    del testData

    for i in range(len(data)):
        data[i] = data[i].replace('\n', '').split('\t')[1]
    # lemmatize
    lemmatized = []
    wnl = WordNetLemmatizer()
    for line in data:
        lemmatized.append([wnl.lemmatize(word) for word in line.split(' ')])
    # remove stopwords
    stopwordRemoved = []
    sw = set(stopwords.words('english'))
    for line in lemmatized:
        stopwordRemoved.append(' '.join([x for x in line if x not in sw]))
    # tf feature
    vec = CountVectorizer()
    features = vec.fit_transform(stopwordRemoved)

    with open('trainFeatures.pkl', 'wb') as f:
        cPickle.dump(features[:trainNum], f)
    with open('testFeatures.pkl', 'wb') as f:
        cPickle.dump(features[trainNum:], f)

def main():
    getFeature()
    '''
    with open('trainFeatures.pkl', 'rb') as f:
         trainFeatures = cPickle.load(f)
    with open('testFeatures.pkl', 'rb') as f:
         testFeatures = cPickle.load(f)
    '''


if __name__ == '__main__':
    main()
