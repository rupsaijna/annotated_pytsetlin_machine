#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split #py3
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
from sklearn.naive_bayes import svm #change

data_files=['entity_origin','cause_effect','entity_destination','member_collection','component_whole','message_topic','content_container','instrument_agency','product_producer', 'all_classes']
timestr = time.strftime("%Y%m%d-%H%M%S")
stop=stop_words.ENGLISH_STOP_WORDS
RUNS=3


def encode_sentences(txt):
        feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
        tnum=0
        for t in txt:
            s_words=t[1:]+list(set(list(everygrams(t[1:], min_len=2,max_len=2))))
            for w in s_words:
                idx=word_idx[w]
                feature_set[tnum][idx]=1
            feature_set[tnum][-1]=t[0]
            tnum+=1
        return feature_set

for data_names in data_files:
    inp='../data/training_'+data_names+'.csv'

    fo=open('svm_'+data_names+'.txt','w') #change
    fo.write('SEMEVAL 2010 task 8. Sentences classified as '+data_names+'/Non-'+data_names+'.\n')

    sents=[]
    labels=[]
    all_words=[]   

    maxlen=0
    lcnt=0

    for line in open(inp).readlines():
        if lcnt>0:
            line=line.replace('\n','').replace(',','').split('\t')
            line[0]=line[0].lower()
            line[0]=line[0].translate(str.maketrans('','',string.punctuation))
            words=line[0].split(' ')
            bl=list(set(list(everygrams(words, min_len=2,max_len=2))))
            all_words+=words+bl
            words.insert(0,lcnt)
            sents.append(words)
            labels.append(int(line[1]))
        lcnt+=1


    word_set=set(all_words)
    i=0
    word_idx = dict((c, i + 1) for i, c in enumerate(word_set,start = -1))
    reverse_word_map = dict(map(reversed, word_idx.items()))
    data=encode_sentences(sents)

    CLASSES=list(set(labels))
    NUM_FEATURES=len(data[0])-1

    result=np.zeros(RUNS)
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo') #change
    
    for r in range(RUNS):
        print('Run:',r)
        x_train, x_test, y_train, y_test = train_test_split(data, labels)
        x_train_ids=x_train[:,-1]
        x_test_ids=x_test[:,-1]
        x_train=x_train[:,:-1]
        x_test=x_test[:,:-1]
        clf.fit(x_train, y_train)
        result[r] = 100*(clf.predict(x_test) == y_test).mean()

    fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
    fo.write('baseline_svm.py\n') #change
    fo.write('\nTotal Runs: '+str(RUNS))
    fo.write('\nBest result:'+str(result.max()))
    fo.write('\nMean result:'+str(result.mean()))
    fo.close()

for r in range(RUNS):
	print('Run:',r)
	x_train, x_test, y_train, y_test = train_test_split(data, labels)
	x_train_ids=x_train[:,-1]
	x_test_ids=x_test[:,-1]
	x_train=x_train[:,:-1]
	x_test=x_test[:,:-1]
	clf.fit(x_train, y_train)
	result[r] = 100*(clf.predict(x_test) == y_test).mean()


fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
fo.write('baseline_svm.py\n')
fo.write('\nTotal Runs: '+str(RUNS))
fo.write('\nBest result:'+str(result.max()))
fo.write('\nMean result:'+str(result.mean()))
fo.close()
