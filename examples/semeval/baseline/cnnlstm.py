#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split #py3
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
from keras.models import Sequential
from keras.layers import Dense, Flatten, LSTM, Conv1D, MaxPooling1D, Dropout, Activation
from keras.layers.embeddings import Embedding #change

#data_files=['entity_origin','cause_effect','entity_destination','member_collection','component_whole','message_topic','content_container','instrument_agency','product_producer', 'all_classes']
timestr = time.strftime("%Y%m%d-%H%M%S")
stop=stop_words.ENGLISH_STOP_WORDS
RUNS=1
data_files=['entity_origin']

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
        
def create_conv_model(inplen):
    model_conv = Sequential()
    model_conv.add(Embedding(len(word_set)+1, 100, input_length=inplen))
    model_conv.add(Dropout(0.2))
    model_conv.add(Conv1D(64, 5, activation='relu'))
    model_conv.add(MaxPooling1D(pool_size=4))
    model_conv.add(LSTM(100))
    model_conv.add(Dense(1, activation='sigmoid'))
    model_conv.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model_conv

for data_names in data_files:
    inp='../data/training_'+data_names+'.csv'

    fo=open('cl_'+data_names+'.txt','w') #change
    fo.write('SEMEVAL 2010 task 8. Sentences classified as '+data_names+'/Non-'+data_names+'.\n')

    sents=[]
    labels=[]
    all_words=[]   

    maxlen=85
    lcnt=0

    for line in open(inp).readlines():
        if lcnt>0:
            line=line.replace('\n','').replace(',','').split('\t')
            line[0]=line[0].lower()
            line[0]=line[0].translate(str.maketrans('','',string.punctuation))
            words=line[0].split(' ')
            if len(words)<maxlen:
                for ms in range(maxlen-len(words)):
                        words.append('<PAD>')
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
    clf = create_conv_model(len(word_set)) #change
    
    for r in range(RUNS):
        print('Run:',r)
        x_train, x_test, y_train, y_test = train_test_split(data, labels)
        x_train_ids=x_train[:,-1]
        x_test_ids=x_test[:,-1]
        x_train=x_train[:,:-1]
        x_test=x_test[:,:-1]
        clf.fit(x_train, np.array(y_train), validation_split=0.4, epochs = 3)
        print(clf.predict(x_test))
        print(clf.predict(x_test) == y_test)
        print(100*(clf.predict(x_test) == y_test).mean())
        result[r] = 100*(clf.predict(x_test) == y_test).mean()

    fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
    fo.write('baseline_cnnlstm.py\n') #change
    fo.write('\nTotal Runs: '+str(RUNS))
    fo.write('\nBest result:'+str(result.max()))
    fo.write('\nMean result:'+str(result.mean()))
    fo.close()
