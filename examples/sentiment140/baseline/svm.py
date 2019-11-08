#from sklearn.cross_validation import train_test_split
from sklearn.model_selection import train_test_split #py3
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
from sklearn import svm #change

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

inp='../tweets_positivenegative.csv'

fo=open('svm_res.txt','w') #change
fo.write('Sentiment140. Positive/Negative.\n')

sents=[]
labels=[]
all_words=[]   

df=pd.read_csv(inp,sep='\t', quoting=2, dtype={'id ':int,'polarity': int })
df = df.dropna()
data=df.iloc[np.r_[0:num_ex, -num_ex:0]]

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

maxlen=0
lcnt=0

for ind, row in data.iterrows():
	tw=row['tweet'].lower()
	words=tknzr.tokenize(tw)
	bl=list(set(list(everygrams(words, min_len=2,max_len=2))))
	all_words+=words+bl
	words.insert(0,lcnt)
	sents.append(words)
	if row['polarity']==4:
		labels.append(1)
	else:
		labels.append(0)
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

fo.write('bigrams and unigrams. stopwords not removed. punctuation not removed.\n')
fo.write('baseline_svm.py\n') #change
fo.write('\nTotal Runs: '+str(RUNS))
fo.write('\nBest result:'+str(result.max()))
fo.write('\nMean result:'+str(result.mean()))
fo.close()
