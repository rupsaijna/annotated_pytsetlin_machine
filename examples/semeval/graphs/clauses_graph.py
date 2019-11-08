#identify if a sentence is causal or not (presence of causal connective)
import sys
sys.path.append('../../../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
#from sklearn.cross_validation import train_test_split #py2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
import matplotlib.pyplot as plt


timestr = time.strftime("%Y%m%d-%H%M%S")
meta_file='accoverclauses'+timestr+'.txt'
fig_file='accoverclauses'+timestr+'.png'

STEP=2
STEP_SIZE=10
RUNS=10

inp='../data/training_cause_effect.csv'

sents=[]
labels=[]
all_words=[]

stop=stop_words.ENGLISH_STOP_WORDS

def encode_sentences(txt):
	feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
	tnum=0
	for t in txt:
		if t[0]==10:
			print (t)
		s_words=t[1:]+list(set(list(everygrams(t[1:], min_len=2,max_len=2))))
		for w in s_words:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		feature_set[tnum][-1]=t[0]
		tnum+=1
	return feature_set

maxlen=0
lcnt=0

for line in open(inp).readlines():
	if lcnt>0:
		line=line.replace('\n','').replace(',','').split('\t')
		line[0]=line[0].lower()
		line[0]=line[0].translate(str.maketrans('','',string.punctuation)) #.translate(None, string.punctuation) #py2
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

NUM_CLAUSES=-5
T=15
s=3.9
TRAIN_EPOCHS=40
CLASSES=list(set(labels))
NUM_FEATURES=len(data[0])-1

fo=open(meta_file,'w')
fo.write('SEMEVAL 2010 task 8. Sentences classified as Causal/Non-Causal.\n')
fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
fo.write('semeval_causal_tm4.py\n')
fo.write('\nNum Clauses:'+str(NUM_CLAUSES))
fo.write('\nNum Classes: '+ str(len(CLASSES)))
fo.write('\nT: '+str(T))
fo.write('\ns: '+str(s))
fo.write('\nNum Features: '+ str(NUM_FEATURES))
fo.write('\nTotal Runsper Step: '+str(RUNS))
fo.write('\nTotal Steps: '+str(STEPS))
fo.write('\nSTEP Size: '+str(STEP_SIZE))
fo.write('\nTrain Epochs: '+str(TRAIN_EPOCHS)+'\n\n')
fo.write('Num_CLAUSES\tMean\tMax')

result_mean=np.zeros(STEPS)
result_max=np.zeros(STEPS)
clausesizes=np.zeros(STEPS)

x_train, x_test, y_train, y_test = train_test_split(data, labels)
x_train_ids=x_train[:,-1]
x_test_ids=x_test[:,-1]
x_train=x_train[:,:-1]
x_test=x_test[:,:-1]

for s in range(STEP):
	lr=np.zeros(RUNS)
	for r in range(RUNS):
		NUM_CLAUSES+=STEP_SIZE
		tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s)
		tm.fit(x_train, y_train, epochs=TRAIN_EPOCHS, incremental=True)
		lr[r]=100*(tm.predict(x_test) == y_test).mean()
	result_mean[s] = lr.mean()
	result_max[s] = lr.max()
	clausesizes[s]=NUM_CLAUSES
	
	fo.write(str(NUM_CLAUSES)+'\t'+str(result_mean[r])+'\t'+str(result_max[r])+'\n')

plt.plot(clausesizes,result_mean)
plt.plot(clausesizes,result_max)

plt.legend(['Avg.Accuracy', 'Max Accuracy'], loc='upper left')
plt.xlabel('#Clauses')
plt.ylabel('Accuracy')

plt.show()
plt.savefig(fig_file)

fo.write('\n\nBest result:'+str(result_max.max()))
fo.write('\nMean result:'+str(result_max.mean()))
fo.close()
