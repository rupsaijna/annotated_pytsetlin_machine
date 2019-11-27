#identify if a sentence is causal or not (presence of causal connective)
import sys
sys.path.append('../../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
#from sklearn.cross_validation import train_test_split #py2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
clause_file='clause_details'+timestr+'.txt'
testing_file='testing_details'+timestr+'.txt'
meta_file='meta_details'+timestr+'.txt'

RUNS=1

inp='data/training_cause_effect.csv'

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
		'''for s in stop:
			if s not in ['because','caused','cause','due','by','to','of','since','he','in', 'therefore', 'hence','causing']:
				regex = r"( |^)"+re.escape(s)+r"( |$)"
				subst = " "
				line[0]=re.sub(regex, subst, line[0], 0, re.MULTILINE).strip()'''
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

NUM_CLAUSES=40
T=15
s=3.9
TRAIN_EPOCHS=1
CLASSES=list(set(labels))
NUM_FEATURES=len(data[0])-1

fo=open(meta_file,'w')
fo.write('SEMEVAL 2010 task 8. Sentences classified as Causal/Non-Causal.\n')
fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
fo.write('semeval_causal_tm_printop.py\n')
fo.write('\nNum Clauses:'+str(NUM_CLAUSES))
fo.write('\nNum Classes: '+ str(len(CLASSES)))
fo.write('\nT: '+str(T))
fo.write('\ns: '+str(s))
fo.write('\nNum Features: '+ str(NUM_FEATURES)+'\n\n')
fo.write('\nTotal Runs: '+str(RUNS))
fo.write('\nTrain Epochs: '+str(TRAIN_EPOCHS))

result=np.zeros(RUNS)

clauses=np.zeros((RUNS*NUM_CLAUSES,NUM_FEATURES*2+1))

clause_dict={}

fot=open(testing_file,'w')

for r in range(RUNS):
	print('Run:',r)
	x_train, x_test, y_train, y_test = train_test_split(data, labels)
	x_train_ids=x_train[:,-1]
	x_test_ids=x_test[:,-1]
	x_train=x_train[:,:-1]
	x_test=x_test[:,:-1]
	#print('\nsplits ready:',x_train.shape, x_test.shape)
	tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s)
	tm.fit(x_train, y_train, epochs=TRAIN_EPOCHS, incremental=True)
	print('\nfit done')
	res=tm.predict(x_test)
	result[r] = 100*(res == y_test).mean()
	X_test_transformed = tm.transform(x_test)
	print('here', len(x_test))
	for testsample in range(5):
		print(testsample)
		print(res[testsample],y_test[testsample])
		if res[testsample]!=y_test[testsample]:
			sid=x_test_ids[testsample]
			fot.write(str(sid)+'\t'+' '.join(sents[sid])+'\n')
			strTransformed=[str(k) for k in X_test_transformed[testsample]]
			fot.write(' '.join(strTransformed)+'\n')
			for cur_cls in CLASSES:
				for cur_clause in range(NUM_CLAUSES):
					if X_test_transformed[testsample][cur_clause]==1:
						if cur_clause%2==0:
							clause_type='positive'
						else:
							clause_type='negative'
						this_clause='cur_clause\t'
						for f in range(0,NUM_FEATURES):
							action_plain = tm.ta_action(int(cur_cls), cur_clause, f)
							action_negated = tm.ta_action(int(cur_cls), cur_clause, f+NUM_FEATURES)
							if action_plain==1:
								this_clause+=str(reverse_word_map[f])+';'
							if action_negated==1:
								this_clause+=' #'+str(reverse_word_map[f])+';'
						this_clause+='\t'+clause_type+'\t'+str(cur_cls)	
						fot.write('cl:'+ this_clause+'\n')
					

fo.write('\nBest result:'+str(result.max()))
fo.write('\nMean result:'+str(result.mean()))
fo.close()
