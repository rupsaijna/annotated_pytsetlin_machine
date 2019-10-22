#identify if a sentence is causal or not (presence of causal connective)
import sys
sys.path.append('../../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
from sklearn.cross_validation import train_test_split
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
clause_file='clause_details'+timestr+'.txt'
feature_file='feature_details'+timestr+'.txt'
meta_file='meta_details'+timestr+'.txt'

RUNS=200

inp='training.csv'

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
		line[0]=line[0].translate(None, string.punctuation)
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

NUM_CLAUSES=30
T=15
s=3.9
TRAIN_EPOCHS=10
CLASSES=list(set(labels))
NUM_FEATURES=len(data[0])-1

fo=open(meta_file,'w')
fo.write('SEMEVAL 2010 task 8. Sentences classified as Causal/Non-Causal.\n')
fo.write('bigrams and unigrams. stopwords not removed. punctuation removed.\n')
fo.write('\nNum Clauses:'+str(NUM_CLAUSES))
fo.write('\nNum Classes: '+ str(len(CLASSES)))
fo.write('\nT: '+str(T))
fo.write('\ns: '+str(s))
fo.write('\nNum Features: '+ str(NUM_FEATURES)+'\n\n')
fo.write('\nTotal Runs: '+str(RUNS))
fo.write('\nTrain Epochs: '+str(TRAIN_EPOCHS))

result=np.zeros(RUNS)
feature_count_plain=np.zeros(NUM_FEATURES)
feature_count_negated=np.zeros(NUM_FEATURES)



clauses=np.zeros((RUNS*NUM_CLAUSES,NUM_FEATURES*2+1))

'''for f in range(NUM_FEATURES):
	fout_c.write(str(reverse_word_map[f])+'\t')
for f in range(NUM_FEATURES):
	fout_c.write('^'+str(reverse_word_map[f])+'\t')
fout_c.write('result\n')'''

clause_dict={}

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
	result[r] = 100*(tm.predict(x_test) == y_test).mean()
	feature_vector=np.zeros(NUM_FEATURES*2)
	for cur_cls in CLASSES:
		for cur_clause in range(NUM_CLAUSES):
			if cur_clause%2==0:
				clause_type='positive'
			else:
				clause_type='negative'
			this_clause=''
			for f in range(0,NUM_FEATURES):
				action_plain = tm.ta_action(int(cur_cls), cur_clause, f)
				action_negated = tm.ta_action(int(cur_cls), cur_clause, f+NUM_FEATURES)
				feature_vector[f]=action_plain
				feature_vector[f+NUM_FEATURES]=action_negated
				feature_count_plain[f]+=action_plain
				feature_count_negated[f]+=action_negated
				'''if action_plain==1:
					this_clause+=str(reverse_word_map[f])+';'
				if action_negated==1:
					this_clause+=' #'+str(reverse_word_map[f])+';''''
				if action_plain==1:
					this_clause+=str(f)+';'
				if action_negated==1:
					this_clause+=' #'+str(f)+';'
			this_clause+='\t'+clause_type+'\t'+str(cur_cls)	
			print('cl:', this_clause)
			if this_clause in clause_dict.keys():
				clause_dict[this_clause]+=1
			else:
				clause_dict[this_clause]=1
	fout_f=open(feature_file,'w')
	fout_f.write('run\tfeature\tcount_plain\tcount_negated\n')
	for f in range(0,NUM_FEATURES):
		fout_f.write(str(r)+'\t'+str(reverse_word_map[f])+'\t'+str(feature_count_plain[f])+'\t'+str(feature_count_negated[f])+'\n')
	fout_f.close()
	
	fout_c=open(clause_file,'w')
	fout_c.write('Run\tClause\tp/n\tclass\tcount\n')
	for c in clause_dict.keys():
		fout_c.write(str(r)+'\t')
		fout_c.write(c+'\t'+str(clause_dict[c])+'\n')
	fout_c.close()

fo.write('\nBest result:'+str(result.max()))
fo.write('\nMean result:'+str(result.mean()))
fo.close()

