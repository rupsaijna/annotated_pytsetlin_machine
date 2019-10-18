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

print('bigrams and unigrams. stopwords not removed. punctuation not removed')
RUNS=100

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
		#line[0]=line[0].translate(None, string.punctuation)
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

NUM_CLAUSES=200
T=15
s=3.9
CLASSES=list(set(labels))
NUM_FEATURES=len(data[0])-1



result=np.zeros(RUNS)
feature_count_plain=np.zeros((RUNS,NUM_FEATURES))
feature_count_negated=np.zeros((RUNS,NUM_FEATURES))
fout_c=open('clause_details.csv','w')
fout_c.write('Num Clauses:'+str(NUM_CLAUSES))
fout_c.write('Num Classes: '+ str(len(CLASSES)))
fout_c.write('T: '+str(T))
fout_c.write('s: '+str(s))
fout_c.write('Num Features: '+ str(NUM_FEATURES))
clauses=np.zeros((RUNS*NUM_CLAUSES,NUM_FEATURES*2+1))

for r in range(RUNS):
	x_train, x_test, y_train, y_test = train_test_split(data, labels)
	x_train_ids=x_train[:,-1]
	x_test_ids=x_test[:,-1]
	x_train=x_train[:,:-1]
	x_test=x_test[:,:-1]

	#print('\nsplits ready:',x_train.shape, x_test.shape)
	tm = MultiClassTsetlinMachine(NUM_CLAUSES, T, s)
	tm.fit(x_train, y_train, epochs=200, incremental=True)
	print('\nfit done')
	result[r] = 100*(tm.predict(x_test) == y_test).mean()
	fout_c.write(str(r)+'\t')
	feature_vector=np.zeroes(NUM_FEATURES*2)
	for cur_cls in CLASSES:
		for cur_clause in range(NUM_CLAUSES):
			fout_c.write(str(cur_clause)+'\t')
			if cur_clause%2==0:
				fout_c.write('positive'+'\t')
			else:
				fout_c.write('negative'+'\t')
			fout_c.write(str(cur_cls)+'\t')
			for f in range(0,NUM_FEATURES):
				action_plain = tm.ta_action(int(cur_cls), cur_clause, f)
				action_negated = tm.ta_action(int(cur_cls), cur_clause, f+NUM_FEATURES)
				feature_vector[f]=action_plain
				feature_vector[f+NUM_FEATURES]=action_negated
				feature_count_plain[r][f]+=action_plain
				feature_count_negated[r][f]+=action_negated
				
			for fv in feature_vector:
				if fv==0:
					fout_c.write('*'+'\t')
				else:
					fout_c.write('1'+'\t')
			fout_c.write(str(result[r])+'\n')

fout=open('feature_details.csv','w')
for r in range(RUNS):
	for f in range(0,NUM_FEATURES):
		fout.write(str(r)+'\t'+feature_count_plain[r][f]+'\t'+feature_count_negated[r][f]+'\n')
fout.close()
fout_c.close()
