#identify if a sentence is causal or not (presence of causal connective)
import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
from keras.preprocessing.sequence import pad_sequences
from sklearn.cross_validation import train_test_split
import numpy as np
inp='is_causal_data.txt'
	
sents=[]
labels=[]
all_words=[]

def encode_sentences(txt):
	feature_set=np.zeros((len(txt), len(word_set)+2),dtype=int)
	tnum=0
	for t in txt:
		for w in t[1:]:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		feature_set[tnum][-1]=t[0]
		tnum+=1
	return feature_set

maxlen=0
lcnt=0
for line in open(inp).readlines():
  line=line.replace('\n','').replace(',','').split('\t')
  words=line[0].lower().split(' ')
  if len(words)>maxlen:
    maxlen=len(words)
  all_words+=words
  words.insert(0,lcnt)
  lcnt+=1
  sents.append(words)
  labels.append(int(line[1]))
  
word_set=set(all_words)
word_idx = dict((c, i + 1) for i, c in enumerate(word_set))
reverse_word_map = dict(map(reversed, word_idx.items()))
data=encode_sentences(sents)

#print(word_idx)
#print(sents[10], data[10])

x_train, x_test, y_train, y_test = train_test_split(data, labels)
x_train_ids=x_train[:,-1]
x_test_ids=x_test[:,-1]
x_train=x_train[:,:-1]
x_test=x_test[:,:-1]


print('\nsplits ready:',x_train.shape, x_test.shape)
tm = MultiClassTsetlinMachine(20, 15, 3.9)
tm.fit(x_train, y_train, epochs=5, incremental=True)
print('\nfit done')
result = 100*(tm.predict(x_test) == y_test).mean()
print(result)

res=tm.predict(x_test)
for i in range(len(x_test_ids)):
	sidx=x_test_ids[i]
	print(sents[sidx], res[i])
