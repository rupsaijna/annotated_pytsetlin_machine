#identify if a sentence is causal or not (presence of causal connective)
from keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
import numpy as np
inp='is_causal_data.txt'
	
sents=[]
labels=[]
all_words=[]

def encode_sentences(txt):
	feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
	tnum=0
	for t in txt:
		for w in t:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		tnum+=1
	return feature_set

maxlen=0
for line in open(inp).readlines():
  line=line.replace('\n','').replace(',','').split('\t')
  words=line[0].lower().split(' ')
  if len(words)>maxlen:
    maxlen=len(words)
  all_words+=words
  sents.append(words)
  labels.append(int(line[1]))
  
word_set=set(all_words)
word_idx = dict((c, i + 1) for i, c in enumerate(word_set))
reverse_word_map = dict(map(reversed, word_idx.items()))
data=encode_sentences(sents)

#print(word_idx)
#print(sents[0], data[0])

x_train, x_test, y_train, y_test = train_test_split(data, labels)


print('\nsplits ready:',x_train.shape, x_test.shape)
tm = MultiClassTsetlinMachine(20, 10, 2.9)
tm.fit(x_train, y_train, epochs=1, incremental=True)
print('\nfit done')
result = 100*(tm.predict(x_test) == y_test).mean()
print(result)


