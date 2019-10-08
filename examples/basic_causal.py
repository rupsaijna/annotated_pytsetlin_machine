#identify if a sentence is causal or not (presence of causal connective)
from keras.preprocessing.text import text_to_word_sequence
inp='is_causal_data.txt'
	
def vectorize_stories(txts, ml):
	vectors=[]
	for t in txts:
		v=[word_idx[w] for w in t]
		vectors.append(v)
	return pad_sequences(vectors,maxlen=ml,padding='post')

sents=[]
labels=[]
all_words=[]

maxlen=0
for line in open(inp).readlines():
  line=line.replace('\n','').replace(',','').split('\t')
  words=line[0].lower().split(' ')
  if len(words)>maxlen:
    maxlen=len(words)
  all_words+=words
  sents.append(words)
  labels.append(int(line[1]))
  

word_idx = dict((c, i + 1) for i, c in enumerate(bigram_set))
reverse_word_map = dict(map(reversed, word_idx.items()))
vs=vectorize_sentences(sents)

print(word_idx)
print(sents[0], vs[0])
