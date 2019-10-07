#identify if a sentence is causal or not (presence of causal connective)
from keras.preprocessing.text import text_to_word_sequence
inp='is_causal_data.txt'

sents=[]
labels=[]
all_words=[]

for line in open(inp).readlines():
  line=line.replace('\n','').split('\t')
  words=line[0].split(' ')
  all_words+=words
  sents.append(words)
  labels.append(int(line[1]))
  
print(sents)
print(set(tuple(row) for row in sents))
print(labels)
