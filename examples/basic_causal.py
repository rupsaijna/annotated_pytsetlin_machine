#identify if a sentence is causal or not (presence of causal connective)
from keras.preprocessing.text import text_to_word_sequence
inp='is_causal_data.txt'

sents=[]
labels=[]
all_words=[]

for line in open(inp).readlines():
  line=line.replace('\n','').replace(',','').split('\t')
  words=line[0].lower().split(' ')
  all_words+=words
  sents.append(words)
  labels.append(int(line[1]))
  
print(sents)
print(set(all_words))
print(labels)
