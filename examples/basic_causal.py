#identify if a sentence is causal or not (presence of causal connective)
inp='is_causal_data.txt'

sents=[]
labels=[]

for line in open(inp).readlines():
  line=line.replace('\n','').split('\t')
  print(line)
  words=line[0].split(' ')
  sents.append(words)
  labels.append(int(line[1]))
  
print(sent)
print(labels)
