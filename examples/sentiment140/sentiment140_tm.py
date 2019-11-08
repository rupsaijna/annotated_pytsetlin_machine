#identify if a sentence is Positive or Negative
import pandas as pd
import sys
sys.path.append('../../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import stop_words
import numpy as np
from nltk.util import ngrams,everygrams
import re
import string
import time
timestr = time.strftime("%Y%m%d-%H%M%S")
clause_file='senti140_clause_details'+timestr+'.txt'
feature_file='senti140_feature_details'+timestr+'.txt'
meta_file='senti140_meta_details'+timestr+'.txt'

RUNS=1

inp='tweets_positivenegative.csv'

sents=[]
labels=[]
all_words=[]

def encode_sentences(txt):
	feature_set=np.zeros((len(txt), len(word_set)+1),dtype=int)
	tnum=0
	for t in txt:
		s_words=t[1:]+list(set(list(everygrams(t[1:], min_len=2,max_len=2))))
		for w in s_words:
			idx=word_idx[w]
			feature_set[tnum][idx]=1
		feature_set[tnum][-1]=t[0]
		tnum+=1
	return feature_set

df=pd.read_csv(inp,sep='\t', quoting=2, dtype={'id ':int,'polarity': int })
data=df.iloc[np.r_[0:2, -2:0]]

from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()

for ind, row in data.iterrows():
	tw=tw.lower()
	words=tknzr.tokenize(tw)
	bl=list(set(list(everygrams(words, min_len=2,max_len=2))))
	all_words+=words+bl
	words.insert(0,lcnt)
	sents.append(words)
	labels.append(row['polarity'])
	
word_set=set(all_words)
i=0
word_idx = dict((c, i + 1) for i, c in enumerate(word_set,start = -1))
reverse_word_map = dict(map(reversed, word_idx.items()))
data=encode_sentences(sents)

print(reverse_word_map)
print(data)
