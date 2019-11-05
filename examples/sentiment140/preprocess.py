import pandas as pd
import re
import string
from tqdm import tqdm
regex=r"(?:\@|https?\://)\S+"

df=pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, names=['polarity', 'id','date','query','user','tweet'], encoding='latin-1')

allow_polarity=['0','2'] #change
fout='tweets_neutralnegative.csv'

data=df.loc[df['polarity'].isin(allow_polarity)]

for ind, row in tqdm(data.iterrows(), total=data.shape[0]):
    text = re.sub(regex, "", row['tweet'])
    text=text.encode('ascii', 'ignore').decode('ascii')
    text=text.replace('...',' ELLIPSIS ')
    text=text.replace('..',' ELLIPSIS ')
    text=text.replace('  ',' ')
    text=text.strip()
    #text=text.translate(str.maketrans('','',string.punctuation))
    data.loc[ind, 'tweet']=text
  

data.to_csv(fout, columns=['id','tweet','polarity'], index=False, sep='\t', quoting=2)
