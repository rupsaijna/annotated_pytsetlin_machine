import pandas as pd
import re
regex=r"(?:\@|https?\://)\S+"
text = re.sub(regex, "", text)

df=pd.read_csv('training.1600000.processed.noemoticon.csv', header=None, names=['polarity', 'id','date','query','user','tweet'], encoding='latin-1')

allow_polarity=['0','2'] #change
fout='tweets_neutralnegative.csv'

print(df)

data=df.loc[df['polarity'].isin(allow_polarity)]

for ind, row in data.iterrows():
    text = re.sub(regex, "", row['tweet'])
    data[ind, 'tweet']=text
  

data.to_csv(fout, columns=['id','tweet','polarity'], index=False, sep='\t', quoting=2)
