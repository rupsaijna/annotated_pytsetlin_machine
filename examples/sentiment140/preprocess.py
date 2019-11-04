import pandas as pd

df=pd.read_csv('training.1600000.processed.noemoticon.csv', header=['polarity', 'id','date','query','user','tweet'])

allow_polarity=['0','2'] #change
fout='tweets_neutralnegative.csv'

print(df)

data=df.loc[df['polarity'].isin(allow_polarity)]

data=data['id','tweet','polarity'].copy()

print(data)
data.to_csv(fout, index=False)
