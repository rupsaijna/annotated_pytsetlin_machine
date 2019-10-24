import pandas as pd
input='clause_details20191022-155249'

data=pd.read_csv(input, sep='\t')

multiples={}
full_clauses_count=0

for index, row in df.iterrows():
    if row['count']>50:
        multiples[row['Clause']]=row['count']
        full_clauses_count+=1
        
print(full_clauses_count)