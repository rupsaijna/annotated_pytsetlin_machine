import pandas as pd
input='clause_details20191022-155249.txt'

data=pd.read_csv(input, sep='\t')

multiples={}
full_clauses_count=0
print(data[1])
mlk
for index, row in data.iterrows():
    if row['count']>50:
        print(row)
        multiples[row['Clause']]=row['count']
        full_clauses_count+=1
        
print(full_clauses_count)
print(multiples)
