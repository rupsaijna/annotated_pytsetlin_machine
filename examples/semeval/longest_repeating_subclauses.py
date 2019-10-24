import pandas as pd
input='clause_details20191022-155249.txt'

data=pd.read_csv(input, sep='\t', na_filter = False)

multiples={}
full_clauses_count=0

clauses=data['Clause']

print(clauses)
        
print(full_clauses_count)
print(multiples)
