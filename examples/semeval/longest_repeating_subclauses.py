import pandas as pd
input='clause_details20191022-155249.txt'

data=pd.read_csv(input, sep='\t', na_filter = False)

multiples={}
full_clauses_count=0

clauses=data['Clause'].values()

def extend_clauses(clauses):
  for c in clauses:
    c_list=c.split(';')
    for i in range(0,num_clauses):
      if str(i) not in c_list:
        if i==0:
          c_list.insert(0,'*'+str(i))
        if i>0:
          try:
            prev_loc=c_list.index(str(i-1))
          except:
            prev_loc=c_list.index('*'+str(i-1))
          c_list.insert(prev_loc+1,'*'+str(i))
    for i in range(0,num_clauses):
      if '#'+str(i) not in c_list:
        if i==0:
          c_list.insert(num_clauses+0,'*#'+str(i))
        if i>0:
          try:
            prev_loc=c_list.index('#'+str(i-1))
          except:
            prev_loc=c_list.index('*#'+str(i-1))
          c_list.insert(prev_loc+1,'*#'+str(i))
    extended_clauses.append(';'.join(c_list))
    return extended_clauses     
    
