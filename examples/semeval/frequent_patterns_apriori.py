import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

#fp='data/productproducer/productproducer_'
fp='../sentiment140/senti140_'
file_date='20191108-133244'

input_clause=fp+'clause_details'+file_date+'.txt'
data=pd.read_csv(input_clause, sep='\t', na_filter = False)

#data=data.head(500)

dataset=[]
for ind, row in data.iterrows():
    cl=row['Clause'].split(';')[:-1]
    cl=[c.strip() for c in cl]
    dataset.append(cl)

##one hot encoding, sparse
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset) ##one_hot encoding
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df)

'''##sparse encoding
oht_ary = te.fit(dataset).transform(dataset, sparse=True)
sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)
#print (sparse_df)
'''

frequent_itemsets=apriori(df, min_support=0.02, use_colnames=True)

frequent_itemsets = frequent_itemsets.sort_values(by='support', ascending=False)
print(frequent_itemsets)
frequent_itemsets['Word_clause']=''

input_features=fp+'feature_details'+file_date+'.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)

for idx, row in frequent_itemsets.iterrows():
    fi=row['itemsets']
    ext_cl=[]
    for c in fi:
        if '#' not in c:
            ext_cl.append(str(df_features.loc[df_features['fnum'] == int(c),'feature'].item()))
        else:
             ext_cl.append('#'+str(df_features.loc[df_features['fnum'] == int(c.replace('#','')),'feature'].item()))
    frequent_itemsets.at[idx,'Word_clause']=ext_cl
        
#print(frequent_itemsets)

with open(fp+'meta_details'+file_date+'.txt') as f:
    lines = f.readlines()
    with open(fp+'frequent_itemsets_details'+file_date+'.csv', 'w') as f1:
        f1.writelines(lines)
f1= open(fp+'frequent_itemsets_details'+file_date+'.csv', 'a+')
f1.write('\n\n')
f1.close()
          
with open(fp+'frequent_itemsets_details'+file_date+'.csv', 'a') as f:
    frequent_itemsets.to_csv(f, sep='\t')
    
    
'''
##adding length filter
frequent_itemsets = apriori(df, min_support=0.8, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
fi=frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.8) ]
print(fi)
'''
