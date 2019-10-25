import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori

input_clause='clause_details20191023-120835.txt'
data=pd.read_csv(input_clause, sep='\t', na_filter = False)
data=data.head(50)

dataset=[]
for ind, row in data.iterrows():
    cl=row['Clause'].split(';')[:-1]
    cl=[c.strip() for c in cl]
    dataset.append(cl)


'''
df_clause= data[['Clause']].copy()
df_clause['Extended']=''

##replacing feature_numbers with features
for index, row in df_clause.iterrows():
    cl=row['Clause']
    cl_list=cl.split(';')[:-1]
    ext_cl=[]
    for c in cl_list:
        if '#' not in c:
            ext_cl.append(str(df_features.loc[df_features['fnum'] == int(c),'feature'].item()))
        else:
             ext_cl.append('#'+str(df_features.loc[df_features['fnum'] == int(c.replace('#','')),'feature'].item()))
    df_clause.iloc[index]['Extended']=ext_cl

##Working with clauses with feature names
dataset=df_clause[['Extended']].values'''

##one hot encoding, sparse
te = TransactionEncoder()
'''te_ary = te.fit(dataset).transform(dataset) ##one_hot encoding
df = pd.DataFrame(te_ary, columns=te.columns_)'''


##sparse encoding
oht_ary = te.fit(dataset).transform(dataset, sparse=True)
sparse_df = pd.SparseDataFrame(oht_ary, columns=te.columns_, default_fill_value=False)

#print (sparse_df)

frequent_itemsets=apriori(sparse_df, min_support=0.2, use_colnames=True)
print(frequent_itemsets)

frequent_itemsets['Word_clause']=''

input_features='feature_details20191023-120835.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)

for idx, row in frequent_itemsets.iterrows():
    fi=row['itemsets']
    ext_cl=[]
    for c in fi:
        if '#' not in c:
            ext_cl.append(str(df_features.loc[df_features['fnum'] == int(c),'feature'].item()))
        else:
             ext_cl.append('#'+str(df_features.loc[df_features['fnum'] == int(c.replace('#','')),'feature'].item()))
    frequent_itemsets.iloc[idx,'Word_clause']=ext_cl
        
print(frequent_itemsets)
'''
##adding length filter
frequent_itemsets = apriori(df, min_support=0.8, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)
fi=frequent_itemsets[ (frequent_itemsets['length'] == 2) & (frequent_itemsets['support'] >= 0.8) ]
print(fi)
'''
