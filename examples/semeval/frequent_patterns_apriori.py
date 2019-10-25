import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

input_clause='clause_details20191023-120835.txt'
data=pd.read_csv(input_clause, sep='\t', na_filter = False)
df_clause= data[['Clause']].copy()

input_features='feature_details20191023-120835.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)
#df_clause['Extended']=[]

for index, row in df_clause.iterrows():
    cl=row['Clause']
    cl_list=cl.split(';')[:-1]
    ext_cl=[]
    for c in cl_list:
        if '#' not in c:
            ext_cl.append(str(df_features.loc[df_features['fnum'] == int(c)]))
        else:
             ext_cl.append('#'+str(df_features.loc[df_features['fnum'] == int(c.replace('#',''))]))
    df_clause.iloc[index]['Extended']=ext_cl
    break
    
print(df_clause)
jhfrgk

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
frequent_itemsets=apriori(df, min_support=0.6, use_colnames=True)
print(frequent_itemsets)

fi=frequent_itemsets[0]
print('Frequent Itemset',fi)



'''frequent_itemsets = apriori(df, min_support=0.6, use_colnames=True)
frequent_itemsets['length'] = frequent_itemsets['itemsets'].apply(lambda x: len(x))
print(frequent_itemsets)'''
