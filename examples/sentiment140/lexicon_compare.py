import pandas as pd
#import types
#NumberTypes = (types.IntType, types.LongType, types.FloatType, types.ComplexType,np.float32)
fp='../sentiment140/senti140_'
file_date='20191108-133244'
input_features=fp+'feature_details'+file_date+'.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)
input_clauses=fp+'clause_details'+file_date+'.txt'
df_clauses=pd.read_csv(input_clauses, sep='\t', na_filter = False).head(1)
df_clause_positive=df_clauses.loc[df_clauses['p/n'] == 'positive'].copy()
df_clause_negative=df_clauses.loc[df_clauses['p/n'] == 'negative'].copy()


#generic
word_dict={'negative':100,'positive':100,'anger':100,'sadness':100,'happiness':100,'fear':100, 'anticipation':100, 'trust':100, 'surprise':100, 'sadness':100, 'joy':100,'disgust':100}

#outputfilename="lexiconscompared.txt"
#fout = open(outputfilename,'w')

########lex##############################
print ("Loading lexicons...")

lex_files=pd.read_csv("lexicon/the_lexicon_guide.txt",sep='\t',header=0,names=['file','tot','word','sp','hd'])
print(lex_files)

dict_df={}
dict_counts={}
t=0
for ind,row in lex_files.iterrows():
	print (row)
	t+=row['tot']-1
	if row['hd']==1:
		if row['sp']=="','":
			dict_df[row['file']]=pd.read_csv(row['file'],header=0).fillna('')
		else:
			dict_df[row['file']]=pd.read_csv(row['file'],sep='\t',header=0).fillna('')
	else:
		if row['sp']=="','":
			dict_df[row['file']]=pd.read_csv(row['file'],header=None).fillna('')
		else:
			dict_df[row['file']]=pd.read_csv(row['file'],sep='\t',header=None).fillna('')
	dict_counts[row['file']]['in_text']=[]
	dict_counts[row['file']]['in_positive_features']=[]
	dict_counts[row['file']]['in_negative_features']=[]
      
########lex##############################

covered=[]
for idx, row in df_clause_positive.iterrows():
	cl=row['Clause'].split(';')[:-1]
    cl=[c.strip() for c in cl]
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except
				word_feature=word_feature
			covered.append(word_feature)
			for l in dict_df:
				df=dict_df[l]
				det=lex_files[lex_files['file']==l]
				words_in_det=list(det['word'].values)
				if word_feature in words_in_det:
					dict_counts[l]['in_positive_features'].append(word_feature)

covered=[]
for idx, row in df_clause_negative.iterrows():
	cl=row['Clause'].split(';')[:-1]
    cl=[c.strip() for c in cl]
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except
				word_feature=word_feature
			covered.append(word_feature)
			for l in dict_df:
				df=dict_df[l]
				det=lex_files[lex_files['file']==l]
				words_in_det=list(det['word'].values)
				if word_feature in words_in_det:
'''					dict_counts[l]['in_negative_features'].append(word_feature)
covered=[]
for idx, row in df_features.iterrows():
	cl=row['feature']
	if this_feature!='':
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except
				word_feature=word_feature
			covered.append(word_feature)
			for l in dict_df:
				df=dict_df[l]
				det=lex_files[lex_files['file']==l]
				words_in_det=list(det['word'].values)
				if word_feature in words_in_det:
					dict_counts[l]['in_text'].append(word_feature)
'''
print(dict_counts)
