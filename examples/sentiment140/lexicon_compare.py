import pandas as pd
import string
#import types
#NumberTypes = (types.IntType, types.LongType, types.FloatType, types.ComplexType,np.float32)
fp='../sentiment140/senti140_'
file_date='20191108-133244'
input_features=fp+'feature_details'+file_date+'.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)
input_clauses=fp+'clause_details'+file_date+'.txt'
df_clauses=pd.read_csv(input_clauses, sep='\t', na_filter = False).head(5)
df_clause_positive=df_clauses.loc[df_clauses['p/n'] == 'positive'].copy()
df_clause_negative=df_clauses.loc[df_clauses['p/n'] == 'negative'].copy()

print(df_clauses)
#generic
word_dict={'negative':100,'positive':100,'anger':100,'sadness':100,'happiness':100,'fear':100, 'anticipation':100, 'trust':100, 'surprise':100, 'sadness':100, 'joy':100,'disgust':100}

#outputfilename="lexiconscompared.txt"
#fout = open(outputfilename,'w')

########lex##############################
print ("Loading lexicons...")

lex_files=pd.read_csv("lexicon/the_lexicon_guide.txt",sep='\t',header=0,names=['file','tot','word','sp','hd'])

dict_df={}
dict_counts={}
t=0
for ind,row in lex_files.iterrows():
	t+=row['tot']-1
	if row['hd']==1:
		if row['sp']=="','":
			tempdf=pd.read_csv(row['file'],header=0).fillna('')
		else:
			tempdf=pd.read_csv(row['file'],sep='\t',header=0).fillna('')
	else:
		if row['sp']=="','":
			tempdf=pd.read_csv(row['file'],header=None).fillna('')
		else:
			tempdf=pd.read_csv(row['file'],sep='\t',header=None).fillna('')
	word_location=row['word']
	word_list=list(tempdf.iloc[:,word_location].values)
	word_list=[e.translate(str.maketrans('','',string.punctuation)).strip() for e in word_list]
	dict_df[row['file']]=word_list
	dict_counts[row['file']]={}
	dict_counts[row['file']]['in_text']=[]
	dict_counts[row['file']]['in_positive_features']=[]
	dict_counts[row['file']]['in_negative_features']=[]
	dict_counts[row['file']]['in_partial_positive_features']=[]
	dict_counts[row['file']]['in_partial_negative_features']=[]
      
########lex end##############################

covered=[]
for idx, row in df_clause_positive.iterrows():
	cl=row['Clause'].split(';')[:-1]
	cl=[c.strip() for c in cl]
	print('\nPos Clause:',cl)
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except:
				word_feature=word_feature
			covered.append(this_feature)
			word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
			print('Feature:',word_feature)
			added=0
			for l in dict_df:
				words_in_det=dict_df[l]
				if word_feature in words_in_det:
					dict_counts[l]['in_positive_features'].append(word_feature)
					print('added')
					added=1
			if added==0 and ' ' in word_feature:
				wf=word_feature.split(' ')
				for eachword in wf:
					for l in dict_df:
						words_in_det=dict_df[l]
						if eachword in words_in_det:
							dict_counts[l]['in_partial_positive_features'].append(eachword)

covered=[]
for idx, row in df_clause_negative.iterrows():
	cl=row['Clause'].split(';')[:-1]
	cl=[c.strip() for c in cl]
	print('\nNeg Clause:',cl)
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except:
				word_feature=word_feature
			covered.append(this_feature)
			word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
			print('Feature:',word_feature)
			added=0
			for l in dict_df:
				words_in_det=dict_df[l]
				if word_feature in words_in_det:
					dict_counts[l]['in_negative_features'].append(word_feature)
					added=1
			if added==0 and ' ' in word_feature:
				wf=word_feature.split(' ')
				for eachword in wf:
					for l in dict_df:
						words_in_det=dict_df[l]
						if eachword in words_in_det:
							dict_counts[l]['in_partial_negative_features'].append(eachword)

df_features=df_features.head(5)
print(df_features)
for idx, row in df_features.iterrows():
	cl=row['feature']
	if this_feature!='':
		this_feature=this_feature.replace('#','')
		word_feature=this_feature
		try:
			b=eval(word_feature)
			if type(b) is tuple:
				word_feature=' '.join(b)
		except:
			word_feature=word_feature
		print('Feature:',word_feature)
		word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
		for l in dict_df:
			words_in_det=dict_df[l]
			if word_feature in words_in_det:
				dict_counts[l]['in_text'].append(word_feature)

for d in dict_counts:
	print('\n',d,dict_counts[d])
