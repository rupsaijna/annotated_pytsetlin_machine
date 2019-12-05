import pandas as pd
import string
import pickle
#import types
#NumberTypes = (types.IntType, types.LongType, types.FloatType, types.ComplexType,np.float32)
fp='../sentiment140/senti140_'
file_date='20191108-133244'
input_features=fp+'feature_details'+file_date+'.txt'
df_features=pd.read_csv(input_features, sep='\t', na_filter = False)
input_clauses=fp+'clause_details'+file_date+'.txt'
df_clauses=pd.read_csv(input_clauses, sep='\t', na_filter = False)
df_clause_positive=df_clauses.loc[df_clauses['p/n'] == 'positive'].copy()
df_clause_negative=df_clauses.loc[df_clauses['p/n'] == 'negative'].copy()

#generic
word_dict={'negative':100,'positive':100,'anger':100,'sadness':100,'happiness':100,'fear':100, 'anticipation':100, 'trust':100, 'surprise':100, 'sadness':100, 'joy':100,'disgust':100}

#outputfilename="lexiconscompared.txt"
#fout = open(outputfilename,'w')

########lex##############################
print ("Loading lexicons...")

lex_files=pd.read_csv("lexicon/the_lexicon_guide.txt",sep='\t',header=0,names=['file','tot','word','sp','hd'])

dict_df={}
dict_counts={}

tempdf=pd.read_csv('lexicon/lexicons_compiled.csv',sep='\t').fillna('')
word_location=0

df_mpqa=tempdf.loc[tempdf['source']=='mpqa'].copy()
word_list_mpqa=list(df_mpqa.iloc[:,word_location].values)
word_list_mpqa=[e.translate(str.maketrans('','',string.punctuation)).strip() for e in word_list_mpqa]
dict_df['mpqa']=word_list
dict_counts['mpqa']={}
dict_counts['mpqa']['in_text']=[]
dict_counts['mpqa']['in_positive_features']=[]
dict_counts['mpqa']['in_negative_features']=[]
dict_counts['mpqa']['in_partial_positive_features']=[]
dict_counts['mpqa']['in_partial_negative_features']=[]


df_opinion=tempdf.loc[tempdf['source']=='opinion'].copy()
word_list_opinion=list(df_opinion.iloc[:,word_location].values)
word_list_opinion=[e.translate(str.maketrans('','',string.punctuation)).strip() for e in word_list_opinion]
dict_df['opinion']=word_list
dict_counts['opinion']={}
dict_counts['opinion']['in_text']=[]
dict_counts['opinion']['in_positive_features']=[]
dict_counts['opinion']['in_negative_features']=[]
dict_counts['opinion']['in_partial_positive_features']=[]
dict_counts['opinion']['in_partial_negative_features']=[]


########lex end##############################

covered=[]
for idx, row in df_clause_positive.iterrows():
	cl=row['Clause'].split(';')[:-1]
	cl=[c.strip() for c in cl]
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			print(this_feature)
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except:
				word_feature=word_feature
			covered.append(this_feature)
			word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
			added=0
			for l in dict_df:
				words_in_det=dict_df[l]
				if word_feature in words_in_det:
					dict_counts[l]['in_positive_features'].append(word_feature)
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
	for this_feature in cl:
		this_feature=this_feature.replace('#','')
		if this_feature not in covered:
			print(this_feature)
			word_feature=str(df_features.loc[df_features['fnum'] == int(this_feature),'feature'].item())
			try:
				b=eval(word_feature)
				if type(b) is tuple:
					word_feature=' '.join(b)
			except:
				word_feature=word_feature
			covered.append(this_feature)
			word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
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


for idx, row in df_features.iterrows():
	this_feature=row['feature']
	if this_feature!='':
		this_feature=this_feature.replace('#','')
		word_feature=this_feature
		try:
			b=eval(word_feature)
			if type(b) is tuple:
				word_feature=' '.join(b)
		except:
			word_feature=word_feature
		word_feature=word_feature.translate(str.maketrans('','',string.punctuation)).strip()
		for l in dict_df:
			words_in_det=dict_df[l]
			if word_feature in words_in_det:
				dict_counts[l]['in_text'].append(word_feature)

				
for d in dict_counts:
	print(d, dict_counts[d])
				
'''f = open("dict_cnts2.pkl","wb")
pickle.dump(dict_counts,f)
f.close()
'''
