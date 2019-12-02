import pandas as pd
#import types

#NumberTypes = (types.IntType, types.LongType, types.FloatType, types.ComplexType,np.float32)

#generic
word_dict={'negative':100,'positive':100,'anger':100,'sadness':100,'happiness':100,'fear':100, 'anticipation':100, 'trust':100, 'surprise':100, 'sadness':100, 'joy':100,'disgust':100}

#outputfilename="lexiconscompared.txt"
#fout = open(outputfilename,'w')

featurefilename=""


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
	dict_counts[row['file']]['in_features']=[]
      
########lex##############################

print(dict_df)
