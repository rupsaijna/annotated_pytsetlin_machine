import pickle

f = open("dict_cnts.pkl","rb")
newdict=pickle.load(f)
cntdict={}


for d in newdict:
	cntdict[d]={}
	cntdict[d]['in_text_cnt']=len(set(newdict[d]['in_text']))
	cntdict[d]['in_positive_features_cnt']=len(set(newdict[d]['in_positive_features']))
	cntdict[d]['in_negative_features_cnt']=len(set(newdict[d]['in_negative_features']))
	cntdict[d]['in_all_features_cnt']=len(set(newdict[d]['in_negative_features']+newdict[d]['in_positive_features']))
	cntdict[d]['text_positive_features_cnt']=len(set(newdict[d]['in_text']).intersection(set(newdict[d]['in_positive_features'])))
	cntdict[d]['text_negative_features_cnt']=len(set(newdict[d]['in_text']).intersection(set(newdict[d]['in_negative_features'])))
	
	
	#print('\n',d,newdict[d])

	
print(cntdict)
