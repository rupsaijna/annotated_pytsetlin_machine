import pickle

f = open("dict_cnts.pkl","rb")
newdict=pickle.load(f)
cntdict={}


for d in newdict:
	cntdict[d]={}
	cntdict[d]['in_text_cnt']=len(set(newdict[d]['in_text']))
	#print('\n',d,newdict[d])

	
print(cntdict)
