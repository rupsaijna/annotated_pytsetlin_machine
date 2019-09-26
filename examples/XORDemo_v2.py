import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
import numpy as np 

#Parameters for the TM
NUM_CLAUSES=10
THRESHOLD=15
S=3.9

data=np.loadtxt("NoisyXORTestData.txt")

np.random.shuffle(data)
training, test = data[:80,:], data[80:,:]


X_train=training[0:2,0:-1]
Y_train = training[:,-1]

X_test=test[0:2,0:-1]
Y_test = test[:,-1]

CLASSES=list(set(Y_train)) #list of classes
NUM_FEATURES=len(X_train[0]) #number of features


tm = MultiClassTsetlinMachine(NUM_CLAUSES, THRESHOLD, S, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test))


for cur_clause in range(NUM_CLAUSES):
	for cur_cls in CLASSES:
		this_clause=''
		for f in range(NUM_FEATURES*2):
			action = tm.ta_action(int(cur_cls), cur_clause, f)
			if action==1:
				if this_clause!='':
					this_clause+='AND '
				if f<NUM_FEATURES:
					this_clause+='F'+str(f)+' '
				else:
					this_clause+='-|F'+str(f-NUM_FEATURES)+' '

		print('CLASS :',cur_cls,' - CLAUSE ',cur_clause, ' : ', this_clause)
	print('\n\n')


for cur_cls in CLASSES:
	print('Class ',cur_cls)
	for cur_clause in range(NUM_CLAUSES):
		print (cur_clause,':',tm.get_typeII_clauses(int(cur_cls), cur_clause))
