import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassTsetlinMachine
#from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 

#Parameters for the TM
NUM_CLAUSES=10
THRESHOLD=15
S=3.9

#Load Training Data
train_data = np.loadtxt("NoisyXORTrainingData.txt")
X_train = train_data[:,0:-1] #last column is class labels
Y_train = train_data[:,-1] #last column is class labels

CLASSES=list(set(Y_train)) #list of classes

#Load Testing Data
test_data = np.loadtxt("NoisyXORTestData.txt")
X_test = test_data[:,0:-1] #last column is class labels
Y_test = test_data[:,-1]  #last column is class labels

#Initialize the Tsetlin Machine
tm = MultiClassTsetlinMachine(NUM_CLAUSES, THRESHOLD, S, boost_true_positive_feedback=0)

#Fit TM on training data
tm.fit(X_train, Y_train, epochs=1)

#Predict on test data, compare to ground truth, calculate accuracy0
print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())


#Prediction on some random data
print("Prediction: x1 = 1, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[1,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[0,1,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 0, x2 = 0, ... -> y = %d" % (tm.predict(np.array([[0,0,1,0,1,0,1,1,1,1,0,0]]))))
print("Prediction: x1 = 1, x2 = 1, ... -> y = %d" % (tm.predict(np.array([[1,1,1,0,1,0,1,1,1,1,0,0]]))))

print("\nLet's try to get clauses....")

NUM_FEATURES=len(X_train[0])

print('Num Clauses:', NUM_CLAUSES)
print('Num Classes: ', len(CLASSES),' : ', CLASSES)
print('Num Features: ', NUM_FEATURES)

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
