import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassConvolutionalTsetlinMachine2D
#from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time

from keras.datasets import mnist
NUM_CLAUSES=10

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

CLASSES=list(set(Y_train)) #list of classes

X_train = np.where(X_train >= 75, 1, 0) 
X_test = np.where(X_test >= 75, 1, 0) 

#tm = MultiClassConvolutionalTsetlinMachine2D(8000, 200, 10.0, (10, 10))
tm = MultiClassConvolutionalTsetlinMachine2D(NUM_CLAUSES, 27, 15.0, (10, 10))

NUM_FEATURES=len(X_train[0])

print("\nAccuracy over 1 epochs:\n")
for i in range(1):
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))

	
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


for cur_cls in CLASSES:
	print('Class ',cur_cls)
	for cur_clause in range(NUM_CLAUSES):
		print (cur_clause,':',tm.get_typeII_clauses(int(cur_cls), cur_clause))
