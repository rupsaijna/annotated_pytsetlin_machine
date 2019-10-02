import sys
sys.path.append('../pyTsetlinMachine/')
from tm import MultiClassConvolutionalTsetlinMachine2D
#from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine2D
import numpy as np
from time import time
X = np.random.random_integers(0, 1, size=(10000, 3, 3))
Y = empty([10000])

Sum = 0
for i in range(10000):
    b = X[i,:,:].flatten()
    Sum = b[0]*8 + b[2]*4 + b[6]*2 + b[8]*1
    Y[i] = Sum
    Sum = 0
    
NOofTestingSamples = len(X)*20//100
NOofTrainingSamples = len(X)-NOofTestingSamples

X_train = X[0:NOofTrainingSamples,:,:].astype(dtype=np.int32)
Y_train = Y[0:NOofTrainingSamples].astype(dtype=np.float32)
X_test = X[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples,:,:].astype(dtype=np.int32)
Y_test = Y[NOofTrainingSamples:NOofTrainingSamples+NOofTestingSamples].astype(dtype=np.float32)

NUM_CLAUSES=15
CLASSES=list(set(Y_train)) #list of classes
THRESHOLD= 15
SP=2.0
RUN_EPOCHS=1
tm = MultiClassConvolutionalTsetlinMachine2D(NUM_CLAUSES, THRESHOLD, SP, (2, 2))

print("\nAccuracy over 1 epochs:\n")
for i in range(RUN_EPOCHS):
	start = time()
	tm.fit(X_train, Y_train, epochs=100, incremental=True)
	stop = time()
	result = 100*(tm.predict(X_test) == Y_test).mean()
	
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
