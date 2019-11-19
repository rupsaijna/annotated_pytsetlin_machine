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

print(X_train.shape)
NUM_FEATURES=len(X_train[0])
print(NUM_FEATURES)
print("\nAccuracy over 1 epochs:\n")
for i in range(1):
	'''
	start = time()
	tm.fit(X_train, Y_train, epochs=1, incremental=True)
	stop = time()
	tm.save_model('mnist_model.npz', Y_train)
	print('saved')
	result = 100*(tm.predict(X_test) == Y_test).mean()
	print("#%d Accuracy: %.2f%% (%.2fs)" % (i+1, result, stop-start))
	'''
	tm2=MultiClassConvolutionalTsetlinMachine2D.load_model('mnist_model.npz', X_train, Y_train)
	result2= 100*(tm2.predict(X_test) == Y_test).mean()
	print("Accuracy2: %.2f%%" % (result2))
