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

print(X_train.shape)
print(len(Y_train))
'''
#Initialize the Tsetlin Machine
tm = MultiClassTsetlinMachine(NUM_CLAUSES, THRESHOLD, S, boost_true_positive_feedback=0)

#Fit TM on training data
tm.fit(X_train, Y_train, epochs=1)

##save
tm.save_model('tm_model.npz')

#tm2=MultiClassTsetlinMachine.load_model('tm_model.npz')


#Predict on test data, compare to ground truth, calculate accuracy0
#print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
#print("Accuracy after saving:", 100*(tm2.predict(X_test) == Y_test).mean())'''
