from pyTsetlinMachine.tm import MultiClassTsetlinMachine
import numpy as np 

data=np.loadtxt("NoisyXORTestData.txt")

numpy.random.shuffle(data)
training, test = data[:80,:], data[80:,:]


X_train=training[0:2,0:-1]
Y_train = training[:,-1]

X_test=test[0:2,0:-1]
Y_test = test[:,-1]


tm = MultiClassTsetlinMachine(10, 15, 3.9, boost_true_positive_feedback=0)

tm.fit(X_train, Y_train, epochs=200)

print("Accuracy:", 100*(tm.predict(X_test) == Y_test).mean())
