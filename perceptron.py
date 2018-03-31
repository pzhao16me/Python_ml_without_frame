import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
np.random.seed(123)

# dataset

X ,y = make_blobs(n_samples = 1000, centers =2)
fig = plt.figure(figsize=(8,6))
plt.scatter(X[:,0],X[:,1],c= y)
plt.title("datasets")
plt.xlabel("first features")
plt.ylabel("second features")
plt.show()


y_true = y[:,np.newaxis]


X_train,X_test,y_train,y_test = train_test_split(X,y_true)

print(f" shape of the X_train:{X_train.shape}")
print(f" shape of the y_train: {y_train.shape}")
print(f" shape of the X_test: {X_test.shape}")
print(f" shape of the y_test: {y_test.shape}")

class Perceptron():

	def __init__(self):
		pass

	def train(self,X,y,learning_rate = 0.5, n_iters = 100):
		n_samples, n_features = X.shape

		# step 0 Initialize the parameters
		self.weights = np.zeros((n_features,1))
		self.bias = 0

		for i in range(n_iters):
			# step 1 compute the activation
			a = np.dot(X,self.weights) + self.bias
			# step 2 compute the activation
			y_predict = self.step_function(a)

			# step 3 compute weight updates
			delta_w =  learning_rate*np.dot(X.T,(y-y_predict))
			delta_b = learning_rate*np.sum(y- y_predict)

			# step 44 update hthe parameters
			self.weights += delta_w
			self.bias ++ delta_b
		return self.weights,self.bias

	def step_function(self,x):
		return np.array([1 if elem >=0 else 0 for elem in x])[:,np.newaxis]
	
	def predict(self,X):
		a = np.dot(X,self.weights) + self.bias
		return self.step_function(a)




Perceptron1 = Perceptron()

w_trained,b_trained = Perceptron1.train(X_train,y_train,learning_rate= 0.05,n_iters=600)

# test

y_p_train = Perceptron1.predict(X_train)
y_p_test = Perceptron1.predict(X_test)

print(f"training accuracy: {100- np.mean(np.abs(y_p_train- y_train))*100}%")
print(f" test accuracy: {100 -np.mean(np.abs(y_p_test- y_test))*100}%")	


def  plot_hyperplane(X,y,weights,bias):

	"""
	plots the dataset and the estimated decision hyperplane
	"""

	slope = -weights[0]/weights[1]
	intercept =- bias/weights[1]
	x_hyperplane = np.linspace(-10,10,10)
	y_hyperplane = slope * x_hyperplane + intercept
	fig = plt.figure(figsize=(8,6))
	plt.scatter(X[:,0],X[:,1],c=y)
	plt.plot(x_hyperplane,y_hyperplane,"-")
	plt.title("dataset and fitted decision hyperplane")
	plt.xlabel("First feature")
	plt.ylabel(" second feature")
	plt.show()


plot_hyperplane(X,y,w_trained,b_trained)