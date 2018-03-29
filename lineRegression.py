import numpy as np
import matplotlib.pyplot as plt
from  sklearn.model_selection import train_test_split
np.random.seed(123)

# we will use a simple training set
X = 2 * np.random.rand(500,1)
y = 5 + 3 * X + np.random.randn(500,1)

fig = plt.figure(figsize =(8,6))
plt.scatter(X,y)
plt.title("DataSet")
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()

# split the data  into the  training and test set
X_train,X_test, y_train,y_test =  train_test_split(X,y)

print(f"shape X_train :{X_train.shape}")
print(f"shape y_train: {y_train.shape}")
print(f"shape X_test: {X_test.shape}")
print(f"shape y_test: {y_test.shape}")


class LinearRegression:
	def __init__(self):
		pass

	def train_gradient_descent(self,X,y,learning_rate =0.01,N_iters = 100):
		"""
		train a linear regression model using gradient descent

		"""
		# step 0 Initialize the parameters
		n_sample,n_features = X.shape
		self.weights = np.zeros(shape= (n_features,1))
		self.bias = 0
		costs= []

		for i in  range(N_iters):
			# step 1 compute a linear combination of the input features ,weights and bias
			y_predict = np.dot(X,self.weights) + self.bias

			# step 2 compute cost over training set
			cost = (1/n_sample) * np.sum((y_predict- y)**2)
			costs.append(cost)

			if i%100 ==0:
				print(f"cost at iteration {i}:{cost}")

			# step 3  compute the gradients
			dJ_dw = (2/n_sample) * np.dot(X.T,(y_predict-y))
			dJ_db = (2/n_sample) * np.sum((y_predict-y))

			# step 4 update the  parameters
			self.weights = self.weights - learning_rate * dJ_dw
			self.bias = self.bias - learning_rate * dJ_db

		return self.weights,self.bias,costs


	def train_normal_equation(self,X,y):
		"""
		 train  a linear regression model using the normal equation
		"""
		self.weights = np.dot(np.dot(np.linalg.inv(np.dot(X.T,X)),X.T),y)
		self.bias =0
		return self.weights,self.bias

	def predict(self,X):
		return np.dot(X,self.weights,) + self.bias




regressor = LinearRegression()
W_trained, b_trained,costs = regressor.train_gradient_descent(X_train,y_train,learning_rate=0.005,
								N_iters = 600)
fig = plt.figure(figsize=(8,6))
plt.plot(np.arange(600),costs)	
plt.title("Development of cost during training")
plt.xlabel("number of iteration")
plt.ylabel("cost")
plt.show()			

#   test

n_sample, _ = X_train.shape
n_sample_test ,_ = X_test.shape

y_p_train = regressor.predict(X_train)

y_p_test = regressor.predict(X_test)

error_train = (1/n_sample) * np.sum((y_p_train -y_train)**2)

error_test = (1/n_sample_test)* np.sum((y_p_test-y_test)**2)


print(f"error of training set : {np.round(error_train,4)}")

print(f" error of the test  set : {np.round(error_test,4)}")

print("begin the normal equation::")



#  to computer  the parameters  using the  normal  equation ,we add a bias value of 1 to  the input  example
X_b_train = np.c_[np.ones((n_sample)),X_train]
X_b_test = np.c_[np.ones((n_sample_test)),X_test]

reg_normal = LinearRegression()

w_trained2 = reg_normal.train_normal_equation(X_b_train,y_train)


# test
y_p_train =reg_normal.predict(X_b_train)
y_p_predict =reg_normal.predict(X_b_test)
error_train = (1/n_sample) * np.sum((y_p_train -y_train)**2)

error_test = (1/n_sample_test)* np.sum((y_p_predict-y_test)**2)


print(f"error of training set : {np.round(error_train,4)}")

print(f" error of the test  set : {np.round(error_test,4)}")



#  plot the  test the predictions
fig = plt.figure(figsize =(16,12))
plt.scatter(X_train,y_train)
plt.scatter(X_test,y_p_test)
plt.xlabel("First feature")
plt.ylabel("Second feature")
plt.show()