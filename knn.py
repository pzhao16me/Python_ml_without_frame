import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
np.random.seed(123)
# make the dataset
# we will use the digits datasets as an examples. It consists of the 1797 images
# of hand_written digits .Each digit is  represented by a 64-dimensional vector of
# pixel values
digits = load_digits()
X,y = digits.data, digits.target
X_train,X_test,y_train,y_test = train_test_split(X,y)
# show the shape of the train and test datasets
print(f" shape of the X_train :{X_train.shape}")
print(f" shape of the y_train: {y_train.shape}")
print(f" shape of the X_test : {X_test.shape}")
print(f" shape of the y_test: {y.shape}")

# some example the digits

fig = plt.figure(figsize=(10,8))
for i in range(10):
	ax = fig.add_subplot(2,5,i+1)
	plt.imshow(X[i].reshape((8,8)),cmap='gray')
	plt.show()


class Knn():
	def __inti__(self):
		pass

	def fit(self,X,y):
		self.data = X
		self.targets = y
	def eculidean_distance(self,X):
		"""
		compute the euclidean distance between the training data and
		a new input example or matrix of input example X
		"""
		# input : single data point
		if X.ndim ==1:
			L2 = np.sqrt(np.sum((self.data -X)**2 ,axis =1))
		# input: matrix of data points
		if X.ndim ==2:
			n_sample,_ = X.shape
			L2 = [np.sqrt(np.sum((self.data - X[i])**2 ,axis =1)) for i in range(n_sample)]
		return np.array(L2)				

	def predict(self,X, k =1):
		"""
		Predict the classification for an input example or matrix of the input
		example X

		"""
		# step 1 comppute distance between input and training data
		dist = self.eculidean_distance(X)

		# step 2d find the k nearst neighbors and their classifications
		if X.ndim ==1:
			if k==1:
				nn = np.argmin(dist)
				return self.targets[nn]
			else:
				knn = np.argsort(dist)[:k]
				y_knn = self.targets[knn]
				max_vote = max(y_knn,key= list(y_knn).count)
				return max_vote

		if X.ndim ==2:
			knn = np.argsort(dist)[:,:k]
			y_knn = self.targets[knn]
			if k==1:
				return y_knn.T
			else:
				n_sample,_ = X.shape
				max_vote = [max(y_knn[i],key= list(y_knn[i]).count) for i in range(n_sample)]
				return max_vote



# init the model and train the model

knn = Knn()

knn.fit(X_train,y_train)

print("test one datapoint ,k=1")
print(f" predictt label :{knn.predict(X_test[0],k=1)}")
print(f" True label :{y_test[0]}")

print("#########################")

print("test one datapoint ,k=5")
print(f"predict label: {knn.predict(X_test[20],k=5)}")
print(f" true label:{y_test[20]}")


print("#########################")

print("testing 10 datapoints,k=1")
print(f" predict label: {knn.predict(X_test[5:15],k=1)}")
print(f" true labels: {y_test[5:15]}")
print("##############")	


print("testin 10 datapoints ,k=4")
print(f"predict labels :{knn.predict(X_test[5:15],k=4)}" )	

print(f"true labels :{y_test[5:15]}")		


# test the accuracy on test set
y_p_test1 = knn.predict(X_test,k=1)
test_acc1 = np.sum(y_p_test1[0]==y_test)/len(y_p_test1[0]) *100
print(f"test accuracy with k =1: {format(test_acc1)}")

#
print("test on the k =5")
y_p_test8 = knn.predict(X_test,k=5)
test_acc8 = np.sum(y_p_test8==y_test)/len(y_p_test8) *100
print(f"test accuracy with k=8:{format(test_acc8)}")