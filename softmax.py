from sklearn.datasets import load_iris
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

np.random.seed(13)
X, y_true = make_blobs(centers=4, n_samples=500)
fig = plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_true)
plt.title("datasets")
plt.xlabel('first feature')
plt.ylabel('second feature')
plt.show()

# reshape targets to get column vector with shape (n_samples,1)
y_true = y_true[:, np.newaxis]
# split the dataset into a train and test set
X_train, X_test, y_train, y_test = train_test_split(X, y_true)
print(f'the shape of the  X_train is :{X_train.shape}')
print(f" the shape of the X_test is :{X_test.shape}")
print(f"the shape of the y_train is :{y_train.shape}")
print(f" the shape of the y_test is :{y_test.shape}")


class SoftmaxRegressor:

    def __init__(self):
        pass

    def train(self, X, y_true, n_classes, n_iters=800, learning_rate=0.1):
        """
		Trains a multinomial logistic regressor model on given set of traing data
		"""
        self.n_samples, n_features = X.shape
        self.n_classes = n_classes
        self.weights = np.random.rand(self.n_classes, n_features)
        self.bias = np.zeros((1, n_classes))
        all_losses = []
        for i in range(n_iters):
            scores = self.compute_scores(X)
            probs = self.softmax(scores)
            y_predict = np.argmax(probs, axis=1)[:, np.newaxis]
            y_one_hot = self.one_hot(y_true)
            loss = self.cross_entropy(y_one_hot, probs)
            all_losses.append(loss)
            dw = (1 / self.n_samples) * np.dot(X.T, (probs - y_one_hot))
            db = (1 / self.n_samples) * np.sum(probs - y_one_hot, axis=0)
            self.weighlts = self.weights - learning_rate * dw.T
            self.bias = self.bias - learning_rate * db

            if i % 100 == 0:
                print(f"iteration number :{i},loss:{np.round(loss,4)}")
        return self.weights, self.bias, all_losses

    def predict(self, X):
        """
		predict  classes labels for samples in X
		Args:
			X: numpy array of shape (n_samples,n_features)
		Returns:
				numpy array of shape (n_samples,1) with predicted classes
		"""
        scores = self.compute_scores(X)
        probs = self.softmax(scores)
        return np.argmax(probs, axis=1)[:np.newaxis]

    def softmax(self, scores):
        exp = np.exp(scores)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def compute_scores(self, X):
        """
		compute class-scores for samples in X
		Args:
			X: numpy array of shape (n_samples,n_features)
		Returns:
			scores: numpy array of shape (n_samples,n_classes)
		"""
        return np.dot(X, self.weights.T) + self.bias

    def cross_entropy(self, y_true, scores):
        loss = -(1 / self.n_samples) * np.sum(y_true * np.log(scores))
        return loss

    def one_hot(self, y):
        """
		Transforms vector y of labels to one-hot encoded matrix
		"""
        one_hot = np.zeros((self.n_samples, self.n_classes))
        one_hot[np.arange(self.n_samples), y.T] = 1
        return one_hot


regressor = SoftmaxRegressor()

W_trained, b_trained, loss = regressor.train(X=X_train, y_true=y_train, learning_rate=0.7, n_iters=80000, n_classes=4)

fig = plt.figure(figsize=(8, 6))
plt.plot(np.arange(80000),loss)

plt.title("development of the loss during training")
plt.xlabel("number of iteration")
plt.ylabel('loss')
plt.show()
