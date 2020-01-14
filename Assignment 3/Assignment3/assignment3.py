import numpy as np
import collections
import math
import random

class KNN:
	def __init__(self, k):
		self.k = k

	def distance(self, featureA, featureB):
		diffs = (featureA - featureB)**2
		return np.sqrt(diffs.sum())

	def train(self, X, y):
		self.X_Train = X
		self.y_Train = y

	def predict(self, X):

		lPredictionList = []

		for lXte in X:
			lNeighbourList = []
			for i in range(0,len(self.X_Train)):
				lNeighbourList.append((self.y_Train[i], self.distance(lXte, self.X_Train[i])))

			lNeighbourList.sort(key=lambda x: x[1])
			lNeighbourCounter = collections.Counter([x for (x, y) in lNeighbourList[:self.k]])
			lPrediction = max(lNeighbourCounter, key=lNeighbourCounter.get)
			lPredictionList.append(lPrediction)

		return np.asarray(lPredictionList)

class ID3:
	def __init__(self, nbins, data_range):
		self.bin_size = nbins
		self.range = data_range
		self.listOfNodes = [] # it will contain all the nodes
		# node structure is [TypeNo, list of child nodes per attribute]
		# if a node is child node then 100 is added to it to distinguish it from output node.

	def preprocess(self, data):
		norm_data = np.clip((data - self.range[0]) / (self.range[1] - self.range[0]), 0, 1)
		categorical_data = np.floor(self.bin_size*norm_data).astype(int)
		return categorical_data

	def train(self, X, y):
		categorical_data = self.preprocess(X)
		self.datalen = len(categorical_data)
		cy=0
		cn=0
		for out in y:
			if out == 0:
				cy = cy + 1
			elif out == 1:
				cn = cn + 1
		self.InfoD = self.calculateInfo(cy,cn)
		self.rootNode = self.getNextNode(categorical_data,y)
		#print(self.listOfNodes)

	def calculateInfoCategory(self, pC, pO):
		templist = []
		attrib = []
		for i in range(len(pC)):
			templist.append((pC[i][0],pO[i]))
			if(pC[i][0] not in attrib):
				attrib.append(pC[i][0])

		lCounter = collections.Counter([(x, y)  for (x, y) in templist])
	#	print(lCounter)
		lFinalInfo = 0
		for a in attrib:
			cya = lCounter[a,1]
			cna = lCounter[a,0]
			infoA = self.calculateInfo(cya, cna)
			infoA = ((cya+cna)/self.datalen) * infoA
			lFinalInfo = lFinalInfo + infoA

		return lFinalInfo,attrib

	def calculateInfo(self, pT, pF):
		if pT == 0 or pF == 0:
			return 0
		val = ((-pT/(pT+pF)) * math.log2(pT/(pT+pF))) + ((-pF/(pT+pF)) * math.log2(pF/(pT+pF)))
		return val

	def getCatData(self, pCategorical_data, pY, pIndex, pAttrib):
		ynew = np.copy(pY)
		catDataNew = np.copy(pCategorical_data)
		ynew = ynew[catDataNew[:, pIndex] == pAttrib]
		catDataNew = catDataNew[catDataNew[:, pIndex] == pAttrib]
		catDataNew = np.delete(catDataNew, pIndex, 1)
	#	print(len(catDataNew[catDataNew[:,pIndex] == pAttrib]))
		return catDataNew,ynew


	def getNextNode(self, pCategorical_data, pY):
		lGainList = []

		if len(np.unique(pY)) == 1:
			return np.unique(pY)[0]

		if(len(pCategorical_data[0]) == 1):
			return np.bincount(pY).argmax()

		for i in range(0,len(pCategorical_data[0])):
			lInfoA,lAttrib = self.calculateInfoCategory(pCategorical_data[:, i:i+1], pY)
			lGain = self.InfoD - lInfoA
			lGainList.append((i,lGain,lAttrib))

		lGainList.sort(key=lambda x: x[1], reverse=True)

		lChildList = []
		for a in lGainList[0][2]:
			lCatDataNew, lYNew = self.getCatData(pCategorical_data, pY,lGainList[0][0],a)
			cNode = self.getNextNode(lCatDataNew, lYNew)
			lChildList.append((a, cNode))

		self.listOfNodes.append((lGainList[0][0], lChildList))
		return 100 +(len(self.listOfNodes)-1)



	def getOutput(self, pX, root):
		cnode = self.listOfNodes[root-100]

		out = -1
		for l,m  in cnode[1]:
			if( pX[cnode[0]] == l ):
				out = m
				if out < 50:
					break
				else:
					out = self.getOutput(pX, out)
		if out == -1:
			out = random.randint(0, 1)
		return out


	def predict(self, X):
		categorical_data = self.preprocess(X)

		lPredictionList = []
		for i in range(0, len(categorical_data)):
			out = self.getOutput(categorical_data[i], self.rootNode)
			lPredictionList.append(out)
		return np.asarray(lPredictionList)


class Perceptron:
	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w
		self.b = b

	def train(self, X, y, steps):

		for i in range(0,steps):
			n = i % len(X)
			activation = sum(self.w * X[n]) + self.b
			if activation > 0:
				out = 1
			else:
				out = 0

			if out == y[n]:
				continue
			else:
				ld = y[n] - out
				self.w = self.w + self.lr*ld*X[n]

	def predict(self, X):
		lPredictionList = []
		for i in range(0, len(X)):
			activation = sum(self.w * X[i]) + self.b
			if activation > 0:
				out = 1
			else:
				out = 0
			lPredictionList.append(out)
		return np.asarray(lPredictionList)


class MLP:
	def __init__(self, w1, b1, w2, b2, lr):
		self.l1 = FCLayer(w1, b1, lr)
		self.a1 = Sigmoid()
		self.l2 = FCLayer(w2, b2, lr)
		self.a2 = Sigmoid()

	def MSE(self, prediction, target):
		return np.square(target - prediction).sum()

	def MSEGrad(self, prediction, target):
		return - 2.0 * (target - prediction)

	def shuffle(self, X, y):
		idxs = np.arange(y.size)
		np.random.shuffle(idxs)
		return X[idxs], y[idxs]

	def train(self, X, y, steps):
		for s in range(steps):
			i = s % y.size
			if(i == 0):
				X, y = self.shuffle(X,y)
			xi = np.expand_dims(X[i], axis=0)
			yi = np.expand_dims(y[i], axis=0)

			pred = self.l1.forward(xi)
			pred = self.a1.forward(pred)
			pred = self.l2.forward(pred)
			pred = self.a2.forward(pred)
			loss = self.MSE(pred, yi) 
			#print(loss)

			grad = self.MSEGrad(pred, yi)
			grad = self.a2.backward(grad)
			grad = self.l2.backward(grad)
			grad = self.a1.backward(grad)
			grad = self.l1.backward(grad)

	def predict(self, X):
		pred = self.l1.forward(X)
		pred = self.a1.forward(pred)
		pred = self.l2.forward(pred)
		pred = self.a2.forward(pred)
		pred = np.round(pred)
		return np.ravel(pred)

class FCLayer:

	def __init__(self, w, b, lr):
		self.lr = lr
		self.w = w	#Each column represents all the weights going into an output node
		self.b = b

	def forward(self, input):
		self.mInput = input
		fcfw = np.dot(self.mInput, self.w)
		fcfw = fcfw + self.b
		#print(fcfw)
		return fcfw

	def backward(self, gradients):
		lwdash = np.dot(np.transpose(self.mInput),gradients)
		lxdash = np.dot(gradients,np.transpose(self.w))
		self.w = self.w - ( self.lr * lwdash)
		self.b = self.b - ( self.lr * gradients)
		return lxdash

class Sigmoid:

	def __init__(self):
		return

	def forward(self, input):
		lermx = np.exp(-input)
		sfw = 1/(1+lermx)
		self.mInput = sfw
		return sfw

	def backward(self, gradients):
		sbw = (1-self.mInput)*self.mInput*gradients
		return sbw