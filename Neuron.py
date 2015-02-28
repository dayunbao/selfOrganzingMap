import numpy as np
import random
import math

class Neuron:
	def __init__(self, numWeights):
		self.posX = 0
		self.posY = 0
		self.numWeights = numWeights
		self.weights = []
		self.generateWeights()
		self.classLabel = ""

	def __str__(self):
		return "Position x: %d Position y: %d Weights: %s" % (self.posX, self.posY, self.weights)

	def __repr__(self):
		return "Position x: %d Position y: %d Weights: %s" % (self.posX, self.posY, self.weights)

	def generateWeights(self):
		for x in range(self.numWeights):
				self.weights.append(random.uniform(-1.0, 1.0))

	def adjustWeights(self, inputVec, learnRate, distFallOff):
		for x in range(len(self.weights)):
			self.weights[x] += (distFallOff * learnRate * (inputVec[x] - self.weights[x]))
		
