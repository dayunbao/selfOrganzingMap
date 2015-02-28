from Neuron import Neuron
import numpy as np
import math
import operator
import csv
import random

class Lattice:
	def __init__(self, row, col, numWeights, numIterations):
		self.row = row
		self.col = col
		self.numWeights = numWeights
		self.featureMap = []
		self.dataSet = []
		#numIter is number of epochs
		self.numIter = numIterations
		#currIter is counter for number of epochs, used in calculation
		self.currIter = 0
		#this value stays fixed
		self.initLearnRate = 0.1
		#this value will decrease over distance
		self.learnRate = self.initLearnRate
		self.latticeRadius = (max(row, col) / 2)
		#self.time = 0
		self.timeConst = (self.numIter / math.log(self.latticeRadius))
		self.neighborRadius = 0
		#self.bmu = []
		self.createLattice()

	def createLattice(self):
		for x in range(self.row):
			temp = []
			for y in range(self.col):
				temp.append(Neuron(self.numWeights))
			self.featureMap.append(temp)

		for x in range(len(self.featureMap)):
			for y in range(len(self.featureMap)):
				self.featureMap[x][y].posX = x 
				self.featureMap[x][y].posY = y 

	def loadData(self):
		with open('bezdekIris.data', 'rb') as csvfile:
			lines = csv.reader(csvfile)
			self.dataSet = list(lines)

		for x in range(len(self.dataSet)): 
			for y in range(self.numWeights):
				self.dataSet[x][y] = float(self.dataSet[x][y])
		random.shuffle(self.dataSet)

	def getInput(self, index):
		return self.dataSet[index]
		

	def distance(self, inputVec, weightVec):
		distance = 0.0 
		#print	"input vector %s" % inputVec
		#print "Size of inpuVec %d" % len(inputVec)
		for x in range(len(inputVec) - 1):
			distance += math.pow((inputVec[x] - weightVec[x]), 2)
		
		return math.sqrt(distance)

	def findBMU(self, inputVec):
		distances = []

		for x in range(len(self.featureMap)):
			for y in range(len(self.featureMap)):
				#print "Input Vector: %s" % inputVec
				#print "Weights %s" % self.featureMap[x][y].weights
				eucDist = self.distance(inputVec, self.featureMap[x][y].weights)
				distances.append((eucDist, self.featureMap[x][y].posX, self.featureMap[x][y].posY, self.featureMap[x][y].weights))
		#print "Distances before sorting %s\n" % distances
		distances.sort(key=operator.itemgetter(0))
		#print "Distances after sorting: %s\n" % distances
		bmu = list(distances[0])
		#print bmu
		#(x,y) = self.featureMap.weights.index(bmu[1])
		#print "Minimum Euclidean distance: %s" % bmu[0] + " from featureMap[%d, %d]" % (bmu[1], bmu[2]) + " with weights %s" %bmu[3]
		#print "\n"
		#print "Bmu indices: [%d, %d]" % (bmu[1], bmu[2])
		return self.featureMap[bmu[1]][bmu[2]]

	def getNeighborhoodRadius(self, iteration):
		tempRad = self.latticeRadius * math.exp(-iteration/self.timeConst)
		#print "Neighborhoood radius %s" % tempRad
		return self.latticeRadius * math.exp(-iteration/self.timeConst)

	def getDistanceFallOff(self, distSq, radius):
		radiusSq = math.pow(radius, 2)
		return math.exp(-(distSq/(2 * radiusSq)))
	
	def distanceToBMU(self, bmu, neuron):
		xDist = math.pow((bmu.posX - neuron.posX), 2)
		yDist = math.pow((bmu.posY - neuron.posY), 2)
		return xDist + yDist

	def prepAdjustWeights(self, bmu, neighborRadius, inputVec):
		for x in range(len(self.featureMap)):
			for y in range(len(self.featureMap[x])):
				distTo = self.distanceToBMU(bmu, self.featureMap[x][y])
				if distTo <= math.pow(neighborRadius, 2):
					distFallOff = self.getDistanceFallOff(distTo, neighborRadius)
					self.featureMap[x][y].adjustWeights(inputVec, self.learnRate, distFallOff)
					self.featureMap[x][y].classLabel = inputVec[-1]

	def decreaseLearningRate(self):
		self.learnRate = self.initLearnRate * math.exp(-float(self.currIter)/self.numIter)
		#print "Learning Rate %f" % self.learnRate

	def train(self):
		self.loadData()
		numInput = len(self.dataSet)
		

		while self.currIter < self.numIter:

			for x in range(numInput):
				inputVec = self.getInput(x)
				#print "Input Vector: %s" % inputVec
				bmu = self.findBMU(inputVec)
				neighborRadius = self.getNeighborhoodRadius(self.currIter)
				#(xStart, xEnd, yStart, yEnd) = self.getNeighborhoodIndices(bmu, neighborRadius)
				self.prepAdjustWeights(bmu, neighborRadius, inputVec)
			
			self.currIter += 1
			self.decreaseLearningRate()

		for x in range(len(self.featureMap)):
			for y in range(len(self.featureMap)):
				print "[%d %d]: %s" % (x, y, self.featureMap[x][y].classLabel),
