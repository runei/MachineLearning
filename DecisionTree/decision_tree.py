from __future__ import division
import sys
import operator as op
import math
# from anytree import Node, RenderTree
from collections import Counter

class Attribute:

	def __init__(self, _id, values):
		self.id = _id
		self.values = values

	def __str__(self):
		return "id = {}\nvalues = {}".format(self.id, self.values)

class TreeNode:

	def __init__(self, value, prediction = None):
		self.value = value
		self.prediction = prediction
		self.predicted = prediction != None

def openFile(file_name):
	file = open(file_name)
	samples = []
	for line in file:
		line = line.strip()
		#line = line.strip(".")
		# ignore empty lines. comments are marked with a |
		if len(line) == 0 or line[0] == '|':
			continue
		entry = [x.strip() for x in line.split(",")]
		entry[-1] = entry[-1].strip('.') #delete '.' in last attribute
		samples.append(entry)

	# print(samples)
	attributes = []
	for i in range(len(samples[0])):
		values = list({x[i] for x in samples}) # set of all different attribute values
		if values[0].isdigit(): # if the first value is a digit, assume all are numeric
			attributes.append([])
		else:
			attributes.append(Attribute(i, values))

	return samples, attributes[:-1] #last attribute is the class where the sample belongs

def getClass(sample):
	#last position is the class
	return sample[-1]

def getAllClasses(samples):
	return list({getClass(sample) for sample in samples})

def isSameClass(samples):
	test_class = getClass(samples[0])
	for i in range(1, len(samples)):
		if test_class != getClass(samples[i]):
			return False
	return True

def mode(samples):
	#If there are no attributes left, label the node with the majority classification for the remaining examples
	count = Counter(getClass(s) for s in samples)
	return list(count.most_common(1)[0][0])
	"""classes = getAllClasses(samples)
	classes_count = dict.fromkeys(classes, 0)
	for s in samples:
		classes_count[getClass(s)] += 1
	#return the key with the maximum  count from the dict
	return max(classes_count.items(), key = op.itemgetter(1))[0]"""

def getEntropy(samples):
	# print(class_count)
	total_count = len(samples)
	class_count = Counter(getClass(s) for s in samples)
	entropy = 0
	for key, value in class_count.items():
		pi = value / total_count
		entropy += -pi * math.log(pi, 2)
		# entropy += -pi * math.log2(pi) #math.log2 is python 3.3
	return entropy

def createSubset(samples, attr_id, value):
	subset = []
	for row in samples:
		if row[attr_id] == value:
			subset.append(row)
	return subset

def getRemainder(samples, attr):
	remainder = 0
	total_count = len(samples)
	attr_count = Counter(row[attr.id] for row in samples)
	for value, count in attr_count.items():
		subset = createSubset(samples, attr.id, value)
		remainder += count / total_count * getEntropy(subset)
	return remainder

def chooseAttribute(samples, attributes):
	entropy = getEntropy(samples)
	information_gain = []
	for attr in attributes:
		if attr.values:
			information_gain.append(entropy - getRemainder(samples, attr))
	best = information_gain.index(max(information_gain))
	return attributes[best]

def trainDecisionTree(samples, attributes, default):
	if not samples:
		return default
	if isSameClass(samples):
		return list(getClass(samples[0]))
	if not attributes:
		return mode(samples)

	best = chooseAttribute(samples, attributes)
	attributes.remove(best)

	tree = [TreeNode(best.id)]
	for value in best.values:
		subset = createSubset(samples, best.id, value)
		subtree = trainDecisionTree(subset, attributes, mode(samples))
		tree.append(subtree)
	return tree

def printTree(root, depth = 0):
	tab = " * " * depth
	for node in root:
		if isinstance(node, list):
			printTree(node, depth + 1)
		else:
			print("{} {}".format(tab, node))

def main():
	if len(sys.argv) != 2:
		print("Usage: python {} <file_name>".format(sys.argv[0]))
		sys.exit(1)

	file_name = sys.argv[1]
	samples, attributes = openFile(file_name)
	# print(mode(samples))
	# print(attributes[0])
	# print(getRemainder(samples, attributes[1], 1))

	tree = trainDecisionTree(samples, attributes, [TreeNode()])
	printTree(tree)

main()

