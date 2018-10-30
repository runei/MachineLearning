from __future__ import division
import sys
import operator as op
import math
from enum import Enum#, auto
from collections import Counter

class AttrType(Enum): #auto() only for python 3
	DISCRETE = 1# auto()
	CONTINUOUS = 2#auto()

class Attribute:

	def __init__(self, _id, _values, _type, _mean = 0):
		self.id = _id
		self.values = _values
		self.type = _type
		self.mean = _mean

	def __str__(self):
		return "id = {}\nvalues = {}".format(self.id, self.values)

class TreeNode:

	def __init__(self, value, predicted_value = None):
		self.value = value
		self.predicted_value = predicted_value
		self.subtree = []

	def __str__(self):
		if self.isPredicted():
			return "{} {}".format(self.value, self.predicted_value)
		else:
			return "{}".format(self.value)

	def printTree(self, depth = 0):
		tab_str = " * "
		tab = tab_str * depth
		print("{} {}".format(tab, self.value))
		tab += tab_str
		for node in self.subtree:
			if not node.isPredicted():
				node.printTree(depth + 1)
				# printTree(node, depth + 1)
			else:
				print("{} {}".format(tab, node))
				# print(node)

	def append(self, value):
		self.subtree.append(value)

	def isPredicted(self):
		return not self.subtree

def discretizeContinuousAttributes(samples, attributes):
	for row in samples:
		for attr in attributes:
			if attr.type == AttrType.CONTINUOUS:
				if float(row[attr.id]) < attr.mean:
					row[attr.id] = attr.values[0]
				else:
					row[attr.id] = attr.values[1]

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
			# attributes.append(Attribute(i, []))
			mean = sum([float(row[i]) for row in samples]) / len(samples)
			possible_vals = ["<{}".format(mean), ">={}".format(mean)]
			attributes.append(Attribute(i, possible_vals, AttrType.CONTINUOUS, mean))
			# pass
		else:
			attributes.append(Attribute(i, values, AttrType.DISCRETE))

	discretizeContinuousAttributes(samples, attributes)

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
	return count.most_common(1)[0][0]
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

def isGreaterThan(a, b):
	return a > b

def isLowerThan(a, b):
	return a < b

def createSubset(samples, attr_id, value, attr_type):
	subset = []
	for row in samples:
		# if attr_type == AttrType.DISCRETE:
		if row[attr_id] == value:
			subset.append(row)
		# elif attr_type == AttrType.CONTINUOUS:
		# 	if row[attr_id] > mean:
		# 		subset.append(row)
	return subset

def getRemainder(samples, attr):
	remainder = 0
	total_count = len(samples)
	attr_count = Counter(row[attr.id] for row in samples)
	for value, count in attr_count.items():
		subset = createSubset(samples, attr.id, value, attr.type)
		remainder += count / total_count * getEntropy(subset)
	return remainder

def getInformationGain(samples, attributes, entropy):
	information_gain = []
	for attr in attributes:
		if attr.values:
			information_gain.append(entropy - getRemainder(samples, attr))
	return information_gain

def chooseAttribute(samples, attributes):
	entropy = getEntropy(samples)
	information_gain = getInformationGain(samples, attributes, entropy)
	best = information_gain.index(max(information_gain))
	return attributes[best]

def trainDecisionTree(samples, attributes, default):
	if not samples:
		return default
	if isSameClass(samples):
		return getClass(samples[0])
	if not attributes:
		return mode(samples)

	best = chooseAttribute(samples, attributes)
	attributes.remove(best)

	tree = TreeNode(best.id)
	for value in best.values:
		subset = createSubset(samples, best.id, value, best.type)
		subtree = trainDecisionTree(subset, attributes, mode(samples))
		if not isinstance(subtree, TreeNode):
			subtree = TreeNode(value, subtree)
		tree.append(subtree)
	return tree

def main():
	if len(sys.argv) != 2:
		print("Usage: python {} <file_name>".format(sys.argv[0]))
		sys.exit(1)

	file_name = sys.argv[1]
	samples, attributes = openFile(file_name)
	# print(mode(samples))
	# print(attributes[0])
	# print(getRemainder(samples, attributes[1], 1))

	tree = trainDecisionTree(samples, attributes, [TreeNode(None, None)])
	tree.printTree()

main()

