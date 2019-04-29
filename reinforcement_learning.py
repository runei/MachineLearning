import numpy as np
from enum import Enum

class Action(Enum):
	UP = 0
	RIGHT = 1
	DOWN = 2
	LEFT = 3

EMPTY = np.ones(len(Action)) * -500 # random number not being utilized

def getNextPos(grid, i, j, action = None):
	perc_correct_dir = 0.8
	perc_left = 0.1
	perc_right = 0.1
	choose_dir = np.random.uniform()

	if action == None:
		id_ua = np.argmax(grid[i, j])
	else:
		id_ua = action
	# if choose_dir > perc_correct_dir:
		# if choose_dir > perc_correct_dir + perc_left:
			# id_ua = (id_ua + len(Action) - 1) % len(Action)
		# else:
			# id_ua = (id_ua + 1) % len(Action)

	new_i = i
	new_j = j
	if Action(id_ua) == Action.UP and new_i < np.ma.size(grid, 0) - 1:
		new_i += 1
	elif Action(id_ua) == Action.DOWN and new_i > 0:
		new_i -= 1
	elif Action(id_ua) == Action.RIGHT and new_j < np.ma.size(grid, 1) - 1:
		new_j += 1
	elif Action(id_ua) == Action.LEFT and new_j > 0:
		new_j -= 1

	if np.array_equal(grid[new_i, new_j], EMPTY):
		return i, j, id_ua
	else:
		return new_i, new_j, id_ua

def main():
	grid = np.zeros((3, 4, len(Action)))
	reward = np.ones((3, 4)) * -0.04
	reward[2, 3] = 1
	reward[1, 3] = -1
	grid[1, 1] = EMPTY
	gamma = 1 #discount_factor
	learning_rate = 0.5
	i = 0
	j = 0
	y = 0
	while True:
		y += 1
		# print("Type de direction")
		x = None#input()
		if y > 2000000:
			break
		new_i, new_j, id_ua = getNextPos(grid, i, j, x)
		# print(new_i, new_j)
		# break
		grid[i, j, id_ua] = (1 - learning_rate) * grid[i, j, id_ua] + (learning_rate * (reward[i, j] + gamma * np.max(grid[new_i, new_j]) - grid[i, j, id_ua]))

		#if reach final state, start again
		if abs(reward[i, j]) == 1:
			i = 0
			j = 0
		else:
			i = new_i
			j = new_j

		# print(grid)
	print(grid)

float_formatter = lambda x: "%.4f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})


main()