import gym
from gym import error, spaces, utils
from gym.utils import seeding

class TicTacEnv(gym.Env):
	metadata = {'render.modes': ['human']}


	def __init__(self):
		self.state = []
		for i in range(3):
			self.state += [[]]
			for j in range(3):
				self.state[i] += [0]
		#self.state[0][0]=1
		#self.state[1][0]=-1
		#self.state[0][1]=-1
		#self.state[1][2]=1
		#self.state[1][1]=-1
		#self.state[2][0]=1
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0

	def check(self):

		if(self.counter<5):
			return 0
		for i in range(3):
			if(self.state[i][0] != 0 and self.state[i][1] == self.state[i][0] and self.state[i][1] == self.state[i][2]):
				if(self.state[i][0] == 1):
					return 1
				else:
					return 2
			if(self.state[0][i] != 0 and self.state[1][i] == self.state[0][i] and self.state[1][i] == self.state[2][i]):
				if(self.state[0][i] == 1):
					return 1
				else:
					return 2
		if(self.state[0][0] != 0 and self.state[1][1] == self.state[0][0] and self.state[1][1] == self.state[2][2]):
			if(self.state[0][0] == 1):
				return 1
			else:
				return 2
		if(self.state[0][2] != 0 and self.state[0][2] == self.state[1][1] and self.state[1][1] == self.state[2][0]):
			if(self.state[1][1] == 1):
				return 1
			else:
				return 2
		return 0

	def step(self, target):

		if self.state[int(target/3)][target%3] != 0:
			#print('inc st', self.counter,'step', target)
			#print("Invalid Step")
			#self.counter += 1
			#self.reset()
			#self.reward += -0.9
			self.done = -200
			return [self.convert(self.state), self.reward, self.done, self.add]
		else:
			if(self.counter%2 == 0):
				self.state[int(target/3)][target%3] = 1
			else:
				self.state[int(target/3)][target%3] = -1
			self.counter += 1
			if(self.counter == 9):
				#self.reward += -0.2
				self.done = -1

		"""win = self.check()
		if(win):
			self.done = 1
			#print("Player ", win, " wins.", sep = "", end = "\n")
			self.add[win-1] = 1
			if win == 1:
				self.reward = 1
			else:
				self.reward = -1
		#self.reward += -10"""
		return [self.convert(self.state), self.reward, self.done, self.add]

	def reset(self):

		for i in range(3):
			for j in range(3):
				self.state[i][j] = 0
		#self.state[0][0]=1
		#self.state[1][0]=-1
		#self.state[0][1]=-1
		#self.state[1][2]=1
		#self.state[1][1]=-1
		#self.state[2][0]=1
		self.counter = 0
		self.done = 0
		self.add = [0, 0]
		self.reward = 0
		return self.convert(self.state)

	def convert(self, state):
		import numpy as np
		tmp = []
		for i in range(3):
			for j in range(3):
				tmp.append(state[i][j])
		return np.array(tmp)

	def render(self,metadata):
		for i in range(3):
			for j in range(3):
				print(self.state[i][j], end = " ")
			print("")
