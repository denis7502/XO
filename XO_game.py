class NN():

	def __init__(self):
		self.model, self.criterion, self.optimizer = self.config_model()

	def config_model(self):
		import torch
		from torch import nn
		def init_weights(m):
		    if isinstance(m, nn.Linear):
		    	torch.nn.init.xavier_uniform(m.weight)
		    	m.bias.data.fill_(0.01)

		model = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 9),
            nn.Softmax(dim = -1),
            )
		#model.apply(init_weights)

		if torch.cuda.is_available():
			model = model.cuda()
		optimizer = torch.optim.Adam(model.parameters())
		criterion = torch.nn.HuberLoss()
		#criterion = torch.nn.MSELoss()
		return model, criterion, optimizer
   
	def hash_value(self, state):
		res = 0
		for i in range(9):
			res *= 3
			res += state[i]
		return res

	def train(self, epochs):
		import numpy as np
		import torch
		import gym
		import gym_tictac
		import random
		import matplotlib.pyplot as plt
		import time
		from tqdm import tqdm

		def convert_state(state, null = 0):
			tmp = []
			for i in range(3):
				for j in range(3):
					tmp.append(state[i][j])
			if null == 1:
				tmp = np.array(tmp)
				tmp[:] = 0
			return torch.from_numpy(np.array(tmp)).float().cuda(), np.array(tmp)

		def egreedy_policy(q_values, state, e):
				max_v = -9999
			#if random.random() < e:
			#	new_pos = random.choice(np.where(state == 0)[0])
			#	return new_pos
			#else:
				pos = np.where(state == 0)[0]
				for i in pos:
					if q_values[i] > max_v:
						max_v = q_values[i]
						new_pos = i
				return new_pos
				#return np.argmax(q_values)
		
		def Q_tables_init():
			Q_tables = {}
			return Q_tables

		def updateQ(Q_tables, steps, rewards):
			next_max = -1.0

			for i in reversed(steps):
				if next_max < 0:
					Q_tables[i[0]][i[1]] = rewards
				else:
					Q_tables[i[0]][i[1]] = Q_tables[i[0]][i[1]] * (1-learning_rate) + learning_rate * gamma * next_max
				next_max = np.argmax(Q_tables[i[0]][:])

			return Q_tables
		
		def get_qval(Q_tables,state):
			if self.hash_value(state) in Q_tables:
				return Q_tables
			else:
				Q_tables[self.hash_value(state)] = np.array([0.6 for i in range(9)])
				return Q_tables

		#def updateW(Q_tables. model, steps):


		env = gym.make('tictac-v1')

		model, criterion,optimizer = self.config_model()
		#m(1- learning_rate) * odel.eval()
		#m = torch.nn.Softmam = torch.nn.Sof+x(dim =learning_rate*gamm)
		e = 0.4
		Q_tables_a = Q_tables_init()
		Q_tables_b = Q_tables_init()
		gamma = 0.97
		action = 0
		learning_rate = 0.01
		rev = []
		win = lose = drawn = 0
		c = 0
		for epoch in tqdm(range(epochs+1)):
			random.seed()
			state = env.reset()
			done = 0
			steps_a = []
			steps_b = []
			rewards = 0

			while done == 0:
				Q_tables_a = get_qval(Q_tables_a, state)
				hash_value = self.hash_value(state)
				action_a = egreedy_policy(Q_tables_a[hash_value], state, e)
				steps_a.append((hash_value,action_a))
				reward, next_state, done = self.space(env,action_a)
				state = next_state
				if done == 0 :
					Q_tables_b = get_qval(Q_tables_b, state)
					hash_value = self.hash_value(state)
					action_b = egreedy_policy(Q_tables_b[hash_value], state, e)
					steps_b.append((hash_value,action_b))
					reward, next_state, done = self.space(env,action_b)
					state = next_state
				if env.check() == 1:
					win += 1
					if done != -1:
						Q_tables_a = updateQ(Q_tables_a, steps_a, 1)
						Q_tables_b = updateQ(Q_tables_b, steps_b, -1)
					else:
						Q_tables_a = updateQ(Q_tables_a, steps_a, 0.5)
						Q_tables_b = updateQ(Q_tables_b, steps_b, -0.2)

					#policy_b.reward_epoch.append(-10)
					done = 1
				elif env.check() == 2:
					lose += 1
					if done != -1:
						Q_tables_a = updateQ(Q_tables_a, steps_a, -1)
						Q_tables_b = updateQ(Q_tables_b, steps_b, 1)
					else:
						Q_tables_a = updateQ(Q_tables_a, steps_a, -0.5)
						Q_tables_b = updateQ(Q_tables_b, steps_b, 1)
					done = 1
				elif done == -1:
					Q_tables_a = updateQ(Q_tables_a, steps_a, 0.2)
					Q_tables_b = updateQ(Q_tables_b, steps_b, 0.6)
					drawn += 1


			
			if epoch > 0:
				rev.append((win/(epoch), lose/(epoch), drawn/(epoch)))
			#print('\n')
			if epoch %100000 == 0:
				self.savePolicy(Q_tables_b)
				fig, ax = plt.subplots()
				ax.plot(rev, label = ['win', 'lose', 'drawn'])
				ax.legend()
				plt.savefig('img/w_l_%s.png'%(epoch))
				plt.close()

		env.close()


	def space(self, env,action):



		state, reward, done, info = env.step(action) 
		#print("reward: ", reward)
		#print("")
		#print(done)


		#env.render()
		return reward, state, done

	def load(self, Q_tables):
		import torch
		import gym
		import gym_tictac
		import numpy as np

		def get_pos(np_out, state):
			max_v = - 9999
			pos = np.where(state == 0)[0]
			for i in pos:
				if np_out[i] > max_v:
					max_v = np_out[i]
					new_pos = i
			return new_pos


		def space(env, action):
			state, reward, done, info = env.step(action) 
			print("reward: ", reward)
			env.render()
			return state, done

		def convert_state(state):
			return torch.from_numpy(np.array(state)).float().cuda()

		env = gym.make('tictac-v1')

		state = env.reset()
		#model, criterion, optimizer = self.config_model()

		st = 0
		while True:
			#output = model(convert_state(state))
			#output = m(output)
			#np_out = output.cpu().detach().numpy()
			#print(output)
			action =  get_pos(Q_tables[self.hash_value(state)], state)
			#print(action,get_pos(np_out, state))
			state, done = space(env, action)
			s = int(input())
			state, done = space(env, s)
			st +=1
			if done == 1 or done == -1:
				self.load(self.loadPolicy())
			
	def savePolicy(self, data):
		import pickle
		fw = open('policy', 'wb')
		pickle.dump(data, fw)
		fw.close()

	def loadPolicy(self):
		import pickle
		fr = open('policy', 'rb')
		Q_tables = pickle.load(fr)
		fr.close()
		return Q_tables





def main():
	nn = NN()
	#nn.train(1_000_000)
	#nn.train(20)
	#nn.space()
	nn.load(nn.loadPolicy())
main()