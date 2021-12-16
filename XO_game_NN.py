import torch
import torch.nn as nn
import random
import numpy as np


torch.manual_seed(1)
device = torch.device('cuda')

class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.l1 = nn.Linear(9, 16, bias=True)
        self.l2 = nn.Linear(16, 16, bias=True)
        self.l3 = nn.Linear(16, 16, bias=True)
        self.l4 = nn.Linear(16, 16, bias=True)
        self.l5 = nn.Linear(16, 9, bias=False)

        self.gamma = 0.96
        self.learning_rate = 1*10**(-4)
        
        # Episode policy and reward history 
        self.policy_history = torch.autograd.Variable(torch.Tensor(), requires_grad = True)
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
    	model = torch.nn.Sequential(
			self.l1,
			nn.Tanh(),
			self.l2,
			nn.Tanh(),
			self.l3,
			nn.Tanh(),
			self.l4,
			nn.Tanh(),
			self.l5,
			nn.Softmax(dim=-1),
    		)
    	return model(x)
   
def select_action(state, policy, E):
	import torch.distributions.categorical as c

	#if random.random() < 0.1:
	#	action = np.array(random.choice(np.where(state == 0)[0]))
	#	actions = torch.from_numpy(action).type(torch.FloatTensor).reshape(1)
	#else:
	state = torch.from_numpy(state).type(torch.FloatTensor)
	state = policy(torch.autograd.Variable(state))
	c = c.Categorical(state)
	action = c.sample()
	actions = c.log_prob(action)
	actions = actions.unsqueeze(0)
	if policy.policy_history.dim() > 0:
		policy.policy_history = torch.cat([policy.policy_history, actions])
	else:
		policy.policy_history = (actions)
	return action

def load_q():
	import pickle
	fr = open('policy', 'rb')
	Q_tables = pickle.load(fr)
	fr.close()
	return Q_tables

def hash_value(state):
	res = 0
	for i in range(9):
		res *= 3
		res += state[i]
	return res

def select_action_C(Q_tables, state):

	def get_pos(np_out, state):
		max_v = - 9999
		pos = np.where(state == 0)[0]
		for i in pos:
			if np_out[i] > max_v:
				max_v = np_out[i]
				new_pos = i
		return new_pos

	h_v = hash_value(state)
	if h_v in Q_tables:
		action =  get_pos(Q_tables[h_v], state)
	else:
		action = random.choice(np.where(state == 0)[0])

	return action


def update_policy(policy, optimizer):
    R = 0

    rewards = []
    
    # Discount future rewards back to the present using gamma
    for r in policy.reward_episode[::-1]:
        R = r + policy.gamma * R
        rewards.insert(0,R)
        
    # Scale rewards
    rewards = torch.FloatTensor(rewards)
    #rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)
    
    # Calculate loss
    loss = (torch.sum(torch.mul(policy.policy_history, torch.autograd.Variable(rewards)).mul(-1), -1))
    
    # Update network weights
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #scheduler.step()
    policy.policy_history = torch.autograd.Variable(torch.Tensor(), requires_grad = True)
    policy.reward_episode= []

def savePolicy(model,name):
	torch.save(model, '%s.pth'%(name))

def loadPolicy(name):

	model = torch.load('%s.pth'%(name))
	#model.eval()
	return model


def main(episodes):
	import gym
	import gym_tictac
	import numpy as np
	import random
	from tqdm import tqdm
	import matplotlib.pyplot as plt

	env = gym.make('tictac-v1')
	win = lose = drawn = err = 0
	rev = []
	Q_tables = load_q()
	E = 0.99
	fig, ax = plt.subplots()


	for episode in tqdm(range(episodes+1)):
		state = env.reset() # Reset environment and record the starting state
		done = 0
		random.seed()
		a = 0
		b = 0
		while done == 0:
			action = select_action(state, policy_a, E)
			state, reward, done, _ = env.step(action.item())
			policy_a.reward_episode.append(0)
			if done == 0 :
				a +=1
				action_b = select_action(state, policy_b, E)#random.choice(np.where(state==0)[0])#select_action_C(Q_tables, state)
				state, reward, done, _ = env.step(action_b)
				policy_b.reward_episode.append(0.1)
				if done == 0:
					b += 1

		if env.check() == 1:
			win += 1
			if done != -1:
				#policy_a.reward_episode.append(1)
				policy_a.reward_episode[-1] = 1
				policy_b.reward_episode[-1] = -1
				#policy_b.reward_episode[-1] = -1
			else:
				#policy_a.reward_episode.append(0.5)
				policy_a.reward_episode[-1] = 0.5
				policy_b.reward_episode[-1] = -1
			done = 1
		elif env.check() == 2:
			lose += 1
			if done != -1:
				#policy_a.reward_episode.append(-1)
				policy_a.reward_episode[-1] = -1
				policy_b.reward_episode[-1] = 1
			else:
				#policy_a.reward_episode.append(-1)
				policy_a.reward_episode[-1] = -1
				policy_b.reward_episode[-1] = 1
			done = 1
		elif done == -1 and env.check()==0:
			drawn += 1
			policy_a.reward_episode[-1] = 0.2
			policy_b.reward_episode[-1] = 0.6
			done = 1
		elif done == -200:
			err += 1
			if a > b:
				policy_b.reward_episode[-1] = -2
			else:
				policy_a.reward_episode[-1] = -2
			#policy_b.reward_episode.append(-100)
			done = 1

		update_policy(policy_a, optimizer_a)
		update_policy(policy_b, optimizer_b)

		if episode % 50_000 == 0 and episode>0:
			rev.append((win/(episode), lose/(episode), drawn/(episode), err/(episode)))



		if episode % 100_000 == 0 and episode>0:
			fig, ax = plt.subplots()
			ax.plot(rev, label = ['win', 'lose', 'drawn', 'err'])
			ax.grid()
			plt.legend()
			plt.savefig('img_n/l_w%s.png'%(episode))
			plt.close()	


	savePolicy(policy_a,'weight_a')
	savePolicy(policy_b,'weight_b')


def game():
	import gym
	import gym_tictac
	import numpy as np
	import random
	import torch

	def get_pos(out, state):
		import torch
		import numpy as np
		import torch.distributions.categorical as c

		c = c.Categorical(out)
		new_pos = c.sample()
		return new_pos

	def convert_state(state):
		return torch.from_numpy(np.array(state)).float().cpu()
	
	env = gym.make('tictac-v1')
	model = loadPolicy('best_weight')
	model.eval()
	state = env.reset()
	with torch.no_grad():
		while True:
			state_v = convert_state(state)
			np_out = model(state_v)
			print(np_out)
			x = get_pos(np_out, state)
			print(x)
			env.step(x)
			env.render()
			s = int(input())
			state, reward, done, _= env.step(s)
			if done == 1 or done < 0:
				game()


		

policy_a = Policy()

policy_b = Policy()
#policy_b = Policy()

#optimizer_b = torch.optim.Adam(policy_b.parameters(), lr=policy_b.learning_rate)
#policy = loadPolicy()
optimizer_a = torch.optim.Adam(policy_a.parameters(), lr=policy_a.learning_rate)
optimizer_b = torch.optim.Adam(policy_b.parameters(), lr=policy_b.learning_rate)
#scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25_000, gamma=0.1)
main(10_000_000)
#main(200_000)
#game()

