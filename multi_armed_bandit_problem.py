import numpy as np
import matplotlib.pyplot as plt

class Bandit:
  def __init__(self, arms=10):
    self.rates = np.random.rand(arms)

  def play(self, arm):
    rate = self.rates[arm]
    if rate > np.random.rand():
      return 1
    else:
      return 0

## METAN's idea 1
class Agent_1:
  def __init__(self,epsilon,action_size=10):
    self.epsilon = epsilon
    self.action_size = action_size
    self.Qs = np.zeros(action_size)
    self.ns = np.zeros(action_size)

  def update(self,action,reward):
    self.ns[action] += 1
    self.Qs[action] += (reward - self.Qs[action])/self.ns[action]

  def get_action(self,step):
        return step%self.action_size

def run_1(runs,steps,epsilon):
  all_rates = np.zeros((runs,steps))
  all_rewards = np.zeros((runs,steps))

  for run in range(runs):
    bandit = Bandit(arms=10)
    agent = Agent_1(epsilon,action_size=10)
    total_reward  = 0
    total_rewards = []
    rates         = []

    for step in range(steps):
      action = agent.get_action(step)
      reward = bandit.play(action)
      agent.update(action,reward)
      total_reward += reward
      total_rewards.append(total_reward)
      rates.append(total_reward/(step+1))
    all_rates[run] = rates
    all_rewards[run] = total_rewards

  avg_rewards = np.average(all_rewards,axis=0)
  avg_rates = np.average(all_rates,axis=0)
  return avg_rates, avg_rewards

steps = 1000
runs  = 10000

avg_rates, avg_rewards = run_1(runs,steps,epsilon=0.1)

plt.figure()
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()

print(avg_rates[-1])

plt.figure()
plt.ylabel('Rewards')
plt.xlabel('Steps')
plt.plot(avg_rewards)
plt.show()

print(avg_rewards[-1])

## METAN's idea 2
class Agent_2:
  def __init__(self,epsilon,action_size=10):
    self.epsilon = epsilon
    self.action_size = action_size
    self.Qs = np.zeros(action_size)
    self.ns = np.zeros(action_size)

  def update(self,action,reward):
    self.ns[action] += 1
    self.Qs[action] += (reward - self.Qs[action])/self.ns[action]

  def get_action(self,step,steps):
    if step < steps*self.epsilon:
        self.argmax = np.argmax(self.Qs)
        return step%self.action_size
    else:
        return self.argmax

def run_2(runs,steps,epsilon):
  all_rates = np.zeros((runs,steps))
  all_rewards = np.zeros((runs,steps))

  for run in range(runs):
    bandit = Bandit(arms=10)
    agent = Agent_2(epsilon,action_size=10)
    total_reward  = 0
    total_rewards = []
    rates         = []

    for step in range(steps):
      action = agent.get_action(step,steps)
      reward = bandit.play(action)
      agent.update(action,reward)
      total_reward += reward
      total_rewards.append(total_reward)
      rates.append(total_reward/(step+1))
    all_rates[run] = rates
    all_rewards[run] = total_rewards

  avg_rewards = np.average(all_rewards,axis=0)
  avg_rates = np.average(all_rates,axis=0)
  return avg_rates, avg_rewards

steps = 1000
runs  = 10000

avg_rates, avg_rewards = run_2(runs,steps,epsilon=0.1)

plt.figure()
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()
print(avg_rates[-1])

plt.figure()
plt.ylabel('Rewards')
plt.xlabel('Steps')
plt.plot(avg_rewards)
plt.show()
print(avg_rewards[-1])

## epsilon-greedy method (epsilon=0.1)
class Agent_epsilon_greedy:
  def __init__(self,epsilon,action_size=10):
    self.epsilon = epsilon
    self.Qs = np.zeros(action_size)
    self.ns = np.zeros(action_size)

  def update(self,action,reward):
    self.ns[action] += 1
    self.Qs[action] += (reward - self.Qs[action])/self.ns[action]

  def get_action(self):
    if np.random.rand() < self.epsilon:
      return np.random.randint(0,len(self.Qs))
    else:
      return np.argmax(self.Qs)

def run_epsilon_greedy(runs,steps,epsilon):
  all_rates = np.zeros((runs,steps))
  all_rewards = np.zeros((runs,steps))

  for run in range(runs):
    bandit = Bandit(arms=10)
    agent = Agent_epsilon_greedy(epsilon,action_size=10)
    total_reward  = 0
    total_rewards = []
    rates         = []

    for step in range(steps):
      action = agent.get_action()
      reward = bandit.play(action)
      agent.update(action,reward)
      total_reward += reward
      rates.append(total_reward/(step+1))
      total_rewards.append(total_reward)
    all_rates[run] = rates
    all_rewards[run] = total_rewards

  avg_rewards = np.average(all_rewards,axis=0)
  avg_rates = np.average(all_rates,axis=0)
  return avg_rates, avg_rewards

steps = 1000
runs  = 10000

avg_rates, avg_rewards = run_epsilon_greedy(runs, steps, epsilon=0.1)

plt.figure()
plt.ylabel('Rates')
plt.xlabel('Steps')
plt.plot(avg_rates)
plt.show()
print(avg_rates[-1])

plt.figure()
plt.ylabel('Rewards')
plt.xlabel('Steps')
plt.plot(avg_rewards)
plt.show()
print(avg_rewards[-1])

## epsilon-greedy method (multi-epsilons)
steps = 1000
runs  = 10000
epsilons  = [0.01, 0.05, 0.1, 0.3, 0.5]

plt.figure()
for epsilon in epsilons:
  avg_rates, avg_rewards = run_epsilon_greedy(runs, steps, epsilon=epsilon)
  plt.ylabel('Rates')
  plt.xlabel('Steps')
  plt.plot(avg_rates,label=str(epsilon))
plt.legend()
plt.show()
