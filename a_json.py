import json
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

f = open('dqn_SoccerScoreGoal-v0_log.json','r')

jsonData = json.load(f)

reward_list = jsonData['episode_reward']
reward_list = reward_list[0:80000]
x = range(0, 80000)
y = reward_list
plt.ylim(-2,9)
plt.plot(x,y)
plt.xlabel("episode")
plt.ylabel("episode_reward")

plt.savefig("ep_reward.png")
f.close()