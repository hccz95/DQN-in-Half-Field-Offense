import numpy as np
import gym
import gym_soccer
import hfo_py
import argparse
import time

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam

from rl.agents.dqn import DQNAgent
from rl.policy import LinearAnnealedPolicy, BoltzmannQPolicy, EpsGreedyQPolicy
from rl.memory import SequentialMemory
from rl.core import Processor
from rl.callbacks import FileLogger, ModelIntervalCheckpoint
import matplotlib.pyplot as plt


ENV_NAME = 'SoccerScoreGoal-v0'

ACTION_LIST = [hfo_py.MOVE, hfo_py.DRIBBLE, hfo_py.SHOOT, hfo_py.REDUCE_ANGLE_TO_GOAL, hfo_py.GO_TO_BALL]
ACTION_SIZE = len(ACTION_LIST)
STATE_SIZE = 59
"""
def gen_pact():
    global ACTION_LIST,ACTION_SIZE
    pacts = []
    for i in range(-18,19,2):
        for j in range(8,11):
            pacts.append([hfo_py.KICK, j*10, i*10])
        pacts.append([hfo_py.DASH, 50, i * 10])
        pacts.append([hfo_py.DASH, 100, i * 10])
        pacts.append([hfo_py.TURN, i*10])
    ACTION_LIST = pacts
    ACTION_SIZE = len(pacts)
"""
def gen_pact():
    global ACTION_LIST,ACTION_SIZE
    pacts = []
    for i in range(-18,19,1):
        pacts.append([hfo_py.TURN, i*10])
        pacts.append([hfo_py.KICK, 100, i * 10])
    for j in range(0,11):
        pacts.append([hfo_py.DASH, j* 10, 0])
    ACTION_LIST = pacts
    ACTION_SIZE = len(pacts)


def action_wrap(action, ACTION_LIST):
    gym_action = [0, 0, 0, 0, 0, 0]

    if ACTION_LIST[action][0] == hfo_py.DASH:
        gym_action[0] = 0
        gym_action[1] = ACTION_LIST[action][1]
        gym_action[2] = ACTION_LIST[action][2]
    elif ACTION_LIST[action][0] == hfo_py.TURN:
        gym_action[0] = 1
        gym_action[3] = ACTION_LIST[action][1]
    elif ACTION_LIST[action][0]== hfo_py.KICK:
        gym_action[0] = 2
        gym_action[4] = ACTION_LIST[action][1]
        gym_action[5] = ACTION_LIST[action][2]
    return gym_action

gen_pact()

parser = argparse.ArgumentParser()
parser.add_argument('--mode', choices=['train', 'test'], default='train')
parser.add_argument('--env-name', type=str, default=ENV_NAME)
parser.add_argument('--weights', type=str, default=None)
args = parser.parse_args()

# Get the environment and extract the number of actions.
env = gym.make(args.env_name)
#np.random.seed(123)
#env.seed(123)
nb_actions = ACTION_SIZE

# Next, we build a very simple model.
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(1024))
model.add(Activation('relu'))
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))
print(model.summary())

# Finally, we configure and compile our agent. You can use every built-in Keras optimizer and
# even the metrics!
memorysize = int(1*1e+6)
policy_step = int(1*1e+6)
num_step = int(10*1e+6)

memory = SequentialMemory(limit=memorysize, window_length=1)
policy = LinearAnnealedPolicy(EpsGreedyQPolicy(), attr='eps', value_max=1., value_min=.1, value_test=.05,nb_steps=policy_step)
#policy = BoltzmannQPolicy()
dqn = DQNAgent(model=model, nb_actions=nb_actions, policy=policy, memory=memory, nb_steps_warmup=50000, gamma=.99, target_model_update=10000, train_interval=4, delta_clip=1.)
dqn.compile(Adam(lr=.00025), metrics=['mae'])
if args.mode == 'train':
    # Okay, now it's time to learn something! We capture the interrupt exception so that training
    # can be prematurely aborted. Notice that now you can use the built-in Keras callbacks!
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    checkpoint_weights_filename = 'dqn_' + args.env_name + '_weights_{step}.h5f'
    log_filename = 'dqn_{}_log.json'.format(args.env_name)
    callbacks = [ModelIntervalCheckpoint(checkpoint_weights_filename, interval=250000)]
    callbacks += [FileLogger(log_filename, interval=10000)]
    history = dqn.fit(env, callbacks=callbacks, nb_steps=num_step, log_interval=10000, visualize=False)

    plt.subplot(1, 1, 1)
    plt.plot(history.history["episode_reward"])
    plt.xlabel("episode")
    plt.ylabel("reward")
    plt.show()  # windowが表示されます。
    # After training is done, we save the final weights one more time.
    dqn.save_weights(weights_filename, overwrite=True)

    # Finally, evaluate our algorithm for 10 episodes.
    dqn.test(env, nb_episodes=100, visualize=True)
elif args.mode == 'test':
    weights_filename = 'dqn_{}_weights.h5f'.format(args.env_name)
    if args.weights:
        weights_filename = args.weights
    dqn.load_weights(weights_filename)
    dqn.test(env, nb_episodes=10, visualize=True)
