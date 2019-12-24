from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())

config = tf.ConfigProto(
    gpu_options=tf.GPUOptions(
        visible_device_list="0", # specify GPU number
        allow_growth=True
    )
)
sess = tf.Session(config=config)
import numpy as np
import collections
import hfo_py


def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = tf.keras.backend.abs(err) < 1.0
    L2 = 0.5 * tf.keras.backend.square(err)
    L1 = (tf.keras.backend.abs(err) - 0.5)
    loss = tf.where(cond, L2, L1)
    return tf.keras.backend.mean(loss)

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

class QNetwork:
    def __init__(self, lr=0.01, s_size=59, a_size=3, hidden_size=1024):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_size, activation='relu', input_dim=s_size))
        self.model.add(tf.keras.layers.Dense(hidden_size/2, activation='relu', input_dim=s_size))
        self.model.add(tf.keras.layers.Dense(hidden_size/4, activation='relu', input_dim=s_size))
        self.model.add(tf.keras.layers.Dense(hidden_size/8, activation='relu', input_dim=s_size))
        self.model.add(tf.keras.layers.Dense(a_size, activation='linear'))
        self.optimizer = tf.keras.optimizers.Adam(lr=lr)
        self.model.compile(loss=huber_loss,optimizer=self.optimizer)
        self.a_size = a_size
        self.s_size = s_size

    def replay(self, memory, batch_size, gamma, targetQN):
        inputs = np.zeros((batch_size, self.s_size))
        targets = np.zeros((batch_size, self.a_size))
        mini_batch = memory.sample(batch_size)
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(mini_batch):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)    # Qネットワークの出力
            targets[i][action_b] = target               # 教師信号

            # shiglayさんよりアドバイスいただき、for文の外へ修正しました
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # ep

    # [※p1] 優先順位付き経験再生で重みの学習
    def pioritized_experience_replay(self, memory, batch_size, gamma, targetQN, memory_TDerror):

        # 0からTD誤差の絶対値和までの一様乱数を作成(昇順にしておく)
        sum_absolute_TDerror = memory_TDerror.get_sum_absolute_TDerror()
        generatedrand_list = np.random.uniform(0, sum_absolute_TDerror, batch_size)
        generatedrand_list = np.sort(generatedrand_list)

        # [※p2]作成した乱数で串刺しにして、バッチを作成する
        batch_memory = Memory(max_size=batch_size)
        idx = 0
        tmp_sum_absolute_TDerror = 0
        for (i, randnum) in enumerate(generatedrand_list):
            while tmp_sum_absolute_TDerror < randnum:
                tmp_sum_absolute_TDerror += abs(memory_TDerror.buffer[idx]) + 0.0001
                idx += 1

            batch_memory.add(memory.buffer[idx])

        # あとはこのバッチで学習する
        inputs = np.zeros((batch_size, self.s_size))
        targets = np.zeros((batch_size, self.a_size))
        for i, (state_b, action_b, reward_b, next_state_b) in enumerate(batch_memory.buffer):
            inputs[i:i + 1] = state_b
            target = reward_b

            if not (next_state_b == np.zeros(state_b.shape)).all(axis=1):
                # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
                retmainQs = self.model.predict(next_state_b)[0]
                next_action = np.argmax(retmainQs)  # 最大の報酬を返す行動を選択する
                target = reward_b + gamma * targetQN.model.predict(next_state_b)[0][next_action]

            targets[i] = self.model.predict(state_b)  # Qネットワークの出力
            targets[i][action_b] = target  # 教師信号
            self.model.fit(inputs, targets, epochs=1, verbose=0)  # epochsは訓練データの反復回数、verbose=0は表示なしの設定


class Memory:
    def __init__(self, max_size=1000):
        self.buffer = collections.deque(maxlen=max_size)

    def add(self, experience):
        self.buffer.append(experience)

    def sample(self, batch_size):
        idx = np.random.choice(np.arange(len(self.buffer)), size=batch_size, replace=False)
        return [self.buffer[ii] for ii in idx]

    def len(self):
        return len(self.buffer)

# [※p3] Memoryクラスを継承した、TD誤差を格納するクラスです
class Memory_TDerror(Memory):
    def __init__(self, max_size=1000):
        super().__init__(max_size)

    # add, sample, len は継承されているので定義不要

    # TD誤差を取得
    def get_TDerror(self, memory, gamma, mainQN, targetQN):
        (state, action, reward, next_state) = memory.buffer[memory.len() - 1]   #最新の状態データを取り出す
        # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
        next_action = np.argmax(mainQN.model.predict(next_state)[0])  # 最大の報酬を返す行動を選択する
        target = reward + gamma * targetQN.model.predict(next_state)[0][next_action]
        TDerror = target - targetQN.model.predict(state)[0][action]
        return TDerror

    # TD誤差をすべて更新
    def update_TDerror(self, memory, gamma, mainQN, targetQN):
        for i in range(0, (self.len() - 1)):
            (state, action, reward, next_state) = memory.buffer[i]  # 最新の状態データを取り出す
            # 価値計算（DDQNにも対応できるように、行動決定のQネットワークと価値観数のQネットワークは分離）
            next_action = np.argmax(mainQN.model.predict(next_state)[0])  # 最大の報酬を返す行動を選択する
            target = reward + gamma * targetQN.model.predict(next_state)[0][next_action]
            TDerror = target - targetQN.model.predict(state)[0][action]
            self.buffer[i] = TDerror

    # TD誤差の絶対値和を取得
    def get_sum_absolute_TDerror(self):
        sum_absolute_TDerror = 0
        for i in range(0, (self.len() - 1)):
            sum_absolute_TDerror += abs(self.buffer[i]) + 0.0001  # 最新の状態データを取り出す

        return sum_absolute_TDerror



class Actor:
    def __init__(self, action_size, max_episodes):
        self.action_size = action_size
        self.epsilon = 0.9
        self.decay = 0.999
        self.min_epsilon = 0.05
        self.old_episode = None
        self.max_episodes = max_episodes

    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        #if self.old_episode != episode:
        #    for i in range(episode):
        #        self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
        #    print("epsilon:{}".format(self.epsilon))
        #epsilon = 0.001 + 0.9 / (1.0+episode*0.005)

        # max_episodes以上で0.1になるようにする
        if episode <= self.max_episodes:
            epsilon = 1 - 0.9/self.max_episodes*episode
        elif episode > self.max_episodes:
            epsilon = 0.1

        if epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice([i for i in range(self.action_size)])  # ランダムに行動する

        self.old_episode = episode
        return action