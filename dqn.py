from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
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


class QNetwork:
    def __init__(self, lr=0.01, s_size=59, a_size=3, hidden_size=[1024, 1024, 1024, 1024]):
        self.model = tf.keras.models.Sequential()
        self.model.add(tf.keras.layers.Dense(hidden_size[0], activation='relu', input_dim=s_size))
        self.model.add(tf.keras.layers.Dense(hidden_size[1], activation='relu'))
        self.model.add(tf.keras.layers.Dense(hidden_size[2], activation='relu'))
        self.model.add(tf.keras.layers.Dense(hidden_size[3], activation='relu'))
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


class Actor:
    def __init__(self, action_size):
        self.action_size = action_size
        self.epsilon = 0.9
        self.decay = 0.999
        self.min_epsilon = 0.05
        self.old_episode = None

    def get_action(self, state, episode, mainQN):   # [C]ｔ＋１での行動を返す
        if self.old_episode != episode:
            for i in range(episode):
                self.epsilon = max(self.min_epsilon, self.epsilon * self.decay)
            print("epsilon:{}".format(self.epsilon))

        #epsilon = 0.001 + 0.9 / (1.0+episode)
        if self.epsilon <= np.random.uniform(0, 1):
            retTargetQs = mainQN.model.predict(state)[0]
            action = np.argmax(retTargetQs)  # 最大の報酬を返す行動を選択する
        else:
            action = np.random.choice([i for i in range(self.action_size)])  # ランダムに行動する

        self.old_episode = episode
        return action