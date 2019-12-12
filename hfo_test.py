import os, subprocess, time, signal
import hfo_py
import socket
import numpy as np
import matplotlib.pyplot as plt
import dqn
import math
from contextlib import closing

# 行動リストと大きさ、状態の大きさ
ACTION_LIST = [hfo_py.MOVE, hfo_py.DRIBBLE, hfo_py.SHOOT, hfo_py.REDUCE_ANGLE_TO_GOAL, hfo_py.GO_TO_BALL]
ACTION_SIZE = len(ACTION_LIST)
STATE_SIZE = 59


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


def plot(y_list):
    x = np.linspace(0,200,200)
    y = y_list
    plt.plot(x, y, label="reward")
    # 凡例の表示
    plt.legend()
    # プロット表示(設定の反映)
    plt.show()


# TODO : 環境のクラスを作る
class HFO(hfo_py.HFOEnvironment):
    def __init__(self):
        super().__init__()
        self.server_port = None
        self.server_process = None
        self.viewer = None
        self.old_ball_prox = 0
        self.old_kickable = 0
        self.old_ball_dist_goal = 0
        self.got_kickable_reward = False
        self.first_step = True
        self.state = None
        self.status = None
        self.reward = None
        self.unum = self.getUnum()

    def find_free_port(self):
        """Find a random free port. Does not guarantee that the port will still be free after return.
        Note: HFO takes three consecutive port numbers, this only checks one.

        Source: https://github.com/crowdAI/marLo/blob/master/marlo/utils.py

        :rtype:  `int`
        """

        with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
            s.bind(('', 0))
            return s.getsockname()[1]

    def start_server(self, frames_per_trial=300,
                          #untouched_time=1000,
                          untouched_time=100,
                          offense_agents=1,
                          defense_agents=0, offense_npcs=0,
                          defense_npcs=0, sync_mode=True, port=None,
                          offense_on_ball=0, fullstate=True, seed=-1,
                          ball_x_min=0.0, ball_x_max=0.2,
                          verbose=False, log_game=False,
                          log_dir="log"):
        """
        Starts the Half-Field-Offense server.
        frames_per_trial: Episodes end after this many steps.
        untouched_time: Episodes end if the ball is untouched for this many steps.
        offense_agents: Number of user-controlled offensive players.
        defense_agents: Number of user-controlled defenders.
        offense_npcs: Number of offensive bots.
        defense_npcs: Number of defense bots.
        sync_mode: Disabling sync mode runs server in real time (SLOW!).
        port: Port to start the server on.
        offense_on_ball: Player to give the ball to at beginning of episode.
        fullstate: Enable noise-free perception.
        seed: Seed the starting positions of the players and ball.
        ball_x_[min/max]: Initialize the ball this far downfield: [0,1]
        verbose: Verbose server messages.
        log_game: Enable game logging. Logs can be used for replay + visualization.
        log_dir: Directory to place game logs (*.rcg).
        """
        hfo_path = hfo_py.get_hfo_path()
        if port is None:
            port = self.find_free_port()
        self.server_port = port
        cmd = hfo_path + \
              " --headless --frames-per-trial %i --offense-agents %i"\
              " --defense-agents %i --offense-npcs %i --defense-npcs %i"\
              " --port %i --offense-on-ball %i --seed %i --ball-x-min %f"\
              " --ball-x-max %f --log-dir %s"\
              % (frames_per_trial,
                 offense_agents,
                 defense_agents, offense_npcs, defense_npcs, port,
                 offense_on_ball, seed, ball_x_min, ball_x_max,
                 log_dir)
        if not sync_mode: cmd += " --no-sync"
        if fullstate:     cmd += " --fullstate"
        if verbose:       cmd += " --verbose"
        if not log_game:  cmd += " --no-logging"
        print('Starting server with command: %s' % cmd)
        self.server_process = subprocess.Popen(cmd.split(' '), shell=False)
        time.sleep(10) # Wait for server to startup before connecting a player

    def _start_viewer(self):
        """
        Starts the SoccerWindow visualizer. Note the viewer may also be
        used with a *.rcg logfile to replay a game. See details at
        https://github.com/LARG/HFO/blob/master/doc/manual.pdf.
        """
        cmd = hfo_py.get_viewer_path() +\
              " --connect --port %d" % (self.server_port)
        self.viewer = subprocess.Popen(cmd.split(' '), shell=False)

    def render(self, mode='human', close=False):
        """ Viewer only supports human mode currently. """
        if close:
            if self.viewer is not None:
                os.kill(self.viewer.pid, signal.SIGKILL)
        else:
            if self.viewer is None:
                self._start_viewer()

    def close(self):
        if self.server_process is not None:
            try:
                os.kill(self.server_process.pid, signal.SIGKILL)
                self.render(close=True)
            except Exception:
                pass

    def _step(self, action):
        if ACTION_LIST[action][0] != hfo_py.TURN:
            self.act(ACTION_LIST[action][0], ACTION_LIST[1], ACTION_LIST[2])
            if ACTION_LIST[action][0] == hfo_py.DASH:
                print("action_num:{0}  action:DASH  speed:{1}  degree:{2}".format(action, ACTION_LIST[action][1], ACTION_LIST[action][2]))
            else:
                print("action_num{0}  action:KICK  power:{1}  degree:{2}".format(action, ACTION_LIST[action][1], ACTION_LIST[action][2]))

        else:
            self.act(ACTION_LIST[action][0], ACTION_LIST[1])
            print("action_num]{0}  action:TURN  degree:{1}".format(action, ACTION_LIST[1]))

        self.state = self.getState()
        self.status = self.step()
        self.reward = self.get_reward(self.state)
        return self.state, self.reward, self.status

    # TODO : reward関数の実装
    def get_reward(self, state):
        """
        Agent is rewarded for minimizing the distance between itself and
        the ball, minimizing the distance between the ball and the goal,
        and scoring a goal.
        """
        # print("State =",current_state)
        # print("len State =",len(current_state))
        ball_proximity = state[53]
        goal_proximity = state[15]
        ball_dist = 1.0 - ball_proximity
        goal_dist = 1.0 - goal_proximity
        kickable = state[12]
        ball_ang_sin_rad = state[51]
        ball_ang_cos_rad = state[52]
        ball_ang_rad = math.acos(ball_ang_cos_rad)
        if ball_ang_sin_rad < 0:
            ball_ang_rad *= -1.
        goal_ang_sin_rad = state[13]
        goal_ang_cos_rad = state[14]
        goal_ang_rad = math.acos(goal_ang_cos_rad)
        if goal_ang_sin_rad < 0:
            goal_ang_rad *= -1.
        alpha = max(ball_ang_rad, goal_ang_rad) - min(ball_ang_rad, goal_ang_rad)
        ball_dist_goal = math.sqrt(ball_dist * ball_dist + goal_dist * goal_dist -
                                   2. * ball_dist * goal_dist * math.cos(alpha))
        # Compute the difference in ball proximity from the last step
        if not self.first_step:
            ball_prox_delta = ball_proximity - self.old_ball_prox
            kickable_delta = kickable - self.old_kickable
            ball_dist_goal_delta = ball_dist_goal - self.old_ball_dist_goal
        old_ball_prox = ball_proximity
        old_kickable = kickable
        old_ball_dist_goal = ball_dist_goal
        # print(self.env.playerOnBall())
        # print(self.env.playerOnBall().unum)
        # print(self.env.getUnum())
        reward = 0
        if not self.first_step:
            mtb = self.__move_to_ball_reward(kickable_delta, ball_prox_delta)
            ktg = 3. * self.__kick_to_goal_reward(ball_dist_goal_delta)
            eot = self.__EOT_reward()
            reward = mtb + ktg + eot
            # print("mtb: %.06f ktg: %.06f eot: %.06f"%(mtb,ktg,eot))

        self.first_step = False
        # print("r =",reward)
        return reward

    def __move_to_ball_reward(self, kickable_delta, ball_prox_delta):
        reward = 0.
        if self.playerOnBall().unum < 0 or self.playerOnBall().unum == self.unum:
            reward += ball_prox_delta
        if kickable_delta >= 1 and not self.got_kickable_reward:
            reward += 1.
            self.got_kickable_reward = True
        return reward

    def __kick_to_goal_reward(self, ball_dist_goal_delta):
        if (self.playerOnBall().unum == self.unum):
            return -ball_dist_goal_delta
        elif self.got_kickable_reward == True:
            return 0.2 * -ball_dist_goal_delta
        return 0.

    def __EOT_reward(self):
        if self.status == hfo_py.GOAL:
            return 5.
        # elif self.status == hfo_py.CAPTURED_BY_DEFENSE:
        #    return -1.
        return 0.

# TODO : パラメーター化されたアクションに対応させる
def main():
    num_episodes = 200 # 総試行回数
    max_timestep = 15000
    num_consecutive_iterations = 10  # 学習完了評価の平均計算を行う試行回数
    ep_reward = []
    gamma = 0.99  # 割引係数
    islearned = 0  # 学習が終わったフラグ

    hidden_size = 100  # Q-networkの隠れ層のニューロンの数
    learning_rate = 0.00001  # Q-networkの学習係数
    memory_size = 10000  # バッファーメモリの大きさ
    batch_size = 32  # Q-networkを更新するバッチの大記載

    # pact生成
    gen_pact()

    # モデルを初期化
    mainQN = dqn.QNetwork(lr=learning_rate, s_size=STATE_SIZE, a_size=ACTION_SIZE)
    targetQN = dqn.QNetwork(lr=learning_rate, s_size=STATE_SIZE, a_size=ACTION_SIZE)
    memory = dqn.Memory(max_size=memory_size)
    actor = dqn.Actor(ACTION_SIZE)

    hfo = HFO()
    hfo.start_server()
    hfo.connectToServer(feature_set=hfo_py.LOW_LEVEL_FEATURE_SET, config_dir=hfo_py.get_config_path(), server_port=hfo.server_port)
    hfo.render()

    # エピソードのループ
    for episode in range(num_episodes):
        state, reward, status = hfo._step(np.random.choice([i for i in range(ACTION_SIZE)]))
        state = np.reshape(state, [1, STATE_SIZE])
        episode_reward = 0
        targetQN.model.set_weights(mainQN.model.get_weights())
        # タイムステップをどのくらい取るのか
        for timestep in range(max_timestep):
            action = actor.get_action(state, episode, mainQN)
            next_state, reward, status = hfo._step(action)
            next_state = np.reshape(state, [1, STATE_SIZE])

            memory.add((state, action, reward, next_state))  # メモリの更新する
            state = next_state  # 状態更新

            # Qネットワークの重みを学習・更新する replay
            if (memory.len() > batch_size) and not islearned:
                mainQN.replay(memory, batch_size, gamma, targetQN)

            targetQN.model.set_weights(mainQN.model.get_weights())
            episode_reward += reward

            # statusがIN_GAMEじゃなかったらエピソード終了
            if timestep % 100 == 0 and timestep != 0:
                print("timestep%d"%(timestep))
            if status != hfo_py.IN_GAME:
                break
        ep_reward.append(episode_reward)
        print("episode:{0}, total_reward:{1}".format(episode, episode_reward))
    hfo.close()
    plot(ep_reward)

"""
DQNの実装
状態数12、行動３、ハイパラメーターどうする？
実装で使うフレームワーク：Pytorch
"""

if __name__ == "__main__":
    main()

"""
An enum of the possible HFO actions, including:
  [Low-Level] Dash(power, relative_direction)
  [Low-Level] Turn(direction)
  [Low-Level] Tackle(direction)
  [Low-Level] Kick(power, direction)
  [Mid-Level] Kick_To(target_x, target_y, speed)
  [Mid-Level] Move(target_x, target_y)
  [Mid-Level] Dribble(target_x, target_y)
  [Mid-Level] Intercept(): Intercept the ball
  [High-Level] Move(): Reposition player according to strategy
  [High-Level] Shoot(): Shoot the ball
  [High-Level] Pass(teammate_unum): Pass to teammate
  [High-Level] Dribble(): Offensive dribble
  [High-Level] Catch(): Catch the ball (Goalie Only)
  NOOP(): Do Nothing
  QUIT(): Quit the game
"""
"""
Possible game statuses:
  [IN_GAME] Game is currently active
  [GOAL] A goal has been scored by the offense
  [CAPTURED_BY_DEFENSE] The defense has captured the ball
  [OUT_OF_BOUNDS] Ball has gone out of bounds
  [OUT_OF_TIME] Trial has ended due to time limit
  [SERVER_DOWN] Server is not alive
"""
"""
High Level State Feature List
    0 X position
    1 Y position
    2 Orientation
    3 Ball X
    4 Ball Y
    5 Able to Kick
    6 Goal Center Proximity
    7 Goal Center Angle
    8 Goal Opening Angle
    9 Proximitu to Opponent 
    10 Last_Action_Success_Possible
    11 Stamina
"""