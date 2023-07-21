import pickle
import grid2op
import numpy as np
import os
import datetime
import numpy as np
import tensorflow as tf
from copy import deepcopy
from grid2op.Agent import BaseAgent

from policy_value_net_numpy import PolicyValueNetNumpy
from power_alphaZero import MCTSPlayer
class power():
    def __init__(self, env):
        self.env = env
        self.actions_space = self.load("../actions_space.npz")
    def load(self,path):
        data = np.load(path, allow_pickle=True)
        np.set_printoptions(threshold=np.inf)
        print(data)
        return data

def run(this_directory_path=None):
    n = 5
    width, height = 8, 8
    model_file = 'best_policy_8_8_5.model'
    try:
        board = Board(width=width, height=height, n_in_row=n)
        game = Game(board)

        try:
            policy_param = pickle.load(open(model_file, 'rb'))
        except:
            policy_param = pickle.load(open(model_file, 'rb'),
                                       encoding='bytes')  # To support python3
        ppo = tf.keras.models.load_model(os.path.join(this_directory_path, '../ppo-ckpt'))
        mcts_player = MCTSPlayer(ppo,
                                 c_puct=5,
                                 n_playout=400)
        game.start_play(human, mcts_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\n\rquit')


if __name__ == '__main__':
    run("power_2022")
