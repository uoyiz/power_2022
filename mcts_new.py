import os
import gym
import copy
import math
import random
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from utils import BatchBuffer
# from model.mcts_model.policy_value_net import PolicyValueNet
# from model.attention_model.hierarchy_actor_critic import Hierarchy_Actor, Critic
from utils import eliminate_orphan_node, load_graph, load_all_graphs, softmax, process_step_reward, \
    choose_graph_from_list

from p_v_net import PolicyValueNet


def reconnect_array(obs):
    new_line_status_array = np.zeros_like(obs.rho)
    disconnected_lines = np.where(obs.line_status == False)[0]
    for line in disconnected_lines[::-1]:
        if not obs.time_before_cooldown_line[line]:
            line_to_reconnect = line  # reconnection
            new_line_status_array[line_to_reconnect] = 1
            break
    return new_line_status_array


def array2action(env, total_array, reconnect_array=None):
    action = env.action_space({'change_bus': total_array[236:413]})
    action._change_bus_vect = action._change_bus_vect.astype(bool)
    if reconnect_array is None:
        return action
    action.update({'set_line_status': reconnect_array})
    return action


class MinMaxStats:
    """A class that holds the min-max values of the tree."""

    def __init__(self, min_value_bound=None, max_value_bound=None):
        self.maximum = min_value_bound if min_value_bound else -float('inf')
        self.minimum = max_value_bound if max_value_bound else float('inf')

    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)

    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            # We normalize only when we have set the maximum and minimum values.
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value


class TreeNode(object):
    """A node in the MCTS tree.

    Each node keeps track of its own value Q, prior probability P, and
    its visit-count-adjusted prior score u.
    """

    def __init__(self, parent, prior_p):
        self._parent = parent
        self._children = {}  # a map from action to TreeNode
        self._n_visits = 0
        self._V = 0
        self._U = 0
        self._P = prior_p
        self._R = 0

    def expand(self, actions, probs, reward, done):
        """Expand tree by creating new children.
        action_priors: a list of tuples of actions and their prior probability
            according to the policy function.
        """
        # expnad node if not at termial state
        # if not done:
        #     for action, prob in action_priors:
        #         if action not in self._children:
        #             self._children[action] = TreeNode(self, prob)
        # record node reward either at terminal or not
        # print(probs.shape)
        probs = probs.tolist()
        if not done:
            for i in range(len(probs)):
                if i not in self._children:
                    self._children[i] = TreeNode(self, probs[i])
        self._R = reward

    def select(self, c_puct):
        """Select action among children that gives maximum action value Q
        plus bonus u(P).
        Return: A tuple of (action, next_node)
        """
        return max(self._children.items(),
                   key=lambda act_node: act_node[1].get_value(c_puct))

    def update(self, leaf_value):
        """Update node values from leaf evaluation.
        leaf_value: the value of subtree evaluation from the current player's
            perspective.
        """
        # Count visit.
        self._n_visits += 1
        # Update Q, a running average of values for all visits.
        self._V += 1.0 * (leaf_value - self._V) / self._n_visits

    def update_recursive(self, leaf_value):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def update_dense_recursive(self, now_value, gamma, min_max_stats):
        """Like a call to update(), but applied recursively for all ancestors.
        """
        # If it is not root, this node's parent should be updated first.
        if self._parent:
            self._parent.update_dense_recursive(self._R + gamma * now_value, gamma, min_max_stats)
        self.update(now_value)
        min_max_stats.update(self._V)

    def get_value(self, c_puct):
        """Calculate and return the value for this node.
        It is a combination of leaf evaluations Q, and this node's prior
        adjusted for its visit count, u.
        c_puct: a number in (0, inf) controlling the relative impact of
            value Q, and prior probability P, on this node's score.
        """
        self._U = (c_puct * self._P *
                   np.sqrt(self._parent._n_visits) / (1 + self._n_visits))

        return self._V + self._U

    def is_leaf(self):
        """Check if leaf node (i.e. no nodes below this have been expanded)."""
        return self._children == {}

    def is_root(self):
        return self._parent is None


class MCTS(object):
    """An implementation of Monte Carlo Tree Search."""

    def __init__(self, config, policy_value_fn, c_puct=5, n_playout=10000, gamma=0.95, discount=True):
        """
        policy_value_fn: a function that takes in a board state and outputs
            a list of (action, probability) tuples and also a score in [-1, 1]
            (i.e. the expected value of the end game score from the current
            player's perspective) for the current player.
        c_puct: a number in (0, inf) that controls how quickly exploration
            converges to the maximum-value policy. A higher value means
            relying on the prior more.
        """
        self.actions = np.load("actions_space.npz")["action_space"]
        self.config = config
        self._root = TreeNode(None, 1.0)
        self._policy = policy_value_fn
        self._c_puct = c_puct
        self._n_playout = n_playout
        ################# add config for dense reward mcts
        self.gamma = gamma

    def _playout(self, state, reward, done, env, min_max_stats):
        """Run a single playout from the root to the leaf, getting a value at
        the leaf and propagating it back through its parents.
        State is modified in-place, so a copy must be provided.
        """
        node = self._root
        while True:
            if node.is_leaf():
                break
            # Greedily select next move.
            # action, node = node.select(self._c_puct)
            action, node = self.my_select_child(node, min_max_stats)
            action_array = array2action(env, self.actions[action]) if action is not None else array2action(env,np.zeros(494),self.reconnect_array(state))
            state, reward, done, info = env.step(action_array)
        # Evaluate the leaf using a network which outputs a list of
        # (action, probability) tuples p and also a score v
        # for the current player.
        actions, action_probs, leaf_value = self._policy(state)
        # change last node value to value predict if not reach the leaf node
        leaf_value = 0.0 if done else leaf_value
        # and then expand child tree
        node.expand(actions, action_probs, reward=
        reward, done=done)
        # Update value and visit count of nodes in this traversal.
        node.update_dense_recursive(leaf_value, self.gamma, min_max_stats)

    def get_move_probs(self, state, reward, done, env, min_max_stats, temp=1e-3):
        """Run all playouts sequentially and return the available actions and
        their corresponding probabilities.
        state: the current game state
        temp: temperature parameter in (0, 1] controls the level of exploration
        """
        # print('------------------ get_move_probs -------------------')
        for n in range(self._n_playout):
            env_copy = copy.deepcopy(env)
            # print('play_time', n)
            self._playout(state, reward, done, env_copy, min_max_stats)
        # calc the move probabilities based on visit counts at the root node
        act_visits = [(act, node._n_visits) for act, node in self._root._children.items()]
        acts, visits = zip(*act_visits)
        act_probs = softmax(1.0 / temp * np.log(np.array(visits) + 1e-10))

        return acts, act_probs

    def update_with_move(self, last_move):
        """Step forward in the tree, keeping everything we already know
        about the subtree.
        """
        if last_move in self._root._children:
            self._root = self._root._children[last_move]
            self._root._parent = None
        else:
            self._root = TreeNode(None, 1.0)

    def my_select_child(self, node, min_max_stats):
        def my_get_ucb(sub_node):
            ############## calculate value
            sub_node._Q = sub_node._R + self.gamma * sub_node._V \
                if not self.config.mcts_node_value_normal else \
                sub_node._R + self.gamma * min_max_stats.normalize(sub_node._V)
            ############## calculate visit
            _c_puct = math.log(
                (sub_node._parent._n_visits + self.config.mcts_c_puct_base + 1) / self.config.mcts_c_puct_base
            ) + self.config.mcts_c_puct_init
            _c_puct *= math.sqrt(sub_node._parent._n_visits) / (sub_node._n_visits + 1)
            sub_node._U = (_c_puct * sub_node._P *
                           np.sqrt(sub_node._parent._n_visits) / (1 + sub_node._n_visits))

            return sub_node._Q + sub_node._U

        return max(node._children.items(), key=lambda act_node: my_get_ucb(act_node[1]))

    def __str__(self):
        return "MCTS"


class MCTSAgent():
    def __init__(self, config, env):
        self.config = config
        self.env = env
        # load graph from saved data
        # self.graph = load_graph(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        # self.graphs = load_all_graphs(self.config.train_graph_path) if self.config.train_with_preload_graph else None
        # self.config.preload_graph_node_num = len(self.graphs[0].nodes())
        # config data buffer
        self.data_buffer = BatchBuffer(config=config)
        # config for record
        self.best_eval_scores = 0.0
        self.play_batch_size = 1
        # config mode / res dir
        self.save_model_dir = config.output_res_dir + '/model/'
        self.save_res_dir = config.output_res_dir + '/res/'
        if not os.path.exists(self.save_model_dir):
            os.makedirs(self.save_model_dir)
        if not os.path.exists(self.save_res_dir):
            os.makedirs(self.save_res_dir)
        if self.config.output_res:
            self.writter = SummaryWriter(self.save_res_dir)
        # config policy value network and optim
        self.policy_value_net = PolicyValueNet()
        self.optimizer = optim.Adam(self.policy_value_net.policy_value_net.parameters(),
                                    weight_decay=config.mcts_l2_const)
        # config for lr decay
        self.lr_multiplier = config.mcts_lr_multiplier
        self.kl_targ = config.mcts_kl_targ
        # config mcts process
        self.mcts = MCTS(config, self.policy_value_net.policy_value, c_puct=config.mcts_c_puct,
                         n_playout=config.mcts_n_playout, gamma=config.reward_gamma,
                         discount=config.mcts_discount_tree_value)
        # normal value to calculate ucb if necessary
        self.min_max_stats = None
        self.actions = self.mcts.actions

    def save_model(self, training_episode):
        torch.save({
            'policy_value_net': self.policy_value_net.state_dict(),
            'lr_multiplier': self.lr_multiplier,
        }, self.save_model_dir + '/model_' + str(training_episode) + '.pth')

    def load_model(self):
        save_model_dict = torch.load(self.config.load_model_path) if torch.cuda.is_available() \
            else torch.load(self.config.load_model_path, map_location=torch.device('cpu'))
        self.policy_value_net.load_state_dict(save_model_dict['policy_value_net'])
        self.lr_multiplier = save_model_dict['lr_multiplier']
        # self.actor_optimizer.load_state_dict(save_model_dict['actor_optimizer'])
        # self.critic.load_state_dict(save_model_dict['critic'])
        # self.critic_optimizer.load_state_dict(save_model_dict['critic_optimizer'])

    ##################### method for play game
    def get_action(self, state, reward, done, test_mode=False, init_state=False):
        # available_nodes = state['available_nodes']
        # the pi vector returned by MCTS as in the alphaGo Zero paper
        move_probs = np.zeros(314)
        if True:
            acts, probs = self.mcts.get_move_probs(
                state, reward, done, env=self.env, min_max_stats=self.min_max_stats,
                temp=self.config.mcts_temperature if not test_mode else 1e-3,
            )  # choose temp 1e-3 for test mode to get argmax action
            move_probs[list(acts)] = probs
            if not test_mode:
                # add Dirichlet Noise for exploration (needed for training stage)
                move = np.random.choice(
                    acts,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                # update the root node and reuse the search tree
                self.mcts.update_with_move(move)
                return move, move_probs
            else:
                # with the default temp=1e-3, it is almost equivalent
                # to choosing the move with the highest prob
                move = np.random.choice(acts, p=probs)
                # reset the root node
                self.mcts.update_with_move(-1)
                return move
        else:
            print("WARNING: not available nodes to choose")

    def reset_player(self):
        self.mcts.update_with_move(-1)

    def start_play_game_collect(self):
        """ start a self-play game using a MCTS player, reuse the search tree,
        and store the self-play data: (state, mcts_probs, z) for training
        """
        state, reward, done = self.env.reset(), 0.0, False
        all_states, actions_probs, all_rewards = [], [], []
        train_episode_reward = 0.0
        init_state = True
        while True:
            action, action_probs = self.get_action(state, reward, done, test_mode=False, init_state=init_state)
            init_state = False
            # store the data
            all_states.append(state)
            actions_probs.append(action_probs)
            # perform a remove action
            action_array = array2action(self.env, self.actions[action]) if action is not None else array2action(
                self.env, np.zeros(494), self.reconnect_array(state))  # action is None means no available nodes
            state, reward, done, info = self.env.step(action_array)
            # store the data
            all_rewards.append(
                reward)
            train_episode_reward += reward
            if done:
                # reset MCTS root node
                self.reset_player()
                # calculate discount reward
                all_discount_rewards, now_discount_value = [], 0.0
                for node_r in reversed(all_rewards):
                    now_discount_value *= self.config.reward_gamma if self.config.mcts_discount_tree_value else 1.0
                    now_discount_value += node_r
                    all_discount_rewards.append(now_discount_value)
                all_discount_rewards.reverse()
                print('all_discount_rewards', all_discount_rewards)
                return all_states, actions_probs, all_discount_rewards, train_episode_reward

    def start_play_game_evaluate(self):
        """start a game for eval"""
        state, reward, done = self.env.reset(choose_graph_from_list(self.graphs)), 0.0, False
        episode_reward = 0.
        init_state = True
        while True:
            action = self.get_action(state, reward, done, test_mode=True, init_state=init_state)
            init_state = False
            # perform a remove action
            state, reward, done, info = self.env.step(action)
            episode_reward = episode_reward + reward
            if done:
                return episode_reward

    def collect_selfplay_data(self, n_games=1):
        """collect self-play data for training"""
        all_train_episode_reward = []
        for i in range(n_games):
            # winner, play_data = self.start_play_game_collect()
            all_states, actions_probs, all_discount_rewards, train_episode_reward = self.start_play_game_collect()
            self.data_buffer.insert_all_data(all_states, actions_probs, all_discount_rewards)
            all_train_episode_reward.append(train_episode_reward)
        # print('train_episode_reward', np.mean(all_train_episode_reward))
        return np.mean(all_train_episode_reward)

    ##################### method for play game

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the trained policy by playing against the pure MCTS player
        Note: this is only for monitoring the progress of training
        """
        ################################# To be checked
        all_episode_rewards = []
        for i in range(n_games):
            episode_rewards = self.start_play_game_evaluate()
            all_episode_rewards.append(episode_rewards)
        print('eval_episode_reward', np.mean(all_episode_rewards))

        return np.mean(all_episode_rewards)

    def policy_update(self):
        """update the policy-value net"""
        batch_states, batch_actions_probs, batch_rewards = \
            self.data_buffer.sample(self.config.batch_size)
        # print('----------------------------')
        # print('batch_features', batch_features.shape)
        # print('batch_adjacency_matrixs', batch_adjacency_matrixs.shape)
        old_probs, old_v = self.policy_value_net.policy_value(batch_states)
        all_value_loss, all_policy_loss, all_entropy = [], [], []
        for i in range(self.config.mcts_update_epochs):
            value_loss, policy_loss, entropy = self.policy_value_net.train_step(
                batch_states=batch_states,
                batch_actions_probs=batch_actions_probs, batch_rewards=batch_rewards,
                optimizer=self.optimizer, lr=self.config.learning_rate * self.lr_multiplier)
            all_value_loss.append(value_loss)
            all_policy_loss.append(policy_loss)
            all_entropy.append(entropy)
            new_probs, new_v = self.policy_value_net.policy_value(batch_states)
            kl = np.mean(np.sum(old_probs * (
                    np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:  # early stopping if D_KL diverges badly
                break
        # adaptively adjust the learning rate
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5
        # print('----------------------------')
        # print('old_v', old_v.shape)
        # print('new_v', new_v.shape)
        # print('batch_rewards', batch_rewards.shape)
        explained_var_old = (1 - np.var(np.array(batch_rewards) - old_v.flatten()) /
                             np.var(np.array(batch_rewards)))
        explained_var_new = (1 - np.var(np.array(batch_rewards) - new_v.flatten()) /
                             np.var(np.array(batch_rewards)))
        print(("kl:{:.5f},"
               "lr_multiplier:{:.3f},"
               "value_loss:{},"
               "policy_loss:{},"
               "entropy:{},"
               "explained_var_old:{:.3f},"
               "explained_var_new:{:.3f}"
               ).format(kl, self.lr_multiplier, value_loss, policy_loss, entropy, explained_var_old, explained_var_new))

        return np.mean(all_value_loss), np.mean(all_policy_loss), np.mean(all_entropy)

    def train(self):
        self.min_max_stats = MinMaxStats()
        ###################### load model
        if self.config.load_model:
            self.load_model()
        all_train_episode_reward = []
        # run the training pipeline
        for i in range(self.config.total_episodes):
            ###################### collect data
            train_episode_reward = self.collect_selfplay_data(self.play_batch_size)
            print("batch i:{}, episode_reward:{}".format(i + 1, train_episode_reward))
            all_train_episode_reward.append(train_episode_reward)
            ###################### update network
            # print('self.data_buffer.get_buffer_size()', self.data_buffer.get_buffer_size())
            if self.data_buffer.get_buffer_size() >= self.config.batch_size:
                all_value_loss, all_policy_loss, all_entropy = self.policy_update()
                self.output_res({
                    'all_value_loss': all_value_loss, 'all_policy_loss': all_policy_loss,
                    'train_episode_reward': np.mean(all_train_episode_reward),
                }, i)
                all_train_episode_reward = []
            ###################### eval model
            if self.config.add_eval_stage and i % self.config.eval_freq_in_train == 0:
                eval_episode_reward = self.policy_evaluate(n_games=self.config.eval_episodes)
                self.output_res({'eval_episode_reward': eval_episode_reward}, i)
            ###################### save model
            if self.config.save_model and i % self.config.save_model_freq == 0:
                self.save_model(i)

    def output_res(self, train_infos, total_num_steps):
        if not self.config.output_res:
            return
        for k, v in train_infos.items():
            self.writter.add_scalars(k, {k: v}, total_num_steps)

    def eval_graph(self):
        if self.config.load_model:
            self.load_model()
        state, reward, done = self.env.reset(choose_graph_from_list(self.graphs)), 0.0, False
        episode_reward, init_state, step = 0., True, 0
        while True:
            # output step connectivity
            self.output_res({'connectivity': state['connectivity'], }, step)
            step += 1
            # get action from tree-search
            action = self.get_action(state, reward, done, test_mode=True, init_state=init_state)
            init_state = False
            # perform a remove action
            state, reward, done, info = self.env.step(action)
            episode_reward = episode_reward + reward
            if done:
                print('episode_reward', episode_reward)
                return episode_reward

