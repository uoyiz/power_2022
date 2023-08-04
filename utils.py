import os
import ray
import copy
import torch
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


######################## method for graph process
def gen_graph(num_min, num_max, g_type):
    cur_n = np.random.randint(num_max - num_min + 1) + num_min
    if g_type == 'erdos_renyi':
        g_graph = nx.erdos_renyi_graph(n=cur_n, p=0.14)
    elif g_type == 'powerlaw':
        g_graph = nx.powerlaw_cluster_graph(n=cur_n, m=1, p=0.05)
    elif g_type == 'small-world':
        g_graph = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif g_type == 'barabasi_albert':
        g_graph = nx.barabasi_albert_graph(n=cur_n, m=2)
    return g_graph


def load_graph(eval_graph_path):
    graph = nx.convert_node_labels_to_integers(
        nx.read_graphml(eval_graph_path), first_label=0, ordering='default') \
        if eval_graph_path is not None else None
    # nodes = np.array([i for i in graph.nodes()])
    # print('load_graph', nodes)

    return graph


def load_all_graphs(eval_graph_dir):
    filelist = os.listdir(eval_graph_dir)
    graph_list = []
    for filename in filelist:
        eval_graph_path = os.path.join(eval_graph_dir, filename)
        if os.path.isfile(eval_graph_path) and eval_graph_path.endswith('graphml'):
            graph_list.append(load_graph(eval_graph_path))
    return graph_list


def choose_graph_from_list(graphs):
    if graphs is None:
        return None
    index = np.random.randint(0, len(graphs), size=1)[0]
    return copy.deepcopy(graphs[index])


def render_graph(graph, pos=None, path=None):
    render = nx.spring_layout(graph) if pos is None else pos
    nx.draw(graph, render, font_size=4, node_color='black')
    nx.draw_networkx_edges(graph, render, width=1, edge_color='b')
    if path is not None:
        plt.savefig(path)
    plt.show()

    return render


def gen_graph_alltypes(num_min, num_max):
    all_graph_types = [
        'erdos_renyi', 'powerlaw', 'small-world', 'barabasi_albert'
    ]
    graph_type = all_graph_types[np.random.choice(np.arange(4), 1)[0]]
    cur_n = np.random.randint(num_max - num_min + 1) + num_min
    if graph_type == 'erdos_renyi':
        g_graph = nx.erdos_renyi_graph(n=cur_n, p=0.15)
    elif graph_type == 'powerlaw':
        g_graph = nx.powerlaw_cluster_graph(n=cur_n, m=4, p=0.05)
    elif graph_type == 'small-world':
        g_graph = nx.connected_watts_strogatz_graph(n=cur_n, k=8, p=0.1)
    elif graph_type == 'barabasi_albert':
        g_graph = nx.barabasi_albert_graph(n=cur_n, m=4)

    return g_graph


######################## method for state / action / reward process
def get_config_from_env(config, env):
    config.feature_num = env.feature_num


def process_step_reward(reward, n_nodes, reward_reflect=False, reward_normalization=False, multi_step=False):
    if reward_reflect:
        return reward * n_nodes + 1 if not multi_step else \
            np.mean(np.array(reward['all']) * n_nodes + 1)
    elif reward_normalization:
        return reward if not multi_step else \
            np.sum(np.array(reward['all']))
    else:
        return reward['discount']


def eliminate_orphan_node(obs):
    nodes, adjacency_list, adjacency_matrix = obs['nodes'], obs['adjacency_list'], obs['adjacency_matrix']
    new_nodes, new_adjacency_list = [], {}
    keep_nodes = np.zeros_like(nodes).astype(np.bool)
    for node in nodes:
        if len(adjacency_list[node]) > 0:
            new_nodes.append(node)
            new_adjacency_list[node] = adjacency_list[node]
            keep_nodes[node] = True
    obs['nodes'], obs['adjacency_list'] = np.array(new_nodes), new_adjacency_list
    obs['adjacency_matrix'] = adjacency_matrix[keep_nodes][:, keep_nodes]


######################## method for deep learning
def softmax(x):
    probs = np.exp(x - np.max(x))
    probs /= np.sum(probs)
    return probs


class BatchBuffer:
    def __init__(self, config):
        self.config = config
        self.size = config.batch_size
        # self.size = config.memory_capacity
        # self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions_probs = []
        self.all_rewards = []
        # pointer
        self.step = 0
        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(
            range(1164, 1223))
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)

    def get_buffer_size(self):
        return len(self.all_obs)

    def insert_all_data(self, all_states, all_actions_probs, all_rewards):
        for state, action_prob, reward in zip(all_states, all_actions_probs, all_rewards):
            state = state.to_vect()[self.chosen]
            if len(self.all_obs) >= self.size:
                self.all_obs[self.step] = state
                self.all_actions_probs[self.step] = action_prob
                self.all_rewards[self.step] = reward
                self.step = (self.step + 1) % self.size
            else:
                self.all_obs.append(state)
                self.all_actions_probs.append(action_prob)
                self.all_rewards.append(reward)

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.all_obs), batch_size, replace=False)
        print(self.all_obs[0].shape)
        # split all obs into different types
        batch_states = torch.from_numpy(np.array(
            [self.all_obs[i] for i in idxes])).to(dtype=torch.float32, device=self.device)
        # actions_probs, rewards
        batch_actions_probs = torch.cat([
            torch.from_numpy(np.array(self.all_actions_probs[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)

        return batch_states, batch_actions_probs, batch_rewards




@ray.remote
class ShareBuffer:
    def __init__(self, config):
        self.config = config
        # self.size = config.batch_size
        self.size = config.memory_capacity
        self.device = config.device
        # config all training data
        self.all_obs = []
        self.all_actions_probs = []
        self.all_rewards = []
        # pointer
        self.step = 0

    def get_buffer_size(self):
        return len(self.all_obs)

    def insert_all_data(self, all_states, all_actions_probs, all_rewards):
        for state, action_prob, reward in zip(all_states, all_actions_probs, all_rewards):
            if len(self.all_obs) >= self.size:
                self.all_obs[self.step] = state
                self.all_actions_probs[self.step] = action_prob
                self.all_rewards[self.step] = reward
                self.step = (self.step + 1) % self.size
            else:
                self.all_obs.append(state)
                self.all_actions_probs.append(action_prob)
                self.all_rewards.append(reward)

    def sample(self, batch_size):
        idxes = np.random.choice(len(self.all_obs), batch_size, replace=False)
        print(self.all_obs[0].shape)
        # split all obs into different types
        batch_states = torch.from_numpy(np.array(
            [self.all_obs[i] for i in idxes])).to(dtype=torch.float32, device=self.device)
        # actions_probs, rewards
        batch_actions_probs = torch.cat([
            torch.from_numpy(np.array(self.all_actions_probs[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(device=self.device)
        batch_rewards = torch.cat([
            torch.from_numpy(np.array(self.all_rewards[i])).unsqueeze(0) for i in idxes
        ], dim=0).to(dtype=torch.float32, device=self.device).unsqueeze(-1)

        return batch_states, batch_actions_probs, batch_rewards
