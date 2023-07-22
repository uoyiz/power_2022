# -*- coding: utf-8 -*-
"""
An implementation of the policyValueNet in Tensorflow
Tested in Tensorflow 1.4 and 1.5

@author: Xiang Zhong
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


def set_learning_rate(optimizer, lr):
    """Sets the learning rate to the given value"""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
class PVNet(nn.Module):
    def __init__(self):
        super(PVNet, self).__init__()
        n_cell = 1000
        initializer = torch.nn.init.orthogonal_

        self.layer1 = nn.Linear(in_features=n_cell, out_features=n_cell)
        initializer(self.layer1.weight)
        self.layer1.activation = nn.ReLU()

        self.layer2 = nn.Linear(in_features=n_cell, out_features=n_cell)
        initializer(self.layer2.weight)
        self.layer2.activation = nn.ReLU()

        self.layer3 = nn.Linear(in_features=n_cell, out_features=n_cell)
        initializer(self.layer3.weight)
        self.layer3.activation = nn.ReLU()

        self.layer4 = nn.Linear(in_features=n_cell, out_features=n_cell)
        initializer(self.layer4.weight)
        self.layer4.activation = nn.ReLU()

        self.act_layer = nn.Linear(in_features=n_cell, out_features=208)
        initializer(self.act_layer.weight)

        self.val_hidden_layer = nn.Linear(in_features=208, out_features=64)
        initializer(self.val_hidden_layer.weight)
        self.val_hidden_layer.activation = nn.ReLU()

        self.val_layer = nn.Linear(in_features=64, out_features=1)
        initializer(self.val_layer.weight)

    def forward(self, s):
        s = self.layer1(s)
        s = self.layer2(s)
        s = self.layer3(s)
        s = self.layer4(s)

        l = self.act_layer(s)  # logits
        p = F.softmax(l, dim=-1)  # probability distribution of actions
        vh = self.val_hidden_layer(s)
        v = self.val_layer(vh)  # state value
        u = torch.rand_like(l, dtype=torch.float32)
        a = torch.argmax(l - torch.log(-torch.log(u)), dim=-1)
        a_one_hot = F.one_hot(a, num_classes=l.size(-1))  # important!
        neg_log_p = F.cross_entropy(l, a, reduction='none')  # calculate -log(pi)
        v = v.squeeze(dim=1)
        return a, v, neg_log_p, l


class PolicyValueNet():
    def __init__(self,model_file=None, use_gpu=False):
        # warm start from Junior Student
        self.use_gpu = False
        self.l2_const = 1e-4  # coef of l2 penalty
        self.chosen = list(range(2, 7)) + list(range(7, 73)) + list(range(73, 184)) + list(range(184, 656))
        self.chosen += list(range(656, 715)) + list(range(715, 774)) + list(range(774, 833)) + list(range(833, 1010))
        self.chosen += list(range(1010, 1069)) + list(range(1069, 1105)) + list(range(1105, 1164)) + list(
            range(1164, 1223))
        self.chosen = np.asarray(self.chosen, dtype=np.int32) - 1  # (1221,)
        # the policy value net module
        if use_gpu:
            self.policy_value_net = PVNet().cuda()
        else:
            self.policy_value_net = PVNet()
        self.optimizer = optim.Adam(self.policy_value_net.parameters(),
                                    weight_decay=self.l2_const)
        if model_file:
            net_params = torch.load(model_file)
            self.policy_value_net.load_state_dict(net_params)

    def policy_value(self, state_batch):
        state_batch=state_batch.to_vect()[self.chosen]
        if self.use_gpu:
            state_batch = Variable(torch.FloatTensor(state_batch).cuda())
            action,value,log_act_probs,_  = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return action,act_probs, value.data.cpu().numpy()
        else:
            state_batch = Variable(torch.FloatTensor(state_batch))
            action, value, log_act_probs, _ = self.policy_value_net(state_batch)
            act_probs = np.exp(log_act_probs.data.cpu().numpy())
            return action,act_probs, value.data.numpy()



    def train_step(self, batch_states, batch_actions_probs, batch_rewards, lr):
        """perform a training step"""
        # wrap in Variable
        if self.use_gpu:
            batch_states = Variable(torch.FloatTensor(batch_states).cuda())
            batch_actions_probs = Variable(torch.FloatTensor(batch_actions_probs).cuda())
            batch_rewards = Variable(torch.FloatTensor(batch_rewards).cuda())
        else:
            batch_states = Variable(torch.FloatTensor(batch_states).cuda())
            batch_actions_probs = Variable(torch.FloatTensor(batch_actions_probs).cuda())
            batch_rewards = Variable(torch.FloatTensor(batch_rewards).cuda())
        # zero the parameter gradients
        self.optimizer.zero_grad()
        # set learning rate
        set_learning_rate(self.optimizer, lr)

        # forward
        _,value,log_act_probs,_   = self.policy_value_net(batch_states)
        value_loss = F.mse_loss(value, batch_rewards)  # Mean Squared Error for value loss
        policy_loss = torch.mean(torch.sum(batch_actions_probs * torch.log(batch_actions_probs), dim=1))
        entropy = torch.mean(torch.sum(torch.exp(batch_actions_probs) * batch_actions_probs, dim=1))
        loss = value_loss + policy_loss
        loss.backward()
        self.optimizer.step()
        # calc policy entropy, for monitoring only
        return loss.data[0], entropy.data[0]

    def save_model(self, model_path):
        torch.save(self.state_dict(), model_path)

    def restore_model(self, model_path):
        self.load_state_dict(torch.load(model_path))
