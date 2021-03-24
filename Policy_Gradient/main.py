import gym
import os
import re
import argparse
import pickle
import torch
import datetime


from train import PolicyGradientTrainer
from test import Tester
from paint import Painter


class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--n_episodes', type=int, default=500, help='train episodes')
        parser.add_argument('--emb_dim', type=list, default=[20, 20, 20, 20], help='dim of embedding layers')
        parser.add_argument('--gamma', type=float, default=0.95, help='decline factor for step reward')
        parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
        parser.add_argument('--step_size', type=int, default=100, help='step size in lr_scheduler for optimizer')
        self.parser = parser

    def parse(self):
        arg = self.parser.parse_args(args=[])
        return arg


if __name__ == '__main__':
    current_time = re.sub(r'\D', '', str(datetime.datetime.now())[0:-7])
    if not os.path.exists('./checkpoints/' + current_time):
        os.makedirs('./checkpoints/' + current_time)
    record_path = './checkpoints/' + current_time + '/data.pkl'
    n_repeat = 10

    args = Options().parse()
    args.env = gym.make('CartPole-v1')
    args.model_path = './checkpoints/' + current_time + '/model.pth.tar'
    '''
        env_name: CartPole-v1
        states  : (位置x, x加速度, 偏移角度theta, 角加速度)
        actions : (向左 0, 向右 1)
    '''
    best_reward = 0.0
    rewards = []
    store_flag = True

    for i in range(n_repeat):
        print(f'========== Repeated Experiment # {i:02d} ===========')
        trainer = PolicyGradientTrainer(args)
        reward = trainer.train()
        rewards.append(reward)
        d = {'episode': range(args.n_episodes), 'reward': rewards}
        with open(record_path, 'wb') as f:
            pickle.dump(d, f, pickle.HIGHEST_PROTOCOL)

        estimate_reward = torch.median(torch.tensor(rewards[-10:]))
        if best_reward < estimate_reward:
            # store the first trained agent in the repeated experiments
            state = {'state_dict': trainer.model.state_dict(), 'estimate_reward': estimate_reward}
            torch.save(state, args.model_path)
            best_reward = estimate_reward

    painter = Painter(record_path)
    painter.paint()

    tester = Tester(args)
    tester.test()



