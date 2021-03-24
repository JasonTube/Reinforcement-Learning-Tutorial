import torch
import numpy as np
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR

from model import DenseNet


class PolicyGradientTrainer:
    def __init__(self, args):
        super().__init__()
        self.env = args.env
        n_states = args.env.observation_space.shape[0]
        n_actions = args.env.action_space.n

        self.n_episodes = args.n_episodes

        self.gamma = args.gamma
        self.lr = args.lr

        self.model = DenseNet(n_states, n_actions, args.emb_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = StepLR(self.optimizer, step_size=args.step_size, gamma=0.7)

        self.log_a = []
        self.step_r = []

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)  # [1, n_states]
        logits = self.model(s)  # [1, n_actions]
        probs = F.softmax(logits, 1)
        action = torch.multinomial(probs, 1)  # 根据概率采样
        self.log_a.append(torch.log(probs[0][action].squeeze(0)))  # 保存公式中的log值
        return action.item()

    def store(self, r):
        self.step_r.append(r)

    def learn(self):
        processed_step_r = np.zeros_like(self.step_r)
        tmp = 0
        for i in reversed(range(0, len(self.step_r))):  # 回溯
            tmp = tmp * self.gamma + self.step_r[i]
            processed_step_r[i] = tmp  # 带衰减地按步计算reward

        eps = np.finfo(np.float32).eps.item()  # 获得一个浮点表示的最小数字，防止出现数值故障
        processed_step_r = (processed_step_r - np.mean(processed_step_r)) / (np.std(processed_step_r) + eps)  # normalize
        processed_step_r = torch.FloatTensor(processed_step_r)

        loss = -torch.sum(torch.cat(self.log_a) * processed_step_r)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()

        self.log_a = []
        self.step_r = []

    def train(self):
        reward = []

        for episode in range(self.n_episodes):
            episode_r, s = 0, self.env.reset()

            while True:
                a = self.choose_action(s)
                s_, r, done, _ = self.env.step(a)
                self.store(r)
                episode_r += r
                s = s_
                if done:
                    break

            self.learn()
            print(f'Episode {episode:03d} || Reward:{episode_r:.03f} Learning Rate:{self.scheduler.get_lr()[0]:.2e}')
            reward.append(episode_r)
        return reward
