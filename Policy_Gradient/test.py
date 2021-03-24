import torch
import torch.nn.functional as F

from model import DenseNet


class Tester:
    def __init__(self, args):
        super().__init__()
        self.env = args.env
        n_states = args.env.observation_space.shape[0]
        n_actions = args.env.action_space.n

        self.model = DenseNet(n_states, n_actions, args.emb_dim)
        trained_model = torch.load(args.model_path)
        self.model.load_state_dict(trained_model['state_dict'])

    def choose_action(self, s):
        s = torch.unsqueeze(torch.FloatTensor(s), 0)  # [1, n_states]
        logits = self.model(s)  # [1, n_actions]
        probs = F.softmax(logits, 1)
        action = torch.multinomial(probs, 1)
        return action.item()

    def test(self):
        episode_r, s = 0, self.env.reset()

        while True:
            self.env.render()
            a = self.choose_action(s)
            s_, r, done, _ = self.env.step(a)
            episode_r += r
            s = s_
            if done:
                break

            print(f'Current Reward:{episode_r:.03f}')