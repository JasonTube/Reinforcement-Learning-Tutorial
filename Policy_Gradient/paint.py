import matplotlib.pyplot as plt
import seaborn as sns
import pickle


class Painter:
    def __init__(self, file):
        self.file = file

    def paint(self):
        with open(self.file, 'rb') as f:
            data = pickle.load(f)

        x = data['episode']
        y = data['reward']

        sns.set(style='darkgrid', font_scale=1.5)
        sns.tsplot(time=x, data=y, color='r', condition="policy_gradient")

        plt.xlabel('episode')
        plt.ylabel('reward')
        plt.title('Reinforcement Learning')

        plt.show()
