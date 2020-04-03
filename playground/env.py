import numpy as np
import scipy.stats


class Environment(object):
    """docstring for Environment"""

    def __init__(self):
        super(Environment, self).__init__()

    def load_data_4(self):
        # 4 Quadrants
        x = np.random.uniform(-1, 1, 10000)
        y = np.random.uniform(-1, 1, 10000)
        x = x[x != 0]
        y = y[y != 0]
        label = x * y > 0

        return np.stack([x, y], 1), label

    def load_data(self):
        # 9 Quadrants
        x = [-1.5, 0, 1.5]
        y = [-1.5, 0, 1.5]
        data = []
        label = []
        idx = 1
        for x_axis in x:
            for y_axis in y:
                xx = self.sample(x_axis)
                yy = self.sample(y_axis)
                data.append(np.stack([xx, yy], 1))
                if idx % 2 == 0:
                    label.append(np.ones(xx.shape[0]))
                else:
                    label.append(np.zeros(xx.shape[0]))
                idx += 1

        return np.concatenate(data), np.concatenate(label)

    def sample(self, mu, N=1000):
        lower = mu - 0.45
        upper = mu + 0.45
        sigma = 0.5

        samples = scipy.stats.truncnorm.rvs(
            (lower - mu) / sigma, (upper - mu) / sigma, loc=mu, scale=sigma, size=N)
        return samples


if __name__ == '__main__':
    env = Environment()
    x, y = env.load_data()
