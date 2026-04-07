import numpy as np

# 用于状态空间动态归一化
@DeprecationWarning
class RunningNorm:
    def __init__(self, state_dim, epsilon=1e-8):
        self.mean = np.zeros(state_dim)
        self.var = np.ones(state_dim)
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean, self.var, self.count = new_mean, new_var, tot_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RunningMeanStd:
    # Dynamically calculate mean and std
    def __init__(self, shape, epsilon=1e-8):
        self.n = 0
        self.mean = np.zeros(shape, dtype=np.float64)
        self.S = np.zeros(shape, dtype=np.float64)   # 实际上是累计二阶中心矩 M2
        self.std = np.ones(shape, dtype=np.float64)
        self.epsilon = epsilon

    def update(self, x):
        x = np.asarray(x, dtype=np.float64)
        self.n += 1

        if self.n == 1:
            self.mean = x.copy()
            self.S = np.zeros_like(x, dtype=np.float64)
            self.std = np.ones_like(x, dtype=np.float64)
            return

        old_mean = self.mean.copy()
        self.mean = old_mean + (x - old_mean) / self.n
        self.S = self.S + (x - old_mean) * (x - self.mean)

        var = self.S / self.n
        self.std = np.sqrt(np.maximum(var, self.epsilon))


# Reward Scaling
class RewardScaling:
    def __init__(self, shape, gamma):
        self.shape = shape  # reward shape=1
        self.gamma = gamma  # discount factor
        self.running_ms = RunningMeanStd(shape=self.shape)
        self.R = np.zeros(self.shape)

    def __call__(self, x):
        self.R = self.gamma * self.R + x
        self.running_ms.update(self.R)
        x = x / (self.running_ms.std + 1e-8)  # Only divided std
        return x

    def reset(self):  # When an episode is done,we should reset 'self.R'
        self.R = np.zeros(self.shape)


class Normalization:
    def __init__(self, shape):
        self.running_ms = RunningMeanStd(shape=shape)

    def __call__(self, x, update=True):
        # Whether to update the mean and std,during the evaluating,update=Flase
        if update:
            self.running_ms.update(x)
        x = (x - self.running_ms.mean) / (self.running_ms.std + 1e-8)

        return x