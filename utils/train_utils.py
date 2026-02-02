import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4):
        """
        计算运行时的均值和标准差 (Welford's algorithm)
        Args:
            shape: 统计量的形状，对于 Critic 的 Return 通常是 (1,)
            epsilon: 防止除零的小数
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4)) # 防止初始除零
        self.epsilon = epsilon
        self.shape = shape

    def update(self, x):
        """
        接收一个 Batch 的数据 x，更新内部的 mean 和 var
        x: [Batch, ...]
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """根据 Welford 算法合并两组统计量"""
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        
        new_var = M2 / tot_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return torch.sqrt(self.var + self.epsilon)