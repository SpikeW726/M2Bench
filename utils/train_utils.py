import torch
import torch.nn as nn

class RunningMeanStd(nn.Module):
    def __init__(self, shape=(), epsilon=1e-4):
        """
        Running mean and std using Welford's algorithm
        Args:
            shape: Shape of statistics, for Critic Return usually (1,)
            epsilon: Small constant to prevent division by zero
        """
        super().__init__()
        self.register_buffer("mean", torch.zeros(shape))
        self.register_buffer("var", torch.ones(shape))
        self.register_buffer("count", torch.tensor(1e-4))  # Prevent initial division by zero
        self.epsilon = epsilon
        self.shape = shape

    def update(self, x):
        """
        Update with a batch of data x, update internal mean and var
        x: [Batch, ...]
        """
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.shape[0]

        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        """Merge statistics using Welford's algorithm"""
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