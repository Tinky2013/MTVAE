import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pyro.distributions as dist

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """
    def __init__(self, sizes, final_activation=None):
        layers = []
        '''
        example: size=[3,4,2]
        layer dim: 3->4; 4->2
        '''
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)  # remove the activation function in the last layer
        if final_activation is not None:
            layers.append(final_activation)
        super().__init__(*layers)

    def append(self, layer):
        assert isinstance(layer, nn.Module)
        self.add_module(str(len(self)), layer)

class DistributionNet(nn.Module):
    """
    Base class for distribution nets.
    """

    @staticmethod
    def get_class(dtype):
        """
        Get a subclass by a prefix of its name, e.g.::
            assert DistributionNet.get_class("bernoulli") is BernoulliNet
        """
        for cls in DistributionNet.__subclasses__():
            if cls.__name__.lower() == dtype + "net":
                return cls
        raise ValueError("dtype not supported: {}".format(dtype))

class BernoulliNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a single ``logits`` value.

    This is used to represent a conditional probability distribution of a
    single Bernoulli random variable conditioned on a ``sizes[0]``-sized real
    value, for example::

        net = BernoulliNet([3, 4])
        z = torch.randn(3)
        logits, = net(z)
        t = net.make_dist(logits).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])
    def forward(self, x):
        logits = self.fc(x).squeeze(-1).clamp(min=-10, max=10)
        return (logits,)
    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)

class MultivariateBernoulliNet(DistributionNet):
    def __init__(self, sizes, k):
        assert len(sizes) >= 1
        super().__init__()
        self.k = k
        self.fc = FullyConnected(sizes + [self.k])
    def forward(self, x):
        logits = self.fc(x)
        logits =  nn.functional.softplus(logits).clamp(min=-10, max=10)
        return logits
    @staticmethod
    def make_dist(logits):
        return dist.Bernoulli(logits=logits)

class CategoricalNet(DistributionNet):
    def __init__(self, sizes, k):
        assert len(sizes) >= 1
        super().__init__()
        self.k = k
        self.fc = FullyConnected(sizes + [self.k])
        self.softmax = nn.Softmax(dim=0)
    def forward(self, x):
        cat = self.fc(x).clamp(min=-1e6, max=1e6) # cat: (batch_size, k)
        return self.softmax(torch.Tensor(cat))
    @staticmethod
    def make_dist(value):
        return dist.Categorical(value)

class ExponentialNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``rate``.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = ExponentialNet([3, 4])
        x = torch.randn(3)
        rate, = net(x)
        y = net.make_dist(rate).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [1])

    def forward(self, x):
        scale = nn.functional.softplus(self.fc(x).squeeze(-1)).clamp(min=1e-3, max=1e6)
        rate = scale.reciprocal()
        return (rate,)


    @staticmethod
    def make_dist(rate):
        return dist.Exponential(rate)

class LaplaceNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Laplace random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = LaplaceNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale


    @staticmethod
    def make_dist(loc, scale):
        return dist.Laplace(loc, scale)


class NormalNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    single Normal random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = NormalNet([3, 4])
        x = torch.randn(3)
        loc, scale = net(x)
        y = net.make_dist(loc, scale).sample()
    """
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        # Normal has two parameters, input: (in_size, out_size, num_par)
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale

    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)

class NormalMeanNet(DistributionNet):
    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        # Only the mean parameter is learned
        self.fc = FullyConnected(sizes + [1])
    def forward(self, x):
        loc = self.fc(x).squeeze(-1).clamp(min=-1e6, max=1e6)
        return (loc,)
    @staticmethod
    def make_dist(loc):
        return dist.Normal(loc, 1)

class MultivariateBetaNet(DistributionNet):
    def __init__(self, sizes, k):
        assert len(sizes) >= 1
        super().__init__()
        self.k = k
        self.fc_alpha = FullyConnected(sizes + [self.k])
        self.fc_beta = FullyConnected(sizes + [self.k])
    def forward(self, x):
        alpha = self.fc_alpha(x)
        beta = self.fc_beta(x)
        alpha = nn.functional.softplus(alpha).clamp(min=1e-3, max=1e6)
        beta = nn.functional.softplus(beta).clamp(min=1e-3, max=1e6)
        return alpha, beta
    @staticmethod
    def make_dist(alpha, beta):
        return dist.Beta(alpha, beta)

class StudentTNet(DistributionNet):
    """
    :class:`FullyConnected` network outputting a constrained ``df,loc,scale``
    triple, with shared ``df > 1``.

    This is used to represent a conditional probability distribution of a
    single Student's t random variable conditioned on a ``sizes[0]``-size real
    value, for example::

        net = StudentTNet([3, 4])
        x = torch.randn(3)
        df, loc, scale = net(x)
        y = net.make_dist(df, loc, scale).sample()
    """

    def __init__(self, sizes):
        assert len(sizes) >= 1
        super().__init__()
        self.fc = FullyConnected(sizes + [2])
        self.df_unconstrained = nn.Parameter(torch.tensor(0.0))

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        df = nn.functional.softplus(self.df_unconstrained).add(1).expand_as(loc)
        return df, loc, scale


    @staticmethod
    def make_dist(df, loc, scale):
        return dist.StudentT(df, loc, scale)


class DiagNormalNet(nn.Module):
    """
    :class:`FullyConnected` network outputting a constrained ``loc,scale``
    pair.

    This is used to represent a conditional probability distribution of a
    ``sizes[-1]``-sized diagonal Normal random variable conditioned on a
    ``sizes[0]``-size real value, for example::

        net = DiagNormalNet([3, 4, 5])
        z = torch.randn(3)
        loc, scale = net(z)
        x = dist.Normal(loc, scale).sample()

    This is intended for the latent ``z`` distribution and the prewhitened
    ``x`` features, and conservatively clips ``loc`` and ``scale`` values.
    """

    def __init__(self, sizes):
        assert len(sizes) >= 2
        self.dim = sizes[-1]
        super().__init__()
        self.fc = FullyConnected(sizes[:-1] + [self.dim * 2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., : self.dim].clamp(min=-1e2, max=1e2)
        scale = (
            nn.functional.softplus(loc_scale[..., self.dim :]).add(1e-3).clamp(max=1e2)
        )
        return loc, scale