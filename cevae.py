import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import pyro
import pyro.distributions as dist
from pyro import poutine
from pyro.infer import SVI, Trace_ELBO
from pyro.infer.util import torch_item
from pyro.nn import PyroModule
from pyro.optim import ClippedAdam
from pyro.util import torch_isnan

class FullyConnected(nn.Sequential):
    """
    Fully connected multi-layer network with ELU activations.
    """

    def __init__(self, sizes, final_activation=None):
        layers = []
        for in_size, out_size in zip(sizes, sizes[1:]):
            layers.append(nn.Linear(in_size, out_size))
            layers.append(nn.ELU())
        layers.pop(-1)
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
        self.fc = FullyConnected(sizes + [2])

    def forward(self, x):
        loc_scale = self.fc(x)
        loc = loc_scale[..., 0].clamp(min=-1e6, max=1e6)
        scale = nn.functional.softplus(loc_scale[..., 1]).clamp(min=1e-3, max=1e6)
        return loc, scale


    @staticmethod
    def make_dist(loc, scale):
        return dist.Normal(loc, scale)

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

class PreWhitener(nn.Module):
    """
    Data pre-whitener.
    """

    def __init__(self, data):
        super().__init__()
        with torch.no_grad():
            loc = data.mean(0)
            scale = data.std(0)
            scale[~(scale > 0)] = 1.0
            self.register_buffer("loc", loc)
            self.register_buffer("inv_scale", scale.reciprocal())

    def forward(self, data):
        return (data - self.loc) * self.inv_scale


class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        x ~ p(x|z)    # partial noisy observation of z
        t ~ p(t|z)    # treatment, whose application is biased by z
        y ~ p(y|t,z)  # outcome

    Each of these distributions is defined by a neural network.  The ``y``
    distribution is defined by a disjoint pair of neural networks defining
    ``p(y|t=0,z)`` and ``p(y|t=1,z)``; this allows highly imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        super().__init__()
        self.x_nn = DiagNormalNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"]
            + [config["feature_dim"]]
        )
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        # The y network is split between the two t values.
        self.y0_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.y1_nn = OutcomeNet(
            [config["latent_dim"]] + [config["hidden_dim"]] * config["num_layers"]
        )
        self.t_nn = BernoulliNet([config["latent_dim"]])

    def forward(self, x, t=None, y=None, size=None):
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            z = pyro.sample("z", self.z_dist())             # z: (batch_size, latent_dim)
            x = pyro.sample("x", self.x_dist(z), obs=x)     # x: (batch_size, feature_dim)
            t = pyro.sample("t", self.t_dist(z), obs=t)     # t: (batch_size)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)  # y: (batch_size)
        return y

    def y_mean(self, x, t=None):
        with pyro.plate("data", x.size(0)):
            z = pyro.sample("z", self.z_dist())
            x = pyro.sample("x", self.x_dist(z), obs=x)
            t = pyro.sample("t", self.t_dist(z), obs=t)
        return self.y_dist(t, z).mean

    # p(z)
    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
    # p(x|z)
    def x_dist(self, z):
        # z: (batch_size, latent_dim)
        loc, scale = self.x_nn(z)   # loc: (batch_size, feature_dim)
        return dist.Normal(loc, scale).to_event(1)
    # p(y|t,z)
    def y_dist(self, t, z):
        # Parameters are not shared among t values.
        params0 = self.y0_nn(z)
        params1 = self.y1_nn(z)
        t = t.bool()
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)
    # p(t|z)
    def t_dist(self, z):
        (logits,) = self.t_nn(z)
        return dist.Bernoulli(logits=logits)

class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::

        t ~ q(t|x)      # treatment
        y ~ q(y|t,x)    # outcome
        z ~ q(z|y,t,x)  # latent confounder, an embedding

    Each of these distributions is defined by a neural network.  The ``y`` and
    ``z`` distributions are defined by disjoint pairs of neural networks
    defining ``p(-|t=0,...)`` and ``p(-|t=1,...)``; this allows highly
    imbalanced treatment.

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        OutcomeNet = DistributionNet.get_class(config["outcome_dist"])
        super().__init__()
        self.t_nn = BernoulliNet([config["feature_dim"]])
        # The y and z networks both follow an architecture where the first few
        # layers are shared for t in {0,1}, but the final layer is split
        # between the two t values.
        self.y_nn = FullyConnected(
            [config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.y0_nn = OutcomeNet([config["hidden_dim"]])
        self.y1_nn = OutcomeNet([config["hidden_dim"]])
        self.z_nn = FullyConnected(
            [1 + config["feature_dim"]]
            + [config["hidden_dim"]] * (config["num_layers"] - 1),
            final_activation=nn.ELU(),
        )
        self.z0_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])
        self.z1_nn = DiagNormalNet([config["hidden_dim"], config["latent_dim"]])

    def forward(self, x, t=None, y=None, size=None):
        # x.shape: [100,5]
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            '''
            The t and y sites are needed for prediction, and participate in
            the auxiliary CEVAE loss. We mark them auxiliary to indicate they
            do not correspond to latent variables during training.
            '''
            t = pyro.sample("t", self.t_dist(x), obs=t, infer={"is_auxiliary": True})   # t: (batch_size)
            y = pyro.sample("y", self.y_dist(t, x), obs=y, infer={"is_auxiliary": True})# y: (batch_size)
            # The z site participates only in the usual ELBO loss.
            pyro.sample("z", self.z_dist(y, t, x))


    # q(t|x)
    def t_dist(self, x):
        (logits,) = self.t_nn(x)
        return dist.Bernoulli(logits=logits)

    # q(y|t,x)
    def y_dist(self, t, x):
        # The first n-1 layers are identical for all t values.
        hidden = self.y_nn(x)
        # In the final layer params are not shared among t values.
        params0 = self.y0_nn(hidden)
        params1 = self.y1_nn(hidden)
        t = t.bool()
        # the final parameters depends on the value of 't'
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return self.y0_nn.make_dist(*params)

    # q(z|x,t,y)
    def z_dist(self, y, t, x):
        # The first n-1 layers are identical for all t values.
        y_x = torch.cat([y.unsqueeze(-1), x], dim=-1)
        hidden = self.z_nn(y_x)
        # In the final layer params are not shared among t values.
        params0 = self.z0_nn(hidden)
        params1 = self.z1_nn(hidden)
        t = t.bool().unsqueeze(-1)
        params = [torch.where(t, p1, p0) for p0, p1 in zip(params0, params1)]
        return dist.Normal(*params).to_event(1)


class TraceCausalEffect_ELBO(Trace_ELBO):
    """
    Loss function for training a :class:`CEVAE`.
    From [1], the CEVAE objective (to maximize) is::

        -loss = ELBO + log q(t|x) + log q(y|t,x)
    """

    def _differentiable_loss_particle(self, model_trace, guide_trace):
        # Construct -ELBO part.
        blocked_names = [
            name
            for name, site in guide_trace.nodes.items()
            if site["type"] == "sample" and site["is_observed"]
        ]
        blocked_guide_trace = guide_trace.copy()
        for name in blocked_names:
            del blocked_guide_trace.nodes[name]
        loss, surrogate_loss = super()._differentiable_loss_particle(
            model_trace, blocked_guide_trace
        )

        # Add log q terms.
        for name in blocked_names:
            log_q = guide_trace.nodes[name]["log_prob_sum"]
            loss = loss - torch_item(log_q)
            surrogate_loss = surrogate_loss - log_q

        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))


class CEVAE(nn.Module):
    def __init__(self, feature_dim, outcome_dist="bernoulli", latent_dim=20, hidden_dim=200, num_layers=3, num_samples=100):
        config = dict(
            feature_dim=feature_dim,    # dimension of feature 'x'
            latent_dim=latent_dim,      # dimension of latent variable 'z'
            hidden_dim=hidden_dim,      # dimension of hidden layers of FC
            num_layers=num_layers,      # numbers of hidden layers in FCN
            num_samples=num_samples,
        )
        for name, size in config.items():
            if not (isinstance(size, int) and size > 0):
                raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.feature_dim = feature_dim
        self.num_samples = num_samples

        super().__init__()
        self.model = Model(config)
        self.guide = Guide(config)

    def fit(self, x, t, y, num_epochs=100, batch_size=100, learning_rate=1e-3, learning_rate_decay=0.1, weight_decay=1e-4, log_every=100):
        '''
        :param num_epochs: number of training epochs
        :param log_every: log loss each this-many steps.
        :return: list of epoch losses
        '''
        assert x.dim() == 2 and x.size(-1) == self.feature_dim
        assert t.shape == x.shape[:1]
        assert y.shape == y.shape[:1]
        self.whiten = PreWhitener(x)

        dataset = TensorDataset(x, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,"weight_decay": weight_decay,"lrd": learning_rate_decay ** (1 / num_steps),})
        # training using :class: '~pyro.infer.svi.SVI' with the 'TraceCausalEffect_ELBO' loss
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())
        losses = []
        for epoch in range(num_epochs):
            for x, t, y in dataloader:
                x = self.whiten(x)
                loss = svi.step(x, t, y, size=len(dataset)) / len(dataset)
                if log_every and len(losses) % log_every == 0:
                    print("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def ite(self, x, num_samples=None, batch_size=None):
        '''
        computes individual treatment effect for a batch of data 'x'
        :param num_samples: the number of monte carlo samples
        :return: a 'len(x)' sized tensor of estimated effects
        '''
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x.dim() == 2 and x.size(-1) == self.feature_dim

        dataloader = [x] if batch_size is None else DataLoader(x, batch_size=batch_size)
        print("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        for x in dataloader:
            x = self.whiten(x)
            # x: (len(x), feature_dim)
            with pyro.plate("num_particles", num_samples, dim=-2):
                print("num_samples:",num_samples)
                with poutine.trace() as tr, poutine.block(hide=["y", "t"]):
                    self.guide(x)

                with poutine.do(data=dict(t=torch.zeros(()))):
                    y0 = poutine.replay(self.model.y_mean, tr.trace)(x)
                with poutine.do(data=dict(t=torch.ones(()))):
                    y1 = poutine.replay(self.model.y_mean, tr.trace)(x)
            ite = (y1 - y0).mean(0) # ite: (len(x))
            if not torch._C._get_tracing_state():
                print("batch ate = {:0.6g}".format(ite.mean()))
            result.append(ite)
        return torch.cat(result)

    def to_script_module(self):
        """
        Compile this module using :func:`torch.jit.trace_module` ,
        assuming self has already been fit to data.

        :return: A traced version of self with an :meth:`ite` method.
        :rtype: torch.jit.ScriptModule
        """
        self.train(False)
        fake_x = torch.randn(2, self.feature_dim)
        with pyro.validation_enabled(False):
            # Disable check_trace due to nondeterministic nodes.
            result = torch.jit.trace_module(self, {"ite": (fake_x,)}, check_trace=False)
        return result


def generate_data():
    """
    This implements the generative process of [1], but using larger feature and
    latent spaces ([1] assumes ``feature_dim=1`` and ``latent_dim=5``).
    """
    z = dist.Bernoulli(0.5).sample([PARAM['num_data']])
    x = dist.Normal(z, 5 * z + 3 * (1 - z)).sample([PARAM['feature_dim']]).t()
    t = dist.Bernoulli(0.75 * z + 0.25 * (1 - z)).sample()
    y = dist.Bernoulli(logits=3 * (z + 2 * (2 * t - 2))).sample()

    # Compute true ite for evaluation (via Monte Carlo approximation).
    t0_t1 = torch.tensor([[0.0], [1.0]])
    y_t0, y_t1 = dist.Bernoulli(logits=3 * (z + 2 * (2 * t0_t1 - 2))).mean
    true_ite = y_t1 - y_t0
    return x, t, y, true_ite


def main():
    if PARAM['cuda']:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Generate synthetic data.
    pyro.set_rng_seed(PARAM['seed'])
    x_train, t_train, y_train, _ = generate_data()  # torch.Size([1000, 5]) torch.Size([1000]) torch.Size([1000])

    # Train.
    pyro.set_rng_seed(PARAM['seed'])
    pyro.clear_param_store()
    cevae = CEVAE(
        feature_dim=PARAM['feature_dim'],
        latent_dim=PARAM['latent_dim'],
        hidden_dim=PARAM['hidden_dim'],
        num_layers=PARAM['num_layers'],
        num_samples=10,
    )
    cevae.fit(
        x_train,
        t_train,
        y_train,
        num_epochs=PARAM['num_epochs'],
        batch_size=PARAM['batch_size'],
        learning_rate=PARAM['learning_rate'],
        learning_rate_decay=PARAM['learning_rate_decay'],
        weight_decay=PARAM['weight_decay'],
    )

    # Evaluate.
    x_test, t_test, y_test, true_ite = generate_data()
    true_ate = true_ite.mean()
    print("true ATE = {:0.3g}".format(true_ate.item()))
    naive_ate = y_test[t_test == 1].mean() - y_test[t_test == 0].mean()
    print("naive ATE = {:0.3g}".format(naive_ate))
    # if args.jit:
    #     cevae = cevae.to_script_module()
    est_ite = cevae.ite(x_test)
    est_ate = est_ite.mean()
    print("estimated ATE = {:0.3g}".format(est_ate.item()))

PARAM = {
    'description': "Causal Effect Variational Autoencoder",
    'num_data': 1000,
    'feature_dim': 5,
    'latent_dim': 20,
    'hidden_dim': 200,
    'num_layers': 3,
    'num_epochs': 20,
    'batch_size': 100,
    'learning_rate': 1e-3,
    'learning_rate_decay': 0.1,
    'weight_decay': 0.01,
    'seed': 100,
    'cuda': False,
}

if __name__ == "__main__":
    main()