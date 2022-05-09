import pandas as pd
import numpy as np

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

from util import NormalNet, DistributionNet,\
    NormalMeanNet, BernoulliNet, MultivariateBetaNet, CategoricalNet, MultivariateBernoulliNet, DiagNormalNet


class Model(PyroModule):
    """
    Generative model for a causal model with latent confounder ``z`` and binary
    treatment ``t``::

        z ~ p(z)      # latent confounder
        x ~ p(x|z)    # partial noisy observation of z
        t ~ p(t|z)    # treatment, whose application is biased by z
        y ~ p(y|t,z)  # outcome

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.treat_dim = config['treat_dim']
        super().__init__()
        # f2
        self.t_nn = MultivariateBetaNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"], self.treat_dim
        )
        # f3
        self.y_nn = NormalMeanNet(
            [config["latent_dim"] + config['treat_dim']]
            + [config["hidden_dim"]] * config["num_layers"]
        )
        # f4
        self.xbi_nn = BernoulliNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"]
        )
        # f5
        self.xcat_nn = CategoricalNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"], 3
        ) # TODO: Num of classes in categorical variables
        # f6/f7
        self.xcon_nn = NormalNet(
            [config["latent_dim"]]
            + [config["hidden_dim"]] * config["num_layers"]
            + [config["xcon_dim"]]
        )

    def forward(self, xbi, xcat, xcon, t=None, y=None, size=None):
        if size is None:
            size = xbi.size(0)
        # TODO: check subsample
        with pyro.plate("data", size, subsample=xbi):
            z = pyro.sample("z", self.z_dist())                     # z: (batch_size, latend_dim)
            xbi = pyro.sample("xbi", self.xbi_dist(z), obs=xbi)     # xbi: (batch_size, xbi_dim)
            xcat = pyro.sample("xcat", self.xcat_dist(z), obs=xcat) # xcat: (batch_size, xcat_dim)
            xcon = pyro.sample("xcon", self.xcon_dist(z), obs=xcon) # xcon: (batch_size, xcon_dim)
            t = pyro.sample("t", self.t_dist(z), obs=t)             # t: (batch_size, treat_dim)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)          # y: (batch_size)
        return y

    def y_mean(self, xbi, xcat, xcon, t=None):
        # TODO: check x.size(0)
        with pyro.plate("data"):
            z = pyro.sample("z", self.z_dist())
            if z.dim() == 3:
                z = torch.mean(z, 0, False)
            xbi = pyro.sample("x", self.xbi_dist(z), obs=xbi)
            xcat = pyro.sample("x", self.xcat_dist(z), obs=xcat)
            xcon = pyro.sample("x", self.xcon_dist(z), obs=xcon)
            t = pyro.sample("t", self.t_dist(z), obs=t)
        return self.y_dist(t, z).mean

    # p(z)
    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
    # p(x|z)
    def xbi_dist(self, z):
        (logits,) = self.xbi_nn(z)
        return dist.Bernoulli(logits=logits).to_event(1)
    def xcat_dist(self, z):
        cat = self.xcat_nn(z)
        return dist.Categorical(cat).to_event(1)
    def xcon_dist(self, z):
        loc, scale = self.xcon_nn(z)
        return dist.Normal(loc, scale).to_event(1)
    # p(t|z)
    def t_dist(self, z):
        Alpha, Beta = self.t_nn(z)
        return dist.Beta(Alpha, Beta).to_event(1)
    # p(y|t,z)
    def y_dist(self, t, z):
        tz = torch.cat((t, z), dim=1).float()
        (mu_yp,) = self.y_nn(tz)
        return self.y_nn.make_dist(mu_yp)

class Guide(PyroModule):
    """
    Inference model for causal effect estimation with latent confounder ``z``
    and binary treatment ``t``::

        t ~ q(t|x)      # treatment
        y ~ q(y|t,x)    # outcome
        z ~ q(z|y,t,x)  # latent confounder, an embedding

    :param dict config: A dict specifying ``feature_dim``, ``latent_dim``,
        ``hidden_dim``, ``num_layers``, and ``outcome_dist``.
    """

    def __init__(self, config):
        self.latent_dim = config["latent_dim"]
        self.treat_dim = config['treat_dim']
        super().__init__()
        # TODO: check whether g3 suitable
        self.t_nn = MultivariateBetaNet(
            [config["feature_dim"]], self.treat_dim
        )
        # g4
        self.y_nn = NormalMeanNet(
            [config["feature_dim"]+config["treat_dim"], config['hidden_dim']]
        )
        # g5
        self.z_nn = DiagNormalNet(
            [config["feature_dim"]+config["treat_dim"]+1,
             config["hidden_dim"],
            config["latent_dim"]]
        )

    def forward(self, x_bi, x_cat, x_con, t=None, y=None, size=None):
        if x_bi.dim()==1:
            x_bi = x_bi.unsqueeze(-1)
        if x_cat.dim()==1:
            x_cat = x_cat.unsqueeze(-1)
        if x_con.dim()==1:
            x_con = x_con.unsqueeze(-1)

        x = torch.cat((x_bi, x_cat, x_con), 1).float()
        # x.shape: (batch_size, feature_dim)
        if size is None:
            size = x.size(0)
        with pyro.plate("data", size, subsample=x):
            '''
            The t and y sites are needed for prediction, and participate in
            the auxiliary CEVAE loss. We mark them auxiliary to indicate they
            do not correspond to latent variables during training.
            '''
            t = pyro.sample("t", self.t_dist(x), obs=t, infer={"is_auxiliary": True})   # t: (batch_size, treat_dim)
            ## TODO: dimension for monte carlo sample
            if t.dim() == 3:
                t = torch.mean(t, 0, False)
            y = pyro.sample("y", self.y_dist(t, x), obs=y, infer={"is_auxiliary": True})# y: (batch_size)
            # The z site participates only in the usual ELBO loss.
            pyro.sample("z", self.z_dist(y, t, x))

    # q(t|x)
    def t_dist(self, x):
        Alpha, Beta = self.t_nn(x)
        return dist.Beta(Alpha, Beta).to_event(1)
    # q(y|t,x)
    def y_dist(self, t, x):
        tx = torch.cat((t,x), dim=1).float()    # concat, tx.shape: (batch_size, feature_dim + treatment_dim)
        (mu_yq,) = self.y_nn(tx)
        return self.y_nn.make_dist(mu_yq)
    # q(z|x,t,y)
    def z_dist(self, y, t, x):
        ytx = torch.cat((y.unsqueeze(1),t,x), dim=1).float()  # concat
        mu, sigma = self.z_nn(ytx)
        return dist.Normal(mu, sigma).to_event(1)

class MTVAE(nn.Module):
    def __init__(self, xbi_dim, xcat_dim, xcon_dim, treat_dim,
                 outcome_dist="bernoulli", latent_dim=20, hidden_dim=200, num_layers=3, num_samples=100):
        super().__init__()
        config = dict(
            # dimension of 'x'
            xbi_dim=xbi_dim,
            xcat_dim=xcat_dim,
            xcon_dim=xcon_dim,
            feature_dim=xbi_dim+xcat_dim+xcon_dim,
            treat_dim=treat_dim,        # dimension of treatment 't'
            latent_dim=latent_dim,      # dimension of latent variable 'z'
            hidden_dim=hidden_dim,      # dimension of hidden layers of FC
            num_layers=num_layers,      # numbers of hidden layers in FCN
            num_samples=num_samples,
        )
        for name, size in config.items():
            if not (isinstance(size, int) and size > 0):
                raise ValueError("Expected {} > 0 but got {}".format(name, size))
        config["outcome_dist"] = outcome_dist
        self.xbi_dim = xbi_dim
        self.xcat_dim = xcat_dim
        self.xcon_dim = xcon_dim
        self.treat_dim = treat_dim
        self.num_samples = num_samples

        self.model = Model(config)
        self.guide = Guide(config)

    def fit(self, x_bi, x_cat, x_con, t, y,
            num_epochs=100, batch_size=100, learning_rate=1e-3, learning_rate_decay=0.1, weight_decay=1e-4, log_every=100):
        '''
        :param num_epochs: number of training epochs
        :param log_every: log loss each this-many steps.
        :return: list of epoch losses
        '''
        assert x_bi.dim() == 2 and x_bi.size(-1) == self.xbi_dim
        assert x_cat.dim() == 2 and x_cat.size(-1) == self.xcat_dim
        assert x_con.dim() == 2 and x_con.size(-1) == self.xcon_dim
        assert t.size(-1) == self.treat_dim
        dataset = TensorDataset(x_bi, x_cat, x_con, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,"weight_decay": weight_decay,"lrd": learning_rate_decay ** (1 / num_steps),})
        # training using :class: '~pyro.infer.svi.SVI' with the 'Trace_ELBO' loss
        svi = SVI(self.model, self.guide, optim, Trace_ELBO())  # TODO: prepare the ELBO estimator
        losses = []
        for epoch in range(num_epochs):
            for x_bi, x_cat, x_con, t, y in dataloader:
                loss = svi.step(x_bi, x_cat, x_con, t, y, size=len(dataset)) / len(dataset)
                if log_every and len(losses) % log_every == 0:
                    print("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def potential_y(self, x_bi, x_cat, x_con, t, num_samples=None, batch_size=None):
        '''
        computes the potential outcome 'y' for a batch of data 'x'
        :param num_samples: the number of monte carlo samples
        :return: a 'len(x)' sized tensor of estimated values
        '''
        if num_samples is None:
            num_samples = self.num_samples
        if not torch._C._get_tracing_state():
            assert x_bi.dim() == 2 and x_bi.size(-1) == self.xbi_dim
            assert x_cat.dim() == 2 and x_cat.size(-1) == self.xcat_dim
            assert x_con.dim() == 2 and x_con.size(-1) == self.xcon_dim

        # x_bi: (num_sample, xbi_dim), t: (num_sample, treat_dim)
        d=torch.cat((x_bi, x_cat, x_con, t),dim=-1) # d: (num_sample, feature_dim+treat_dim)
        dataloader = [d]
        print("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        hide_list = ['t'+str(i) for i in range(self.treat_dim)]+['y']
        for d in dataloader:
            # d: (num_sample, feature_dim+treat_dim)
            with pyro.plate("num_particles", num_samples, dim=-2):
                print("num_samples:", num_samples)
                with poutine.trace() as tr, poutine.block(hide=hide_list):
                    self.guide(d[:,0],d[:,1],d[:,2])    # xbi, xcat, xcon

                with poutine.do(data=dict(t=torch.tensor(d[:,3:]))):
                    y_hat = poutine.replay(self.model.y_mean, tr.trace)(d[:,0],d[:,1],d[:,2])

            if not torch._C._get_tracing_state():
                print("batch y_pot = {:0.6g}".format(y_hat.mean()))
            result.append(y_hat)    # ite: (len(x))
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

def get_dt(path):
    dt = pd.read_csv(path)
    x_bi = torch.tensor(dt['gender']).float()
    x_cat = torch.tensor(dt['cluster']).float()
    x_con = torch.tensor(dt['age']).float()
    if x_bi.dim()==1:
        x_bi = x_bi.unsqueeze(1)
    if x_cat.dim()==1:
        x_cat = x_cat.unsqueeze(1)
    if x_con.dim()==1:
        x_con = x_con.unsqueeze(1)
    t = torch.tensor(np.array(dt.loc[:,'t0':'t7']))
    y = torch.tensor(dt['y'])
    return x_bi, x_cat, x_con, t, y

def main():
    if PARAM['cuda']:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Generate synthetic data.
    pyro.set_rng_seed(PARAM['seed'])

    x_bi, x_cat, x_con, t, y = get_dt('data/network_train.csv')

    # Train.
    pyro.set_rng_seed(PARAM['seed'])
    pyro.clear_param_store()
    mtvae = MTVAE(
        xbi_dim=PARAM['xbi_dim'],
        xcat_dim=PARAM['xcat_dim'],
        xcon_dim=PARAM['xcon_dim'],
        treat_dim=PARAM['treat_dim'],
        latent_dim=PARAM['latent_dim'],
        hidden_dim=PARAM['hidden_dim'],
        num_layers=PARAM['num_layers'],
        num_samples=10,
    )
    mtvae.fit(x_bi, x_cat, x_con, t, y,
        num_epochs=PARAM['num_epochs'],
        batch_size=PARAM['batch_size'],
        learning_rate=PARAM['learning_rate'],
        learning_rate_decay=PARAM['learning_rate_decay'],
        weight_decay=PARAM['weight_decay'],
    )

    # Evaluate.
    x_bi, x_cat, x_con, t, y = get_dt('data/network_eval.csv')

    # estimate the potential outcome
    true_y_mean = y.mean()
    print("true outcome = {:0.3g}".format(true_y_mean.item()))
    est_y = mtvae.potential_y(x_bi, x_cat, x_con, t)
    est_ave_y = est_y.mean()
    print("estimated Average potential outcome = {:0.3g}".format(est_ave_y.item()))

PARAM = {
    'description': "Multiple Treatment Causal Effect Variational Autoencoder",
    'num_data': 100,
    'xbi_dim': 1,
    'xcat_dim': 1,
    'xcon_dim': 1,
    'treat_dim': 8,
    'latent_dim': 20,
    'hidden_dim': 200,
    'num_layers': 3,
    'num_epochs': 20,
    'batch_size': 20,
    'learning_rate': 1e-3,
    'learning_rate_decay': 0.1,
    'weight_decay': 0.01,
    'seed': 100,
    'cuda': False,
}

if __name__ == "__main__":
    main()