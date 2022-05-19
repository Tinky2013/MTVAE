import pandas as pd
import numpy as np
import random

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

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

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

    def __init__(self):
        self.latent_dim = PARAM["latent_dim"]
        self.hidden_dim = PARAM['hidden_dim']
        self.treat_dim = PARAM['treat_dim']
        self.num_layers = PARAM['num_layers']
        self.xbi_dim = PARAM['xbi_dim']
        self.xcon_dim = PARAM['xcon_dim']
        super().__init__()
        # f2
        self.t_nn = MultivariateBetaNet([self.latent_dim] + [self.hidden_dim]*self.num_layers,
                                        self.treat_dim)
        # f3
        self.y_nn = NormalMeanNet([self.latent_dim+self.treat_dim] + [self.hidden_dim]*self.num_layers)
        # f4
        self.xbi_nn = MultivariateBernoulliNet([self.latent_dim] + [self.hidden_dim]*self.num_layers,
                                   self.xbi_dim)
        # f6/f7
        self.xcon_nn = NormalNet([self.latent_dim] + [self.hidden_dim]*self.num_layers + [self.xcon_dim])

    def forward(self, xbi, xcon, t=None, y=None, size=None):
        if size is None:
            size = xbi.size(0)
        # TODO: check subsample
        with pyro.plate("data", size, subsample=xbi):
            z = pyro.sample("z", self.z_dist())                     # z: (batch_size, latend_dim)
            xbi = pyro.sample("xbi", self.xbi_dist(z), obs=xbi)     # xbi: (batch_size, xbi_dim)
            xcon = pyro.sample("xcon", self.xcon_dist(z), obs=xcon) # xcon: (batch_size, xcon_dim)
            t = pyro.sample("t", self.t_dist(z), obs=t)             # t: (batch_size, treat_dim)
            y = pyro.sample("y", self.y_dist(t, z), obs=y)          # y: (batch_size)
            return y

    def y_mean(self, xbi, xcon, t=None):
        # TODO: check x.size(0)
        with pyro.plate("data"):
            z = pyro.sample("z", self.z_dist())
            if z.dim() == 3:
                z = torch.mean(z, 0, False)
            xbi = pyro.sample("x", self.xbi_dist(z), obs=xbi)
            xcon = pyro.sample("x", self.xcon_dist(z), obs=xcon)
            t = pyro.sample("t", self.t_dist(z), obs=t)
        return self.y_dist(t, z).mean

    def get_z(self):
        # TODO: check x.size(0)
        with pyro.plate("data"):
            z = pyro.sample("z", self.z_dist())
        return z

    # p(z)
    def z_dist(self):
        return dist.Normal(0, 1).expand([self.latent_dim]).to_event(1)
    # p(x|z)
    def xbi_dist(self, z):
        logits = self.xbi_nn(z)
        return dist.Bernoulli(logits=logits).to_event(1)
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

    def __init__(self):
        self.latent_dim = PARAM['latent_dim']
        self.hidden_dim = PARAM['hidden_dim']
        self.treat_dim = PARAM['treat_dim']
        self.feature_dim = PARAM['xbi_dim']+PARAM['xcon_dim']
        super().__init__()
        self.t_nn = MultivariateBetaNet([self.feature_dim], self.treat_dim)
        # g4
        self.y_nn = NormalMeanNet([self.feature_dim + self.treat_dim,
                                   self.hidden_dim])
        # g5
        self.z_nn = DiagNormalNet([self.feature_dim + self.treat_dim + 1,
            self.hidden_dim,
            self.latent_dim])

    def forward(self, x_bi, x_con, t=None, y=None, size=None):
        if x_bi.dim()==1:
            x_bi = x_bi.unsqueeze(-1)
        if x_con.dim()==1:
            x_con = x_con.unsqueeze(-1)

        x = torch.cat((x_bi, x_con), 1).float()
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
        Alpha, Beta = self.t_nn(x)  # TODO: beta distribution causes t value nears 1 or 0 infeasible
        return dist.Beta(Alpha, Beta).to_event(1)
    # q(y|t,x)
    def y_dist(self, t, x):
        tx = torch.cat((t,x), dim=1).float()    # concat, tx.shape: (batch_size, feature_dim + treatment_dim)
        (mu_yq,) = self.y_nn(tx)
        return self.y_nn.make_dist(mu_yq)
    # q(z|x,t,y)
    def z_dist(self, y, t, x):
        ytx = torch.cat((y.unsqueeze(1),t,x), dim=1).float()  # concat, ytx: (batch_size, feature_dim + treatment_dim + 1)
        mu, sigma = self.z_nn(ytx)
        return dist.Normal(mu, sigma).to_event(1)

class TraceCausalEffect_ELBO(Trace_ELBO):
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
        return loss, surrogate_loss

    @torch.no_grad()
    def loss(self, model, guide, *args, **kwargs):
        return torch_item(self.differentiable_loss(model, guide, *args, **kwargs))

class MTVAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.xbi_dim = PARAM['xbi_dim']
        self.xcon_dim = PARAM['xcon_dim']
        self.treat_dim = PARAM['treat_dim']
        self.mc_samples = PARAM['mc_samples']

        self.model = Model()
        self.guide = Guide()

    def fit(self, x_bi, x_con, t, y):
        num_epochs, batch_size, learning_rate, learning_rate_decay, weight_decay, log_every = \
            PARAM['num_epochs'], PARAM['batch_size'], PARAM['learning_rate'], PARAM['learning_rate_decay'], PARAM['weight_decay'], PARAM['log_every']
        assert x_bi.dim() == 2 and x_bi.size(-1) == self.xbi_dim
        assert x_con.dim() == 2 and x_con.size(-1) == self.xcon_dim
        assert t.size(-1) == self.treat_dim

        dataset = TensorDataset(x_bi, x_con, t, y)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        print("Training with {} minibatches per epoch".format(len(dataloader)))
        num_steps = num_epochs * len(dataloader)
        optim = ClippedAdam({"lr": learning_rate,"weight_decay": weight_decay,"lrd": learning_rate_decay ** (1 / num_steps),})
        # training using :class : '~pyro.infer.svi.SVI' with the 'Trace_ELBO' loss
        svi = SVI(self.model, self.guide, optim, TraceCausalEffect_ELBO())  # TODO: prepare the ELBO estimator
        losses = []
        for epoch in range(num_epochs):
            for x_bi, x_con, t, y in dataloader:
                loss = svi.step(x_bi, x_con, t, y, size=len(dataset)) / len(dataset)
                if log_every and len(losses) % log_every == 0:
                    print("step {: >5d} loss = {:0.6g}".format(len(losses), loss))
                assert not torch_isnan(loss)
                losses.append(loss)
        return losses

    @torch.no_grad()
    def potential_y(self, x_bi, x_con, t):
        mc_samples = self.mc_samples
        if not torch._C._get_tracing_state():
            assert x_bi.dim() == 2 and x_bi.size(-1) == self.xbi_dim
            assert x_con.dim() == 2 and x_con.size(-1) == self.xcon_dim

        # x_bi: (num_sample, xbi_dim), t: (num_sample, treat_dim)
        d=torch.cat((x_bi, x_con, t),dim=-1) # d: (len(data), feature_dim+treat_dim)
        dataloader = [d]
        print("Evaluating {} minibatches".format(len(dataloader)))
        result = []
        hide_list = ['t'+str(i) for i in range(self.treat_dim)]+['y']
        for d in dataloader:
            # d: (len(data), feature_dim+treat_dim)
            with pyro.plate("num_particles", mc_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=hide_list):
                    self.guide(d[:,0:4],d[:,4])    # xbi, xcon
                with poutine.do(data=dict(t=torch.tensor(d[:,5:]))):
                    y_hat = poutine.replay(self.model.y_mean, tr.trace)(d[:,0:4],d[:,4])

            # y_hat: (len(data))
            # if not torch._C._get_tracing_state():
            #     print("mean y_hat = {:0.6g}".format(y_hat.mean()))
            result.append(y_hat)    # ite: (len(x))
        # len(result)=1
        return torch.cat(result)

    @torch.no_grad()
    def check_z(self, x_bi, x_con, t):
        mc_samples = self.mc_samples
        if not torch._C._get_tracing_state():
            assert x_bi.dim() == 2 and x_bi.size(-1) == self.xbi_dim
            assert x_con.dim() == 2 and x_con.size(-1) == self.xcon_dim

        # x_bi: (num_sample, xbi_dim), t: (num_sample, treat_dim)
        d=torch.cat((x_bi, x_con, t),dim=-1) # d: (len(data), feature_dim+treat_dim)
        dataloader = [d]
        print("getting the embeddings")
        result = []
        hide_list = ['t'+str(i) for i in range(self.treat_dim)]+['y']
        for d in dataloader:
            # d: (len(data), feature_dim+treat_dim)
            with pyro.plate("num_particles", mc_samples, dim=-2):
                with poutine.trace() as tr, poutine.block(hide=hide_list):
                    self.guide(d[:,0:4],d[:,4])    # xbi, xcon
                with poutine.do(data=dict(t=torch.tensor(d[:,5:]))):
                    z = poutine.replay(self.model.get_z, tr.trace)()

            # y_hat: (len(data))
            # if not torch._C._get_tracing_state():
            #     print("mean y_hat = {:0.6g}".format(y_hat.mean()))
            result.append(z)    # ite: (len(x))
        # len(result)=1
        return z


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
    x_bi = torch.tensor(np.array(dt.loc[:,'gender':'cluster_2'])).float()
    x_con = torch.tensor(dt['age']).float()
    if x_bi.dim()==1:
        x_bi = x_bi.unsqueeze(1)
    if x_con.dim()==1:
        x_con = x_con.unsqueeze(1)
    t = torch.tensor(np.array(dt.loc[:,'t0':'t7']))
    y = torch.tensor(dt['y1'])
    return x_bi, x_con, t, y

def main():
    if PARAM['cuda']:
        torch.set_default_tensor_type("torch.cuda.FloatTensor")

    # Generate synthetic data.
    pyro.set_rng_seed(PARAM['seed'])
    x_bi, x_con, t, y = get_dt(PARAM['data'])

    # Train.
    pyro.set_rng_seed(PARAM['seed'])
    pyro.clear_param_store()
    mtvae = MTVAE()
    mtvae.fit(x_bi, x_con, t, y)
    z = mtvae.check_z(x_bi, x_con, t)
    pd.DataFrame(np.mean(z.detach().numpy(),axis=0)).to_csv('save_emb/mtvae_dim_' + str(PARAM['latent_dim']) + '_'+str(PARAM['num_nodes'])+'.csv',
                                                  index=False)

    # Evaluate.
    # x_bi, x_con, t, y = get_dt('data/gendt_test.csv')
    # est_y = mtvae.potential_y(x_bi, x_con, t)    # est_y: (len(data))
    #
    # # estimate the potential outcome
    # true_ave_y = y.float().mean()
    # est_ave_y = est_y.mean()
    # print("true outcome = {:0.3g}".format(true_ave_y.item()))
    # print("estimated Average potential outcome = {:0.3g}".format(est_ave_y.item()))

PARAM = {
    'description': "Multiple Treatment Causal Effect Variational Autoencoder",
    # data
    'data': 'data/gendt_train.csv',
    'num_nodes': 100,
    'xbi_dim': 4,
    'xcon_dim': 1,
    'treat_dim': 8,
    # model
    'latent_dim': 4,
    'hidden_dim': 16,
    'num_layers': 2,
    # train
    'num_epochs': 200,
    'batch_size': 20,
    'learning_rate': 1e-3,
    'learning_rate_decay': 0.1,
    'weight_decay': 1e-4,
    'log_every': 100,
    'seed': 100,
    'cuda': False,
    # eval
    'mc_samples': 200,  # number of monte carlo sample
}

if __name__ == "__main__":
    set_seed(100)
    main()