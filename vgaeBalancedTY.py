import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random

import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit
from torch_geometric.utils import (negative_sampling, remove_self_loops,
                                   add_self_loops)
from torch_geometric.nn.inits import reset
from typing import Any, Optional, Tuple
from torch.autograd import Function
from sklearn.metrics import roc_auc_score, average_precision_score

EPS = 1e-15

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)

class VariationalGCNEncoder(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super(VariationalGCNEncoder, self).__init__()
		self.conv1 = GCNConv(in_channels, 2 * out_channels, cached=True) # cached only for transductive learning
		self.conv_mu = GCNConv(2 * out_channels, out_channels, cached=True)
		self.conv_logstd = GCNConv(2 * out_channels, out_channels, cached=True)

	def forward(self, x, edge_index):
		x = self.conv1(x, edge_index).relu()
		return self.conv_mu(x, edge_index), self.conv_logstd(x, edge_index)

class Balance(torch.nn.Module):
	def __init__(self, out_dim):
		super(Balance,self).__init__()
		self.gNet_inDim = 4, # zdim
		self.gNet = torch.nn.Linear(4, out_dim)
		#self.grl = GRL()

	def forward(self, emb):
		#emb = self.grl(emb)
		t_pred = self.gNet(emb)
		return t_pred

class GradientReverseFunction(Function):
	@staticmethod
	def forward(ctx: Any, input: torch.Tensor, coeff: Optional[float] = 1.) -> torch.Tensor:
		ctx.coeff = coeff
		output = input * 1.0
		return output

	@staticmethod
	def backward(ctx: Any, grad_output: torch.Tensor) -> Tuple[torch.Tensor, Any]:
		return grad_output.neg() * ctx.coeff, None

class GRL(torch.nn.Module):
	def __init__(self):
		super(GRL, self).__init__()

	def forward(self, *input):
		return GradientReverseFunction.apply(*input)

class InnerProductDecoder(torch.nn.Module):
	def forward(self, z, edge_index, sigmoid=True):
		value = (z[edge_index[0]] * z[edge_index[1]]).sum(dim=1)
		return torch.sigmoid(value) if sigmoid else value

	def forward_all(self, z, sigmoid=True):
		adj = torch.matmul(z, z.t())
		return torch.sigmoid(adj) if sigmoid else adj

class VGAE(torch.nn.Module):
	def __init__(self, in_channels, out_channels):
		super().__init__()
		self.encoder = VariationalGCNEncoder(in_channels, out_channels)
		self.decoder = InnerProductDecoder()
		self.balance = Balance(out_dim=1)
		self.balance1 = Balance(out_dim=2)
		VGAE.reset_parameters(self)

	def reset_parameters(self):
		reset(self.encoder)
		reset(self.decoder)
		reset(self.balance)
		reset(self.balance1)

	def reparametrize(self, mu, logstd):
		if self.training:
			return mu + torch.randn_like(logstd) * torch.exp(logstd)
		else:
			return mu

	def encode(self, x, train_pos_edge_index):
		self.__mu__, self.__logstd__ = self.encoder(x, train_pos_edge_index)
		self.__logstd__ = self.__logstd__.clamp(max=10)
		z = self.reparametrize(self.__mu__, self.__logstd__)
		return z

	def decode(self, *args, **kwargs):
		r"""Runs the decoder and computes edge probabilities."""
		return self.decoder(*args, **kwargs)

	def kl_loss(self, mu=None, logstd=None):
		mu = self.__mu__ if mu is None else mu
		logstd = self.__logstd__ if logstd is None else logstd.clamp(
			max=10)
		return -0.5 * torch.mean(
			torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))

	def balance_loss(self, emb, t, y0):
		# emb, embn: (num_nodes, z_dim)
		t_pred = self.balance(emb)
		t =t.unsqueeze(dim=1)

		y0_pred = self.balance1(emb)

		# return F.cross_entropy(t_pred, t.float())
		return F.cross_entropy(y0_pred, y0.long()) * PARAM['y_reg'] #+ F.mse_loss(t_pred, t.float()) * PARAM['t_reg']

	def recon_loss(self, z, pos_edge_index, neg_edge_index=None):
		pos_loss = -torch.log(
			self.decoder(z, pos_edge_index, sigmoid=True) + EPS).mean()

		# Do not include self-loops in negative samples
		pos_edge_index, _ = remove_self_loops(pos_edge_index)
		pos_edge_index, _ = add_self_loops(pos_edge_index)
		if neg_edge_index is None:
			neg_edge_index = negative_sampling(pos_edge_index, z.size(0))
		neg_loss = -torch.log(1 -
							  self.decoder(z, neg_edge_index, sigmoid=True) +
							  EPS).mean()

		return pos_loss + neg_loss

	def test(self, z, pos_edge_index, neg_edge_index):


		pos_y = z.new_ones(pos_edge_index.size(1))
		neg_y = z.new_zeros(neg_edge_index.size(1))
		y = torch.cat([pos_y, neg_y], dim=0)

		pos_pred = self.decoder(z, pos_edge_index, sigmoid=True)
		neg_pred = self.decoder(z, neg_edge_index, sigmoid=True)
		pred = torch.cat([pos_pred, neg_pred], dim=0)

		y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

		return roc_auc_score(y, pred), average_precision_score(y, pred)

def cal_ave_neighbor_z(z, neighbor):
	ave_z = torch.zeros((PARAM['num_nodes'], PARAM['z_dim']))
	for i in range(PARAM['num_nodes']):
		indices = neighbor[i]
		ave_neighbor_z = torch.mean(z[indices], dim=0)
		ave_z[i] = ave_neighbor_z
	return ave_z

def main():
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	t = torch.tensor(np.array(pd.read_csv(PARAM['feature'])['influence_estim'])).to(device)
	y0 = pd.read_csv(PARAM['feature'])['y0']
	y0[y0<0]=0
	y0[y0>0]=1
	y0 = torch.tensor(np.array(y0)).to(device)

	def train(neighbor):
		model.train()
		optimizer.zero_grad()
		z = model.encode(x, train_pos_edge_index)	# z: (num_nodes, z_dim)
		# zn = cal_ave_neighbor_z(z, neighbor).to(device)		# zn: (num_nodes, z_dim), neighbors average embedding
		bal_loss = model.balance_loss(z, t, y0)
		rec_loss = model.recon_loss(z, train_pos_edge_index)
		normKl_loss = (1 / data.num_nodes) * model.kl_loss()
		loss = (rec_loss + normKl_loss) * PARAM['a_reg'] + bal_loss
		#print(rec_loss.item(), bal_loss.item())

		loss.backward()
		optimizer.step()
		return float(loss)

	def test(neighbor, pos_edge_index, neg_edge_index, save_emb=False):
		model.eval()
		with torch.no_grad():
			z = model.encode(x, train_pos_edge_index).detach().cpu()
			zn = cal_ave_neighbor_z(z, neighbor).detach().cpu()
			Z = torch.cat([z,zn],dim=1)
		if save_emb == True:
			if PARAM['has_feature']==True:
				emb_path = 'save_emb/vgae/' + graph + 'A' + str(PARAM['a_reg']) + 'Y' + str(PARAM['y_reg']) + 'X_zdim_' + str(PARAM['z_dim'])
			else:
				emb_path = 'save_emb/vgae/' + graph + 'A' + str(PARAM['a_reg']) + 'Y' + str(PARAM['y_reg']) + '_zdim_' + str(PARAM['z_dim'])
			if not os.path.isdir(emb_path):
				os.makedirs(emb_path)
			# same embedding and neigbors' average embeddings
			pd.DataFrame(Z.numpy()).to_csv(emb_path + '/emb_' + str(i) + '.csv', index=False)
		return model.test(z, pos_edge_index, neg_edge_index)

	A = np.array(pd.read_csv(PARAM['network']))
	if PARAM['has_feature']==True:
		features = np.array(pd.read_csv(PARAM['feature'])[PARAM['feature_col']])
	else:
		features = np.eye(PARAM['num_nodes'],PARAM['num_nodes'])
	if len(features.shape) == 1:
		features = features[:,np.newaxis]
	# features: (num_nodes, feature_dim)

	A, D = sp.coo_matrix(A), sp.coo_matrix(A-np.eye(PARAM['num_nodes']))
	edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
	x = torch.tensor(features, dtype=torch.float)

	neighbor = {i:[] for i in range(PARAM['num_nodes'])}
	for j in range(len(D.col)):
		neighbor[D.row[j]].append(D.col[j])

	data = Data(x=x, y=None, edge_index=edge_index)
	data = train_test_split_edges(data)

	model = VGAE(in_channels=features.shape[1], out_channels=PARAM['z_dim'])

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	x = data.x.to(device)
	train_pos_edge_index = data.train_pos_edge_index.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


	for epoch in range(1, PARAM['num_epochs'] + 1):
		loss = train(neighbor)	# neighbor用于索引计算平均embedding
		auc, ap = test(neighbor, data.test_pos_edge_index, data.test_neg_edge_index)
		# if epoch % 100==0:
		#	print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

	auc, ap = test(neighbor, data.test_pos_edge_index, data.test_neg_edge_index, save_emb=True)
	print("Finish training VGAE_"+str(i))
	print('AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
	print("___________________________")

SEED = 100
set_seed(SEED)

if __name__ == "__main__":
	for ba in [1]:
		for by in [1]:
			graph = 'B_0_3_0.5_100_N'
			for i in range(11,111):
				PARAM = {
					# model
					'z_dim': 4,
					'a_reg': ba,
					'y_reg': by,
					# 't_reg': bt,

					# train
					'num_epochs': 200,
					'learning_rate': 0.01,

					# data
					'feature': 'data/gendt/'+graph+'/gendt_'+str(i)+'.csv',
					'feature_col': 'x',
					'has_feature': False,
					'network': 'data/gendt/'+graph+'/net_'+str(i)+'.csv',
					'num_nodes': 100,
				}
				main()
