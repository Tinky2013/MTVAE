import torch
import numpy as np
import pandas as pd
import scipy.sparse as sp
import random
from torch_geometric.nn import VGAE
import os
import torch
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid
import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv
from torch_geometric.utils import train_test_split_edges
from torch_geometric.transforms import RandomLinkSplit

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

def main():

	def train():
		model.train()
		optimizer.zero_grad()
		z = model.encode(x, train_pos_edge_index)
		loss = model.recon_loss(z, train_pos_edge_index)
		loss = loss + (1 / data.num_nodes) * model.kl_loss()  # new line
		loss.backward()
		optimizer.step()
		return float(loss)

	def test(pos_edge_index, neg_edge_index, save_emb=False):
		model.eval()
		with torch.no_grad():
			z = model.encode(x, train_pos_edge_index)
		if save_emb == True:
			if PARAM['has_feature']==True:
				emb_path = 'save_emb/vgae/' + graph + 'X_zdim_' + str(PARAM['z_dim'])
			else:
				emb_path = 'save_emb/vgae/' + graph + '_zdim_' + str(PARAM['z_dim'])
			if not os.path.isdir(emb_path):
				os.makedirs(emb_path)
			pd.DataFrame(z.detach().numpy()).to_csv(emb_path + '/emb_' + str(i) + '.csv', index=False)
		return model.test(z, pos_edge_index, neg_edge_index)

	A = np.array(pd.read_csv(PARAM['network']))
	if PARAM['has_feature']==True:
		features = np.array(pd.read_csv(PARAM['feature'])[PARAM['feature_col']])
	else:
		features = np.eye(PARAM['num_nodes'],PARAM['num_nodes'])
	if len(features.shape) == 1:
		features = features[:,np.newaxis]
	# features: (num_nodes, feature_dim)

	A = sp.coo_matrix(A)
	edge_index = torch.tensor(np.array([A.row, A.col]), dtype=torch.long)
	x = torch.tensor(features, dtype=torch.float)
	data = Data(x=x, y=None, edge_index=edge_index)
	data = train_test_split_edges(data)

	model = VGAE(VariationalGCNEncoder(in_channels=features.shape[1], out_channels=PARAM['z_dim']))

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = model.to(device)
	x = data.x.to(device)
	train_pos_edge_index = data.train_pos_edge_index.to(device)
	optimizer = torch.optim.Adam(model.parameters(), lr=0.01)


	for epoch in range(1, PARAM['num_epochs'] + 1):
		loss = train()
		auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index)
		# if epoch % 100==0:
		# 	print('Epoch: {:03d}, AUC: {:.4f}, AP: {:.4f}'.format(epoch, auc, ap))

	auc, ap = test(data.test_pos_edge_index, data.test_neg_edge_index, save_emb=True)
	print("Finish training VGAE_"+str(i))
	print('AUC: {:.4f}, AP: {:.4f}'.format(auc, ap))
	print("___________________________")

SEED = 100
set_seed(SEED)

if __name__ == "__main__":
	graph = 'B_0_3_0.1_100_N'
	for i in range(11,111):
		PARAM = {
			# model
			'z_dim': 4,
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
