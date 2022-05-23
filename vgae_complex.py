import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.metrics import roc_auc_score, average_precision_score
import time
import pandas as pd
import os

import numpy as np
import scipy.sparse as sp

def set_seed(seed):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def sparse_to_tuple(sparse_mx):
	# the matrix needs to be coo
	if not sp.isspmatrix_coo(sparse_mx):
		sparse_mx = sparse_mx.tocoo()
	#sparse_mx: {(row_index, col_index, value)}
	coords = np.vstack((sparse_mx.row, sparse_mx.col)).transpose()
	values = sparse_mx.data
	shape = sparse_mx.shape
	return coords, values, shape

def preprocess_graph(adj):
	adj = sp.coo_matrix(adj)    #csr_matrix转成coo_matrix
	adj_ = adj + sp.eye(adj.shape[0])   #S=A+I  #注意adj_的类型为csr_matrix
	rowsum = np.array(adj_.sum(1))    #rowsum的shape=(节点数,1)，对于cora数据集来说就是(2078,1)，sum(1)求每一行的和
	degree_mat_inv_sqrt = sp.diags(np.power(rowsum, -0.5).flatten())    #计算D^{-0.5}
	#p.diags：提取输入矩阵(大小为m×n)的所有非零对角列。输出的大小为 min(m,n)×p，其中p表示输入矩阵的p个非零对角列
	#numpy.power()：用于数组元素求n次方
	#flatten()：返回一个折叠成一维的数组。
	adj_normalized = adj_.dot(degree_mat_inv_sqrt).transpose().dot(degree_mat_inv_sqrt).tocoo()
	# adj_.dot(degree_mat_inv_sqrt)得到 SD^{-0.5}
	# adj_.dot(degree_mat_inv_sqrt).transpose()得到(D^{-0.5})^{T}S^{T}=D^{-0.5}S，因为D和S都是对称矩阵
	# adj_normalized即为D^{-0.5}SD^{-0.5}
	return sparse_to_tuple(adj_normalized)

def mask_test_edges(adj):
	# Function to build test set with 10% positive links

	# Remove diagonal elements
	adj = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)	# A-D
	adj.eliminate_zeros()
	# Check that diag is zero:
	assert np.diag(adj.todense()).sum() == 0

	adj_triu = sp.triu(adj) # extract non-zero values in upper triangle matrix, return coo_matrix
	adj_tuple = sparse_to_tuple(adj_triu)	# (coords, values, shape)
	edges = adj_tuple[0]	# coords (<i,j>), do not record repeated edges (due to the upper tri), do not record diag
	edges_all = sparse_to_tuple(adj)[0]	# coords (<i,j> + <j,i>), all edges
	num_test = int(np.floor(edges.shape[0] / 10.))  # 10% edges for test set
	num_val = int(np.floor(edges.shape[0] / 20.))   # 5% edges for val set

	all_edge_idx = list(range(edges.shape[0]))	# no repeated edges (<i,j>)
	np.random.shuffle(all_edge_idx) # random all_edge_idx order
	# allocate id's to different set
	val_edge_idx = all_edge_idx[:num_val]
	test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
	test_edges = edges[test_edge_idx]
	val_edges = edges[val_edge_idx]
	train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)

	def ismember(a, b, tol=5):
		rows_close = np.all(np.round(a - b[:, None], tol) == 0, axis=-1)
		# np.round返回浮点数x的四舍五入值，第二参数是保留的小数的位数
		# b[:, None]使b从shape=(边数,2)变为shape=(边数,1,2)，而a是长度为2的list，a - b[:, None]触发numpy的广播机制
		# np.all()判断给定轴向上的所有元素是否都为True，axis=-1（此时等同于axis=2）表示3维数组最里层的2维数组的每一行的元素是否都为True
		return np.any(rows_close)
		# np.any()判断给定轴向上是否有一个元素为True,现在不设置axis参数则是判断所有元素中是否有一个True，有一个就返回True。
		# rows_close的shape=(边数,1)
		# 至此，可以知道，ismember( )方法用于判断随机生成的<a,b>这条边是否是已经真实存在的边，如果是，则返回True，否则返回False

	test_edges_false = []
	while len(test_edges_false) < len(test_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], edges_all):	# 这个随机负样本要在原图中不存在
			continue
		if test_edges_false:
			if ismember([idx_j, idx_i], np.array(test_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(test_edges_false)):
				continue
		test_edges_false.append([idx_i, idx_j])

	val_edges_false = []
	while len(val_edges_false) < len(val_edges):
		idx_i = np.random.randint(0, adj.shape[0])
		idx_j = np.random.randint(0, adj.shape[0])
		if idx_i == idx_j:
			continue
		if ismember([idx_i, idx_j], edges_all):	# 这个随机负样本要在原图中不存在
			continue
		if ismember([idx_i, idx_j], train_edges):
			continue
		if ismember([idx_j, idx_i], train_edges):
			continue
		if ismember([idx_i, idx_j], val_edges):
			continue
		if ismember([idx_j, idx_i], val_edges):
			continue
		if val_edges_false:
			if ismember([idx_j, idx_i], np.array(val_edges_false)):
				continue
			if ismember([idx_i, idx_j], np.array(val_edges_false)):
				continue
		val_edges_false.append([idx_i, idx_j])

	assert ~ismember(val_edges, train_edges)
	assert ~ismember(test_edges, train_edges)
	assert ~ismember(val_edges, test_edges)
	assert ~ismember(test_edges_false, edges_all)
	assert ~ismember(val_edges_false, edges_all)


	data = np.ones(train_edges.shape[0])

	# 重建出用于训练阶段的邻接矩阵
	adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
	adj_train = adj_train + adj_train.T

	# #注意：这些边列表只包含一个方向的边（adj_train是矩阵，不是edge lists）
	return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def glorot_init(input_dim, output_dim):
	init_range = np.sqrt(6.0/(input_dim + output_dim))
	initial = torch.rand(input_dim, output_dim)*2*init_range - init_range
	return nn.Parameter(initial)

def dot_product_decode(Z):
	A_pred = torch.sigmoid(torch.matmul(Z,Z.t()))
	return A_pred

class GraphConvSparse(nn.Module):
	def __init__(self, input_dim, output_dim, adj, activation, **kwargs):
		super(GraphConvSparse, self).__init__(**kwargs)
		self.weight = glorot_init(input_dim, output_dim)
		self.adj = adj
		self.activation = activation

	def forward(self, inputs):
		x = inputs	# N*D matrix
		x = torch.mm(x, self.weight)
		#torch.mm(a, b)是矩阵a和b矩阵相乘
		x = torch.mm(self.adj, x)
		outputs = self.activation(x)
		return outputs

class VGAE(nn.Module):
	def __init__(self, adj):
		super(VGAE,self).__init__()
		self.base_gcn = GraphConvSparse(PARAM['input_dim'], PARAM['hidden1_dim'], adj, activation = F.relu)
		self.gcn_mean = GraphConvSparse(PARAM['hidden1_dim'], PARAM['hidden2_dim'], adj, activation=lambda x:x)
		self.gcn_logstddev = GraphConvSparse(PARAM['hidden1_dim'], PARAM['hidden2_dim'], adj, activation=lambda x:x)

	def encode(self, X):
		hidden = self.base_gcn(X)
		self.mean = self.gcn_mean(hidden)
		self.logstd = self.gcn_logstddev(hidden)
		gaussian_noise = torch.randn(X.size(0),PARAM['hidden2_dim'])
		sampled_z = gaussian_noise*torch.exp(self.logstd) + self.mean
		# 这里使用torch.exp是因为论文中log(sigma)=GCN_{sigma}(X,A)，torch.exp(self.logstd)即torch.exp(log(sigma))得到的是sigma；另外还有mu=GCN_{mu}(X,A).
		# 由于每个节点向量经过GCN后都有且仅有一个节点向量表示，所以呢，方差的对数log(sigma)和节点向量表示的均值mu分别是节点经过GCN_{sigma}(X,A)和GCN_{mu}(X,A)后得到的向量表示本身。
		# 从N(mu,sigma^2)中采样一个样本Z相当于在N(0,1)中采样一个xi，然后Z = mu + xi×sigma
		return sampled_z

	def forward(self, X):
		self.Z = self.encode(X)
		A_pred = dot_product_decode(self.Z)
		return A_pred

def sigmoid(x):
	return 1 / (1 + np.exp(-x))


def get_acc(adj_rec, adj_label):
	labels_all = adj_label.to_dense().view(-1).long()   #long()将数字或字符串转换为一个长整型
	preds_all = (adj_rec > 0.5).view(-1).long()
	accuracy = (preds_all == labels_all).sum().float() / labels_all.size(0)
	return accuracy

def load_data():
	A = np.array(pd.read_csv(PARAM['network']))
	feature = np.array(pd.read_csv(PARAM['feature'])['x'])
	return A, feature

def main():
	set_seed(100)
	def get_scores(edges_pos, edges_neg, adj_rec):
		# Predict on test set of edges
		preds = []
		pos = []
		for e in edges_pos:
			# print(e)
			# print(adj_rec[e[0], e[1]])
			preds.append(sigmoid(adj_rec[e[0], e[1]].item()))
			# item()取出单元素张量的元素值并返回该值，保持原元素类型不变，从而能够保留原来的精度。所以在求loss,以及accuracy rate的时候一般用item()
			pos.append(adj_orig[e[0], e[1]])

		preds_neg = []
		neg = []
		for e in edges_neg:
			preds_neg.append(sigmoid(adj_rec[e[0], e[1]].data))
			neg.append(adj_orig[e[0], e[1]])

		preds_all = np.hstack([preds, preds_neg])
		labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
		roc_score = roc_auc_score(labels_all, preds_all)
		ap_score = average_precision_score(labels_all, preds_all)

		return roc_score, ap_score

	adj, features = load_data()	# adj: np.array(num_nodes, num_nodes), features: np.array(num_nodes, feature_dim)
	if PARAM['has_feature'] == False:
		features = np.ones((PARAM['num_nodes'],PARAM['num_nodes']))

	adj = sp.csr_matrix(adj)
	features = sp.lil_matrix(features)

	adj_orig = adj	# store the original adj matrix
	adj_orig = adj_orig - sp.dia_matrix((adj_orig.diagonal()[np.newaxis, :], [0]), shape=adj_orig.shape)
	# This is A-D
	adj_orig.eliminate_zeros()	# eliminate_zeros for csr matrix
	# remove zeros, return sparse matrix

	adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
	adj = adj_train # adj_matrix for train (csr_matrix)

	# Some preprocessing
	adj_norm = preprocess_graph(adj)
	#返回D^{-0.5}SD^{-0.5}的coords(坐标), data, shape，其中S=A+I

	num_nodes = adj.shape[0]

	features = sparse_to_tuple(features.tocoo())
	#features的类型原为lil_matrix，sparse_to_tuple返回features的coords, data, shape
	num_features = features[2][1]
	features_nonzero = features[1].shape[0]

	# Create Model
	pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
	#注意，adj的每个元素非1即0。pos_weight是用于训练的邻接矩阵中负样本边（既不存在的边）和正样本边的倍数（即比值），这个数值在二分类交叉熵损失函数中用到，
	#如果正样本边所占的比例和负样本边所占比例失衡，比如正样本边很多，负样本边很少，那么在求loss的时候可以提供weight参数，将正样本边的weight设置小一点，负样本边的weight设置大一点，
	#此时能够很好的平衡两类在loss中的占比，任务效果可以得到进一步提升。参考：https://www.zhihu.com/question/383567632
	norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)


	adj_label = adj_train + sp.eye(adj_train.shape[0])  #adj_train是用于训练的邻接矩阵，类型为csr_matrix
	adj_label = sparse_to_tuple(adj_label)
	'''
	torch.sparse.FloatTensor(indices, values, shape)
	indices: coords of non-zero elements (needs to be transposed)
	values: corresponding values
	shape: the shape of the sparse matrix
	'''
	adj_norm = torch.sparse.FloatTensor(torch.LongTensor(adj_norm[0].T),     #其中adj_norm是D^{-0.5}SD^{-0.5}的coords, data, shape
								torch.FloatTensor(adj_norm[1]),
								torch.Size(adj_norm[2]))
	adj_label = torch.sparse.FloatTensor(torch.LongTensor(adj_label[0].T),
								torch.FloatTensor(adj_label[1]),
								torch.Size(adj_label[2]))
	features = torch.sparse.FloatTensor(torch.LongTensor(features[0].T),
								torch.FloatTensor(features[1]),
								torch.Size(features[2]))

	weight_mask = adj_label.to_dense().view(-1) == 1
	# view的参数-1 表示做自适应性调整，如果参数只有一个参数-1,则表示将Tensor变成一维张量。
	weight_tensor = torch.ones(weight_mask.size(0))
	weight_tensor[weight_mask] = pos_weight
	#用于在binary_cross_entropy中设置正样本边的weight。负样本边的weight都为1，正样本边的weight都为pos_weight

	# --------------------------------------------------------------------------------------------

	# init model and optimizer
	# model = getattr(model,args.model)(adj_norm)
	#getattr() 函数用于返回一个对象属性值。
	model = VGAE(adj_norm)
	optimizer = Adam(model.parameters(), lr=PARAM['learning_rate'])

	A_pred = None
	# train model
	for epoch in range(PARAM['num_epochs']):
		t = time.time()

		A_pred = model(features)    #得到的A_pred每个元素不再是非1即0

		# optimization
		optimizer.zero_grad()
		loss = norm*F.binary_cross_entropy(A_pred.view(-1), adj_label.to_dense().view(-1), weight = weight_tensor)
		kl_divergence = 0.5/ A_pred.size(0) * (1 + 2*model.logstd - model.mean**2 - torch.exp(model.logstd)**2).sum(1).mean()
		loss -= kl_divergence
		loss.backward()
		optimizer.step()

		# evaluation
		train_acc = get_acc(A_pred,adj_label)
		val_roc, val_ap = get_scores(val_edges, val_edges_false, A_pred)
		# if epoch % 100 == 0:
		# 	print("Epoch:", '%04d' % (epoch), "train_loss=", "{:.5f}".format(loss.item()),
		# 		  "train_acc=", "{:.5f}".format(train_acc), "val_roc=", "{:.5f}".format(val_roc),
		# 		  "val_ap=", "{:.5f}".format(val_ap),
		# 		  "time=", "{:.5f}".format(time.time() - t))

	# test the model
	test_roc, test_ap = get_scores(test_edges, test_edges_false, A_pred)
	print("End of training!", "test_roc=", "{:.5f}".format(test_roc),
		  "test_ap=", "{:.5f}".format(test_ap))

	emb_path = 'save_emb/vgae_'+str(PARAM['num_nodes'])+'/'+graph+'_zdim_'+str(PARAM['hidden2_dim'])
	if not os.path.isdir(emb_path):
		os.makedirs(emb_path)
	pd.DataFrame(model.Z.detach().numpy()).to_csv(emb_path+'/emb_'+str(i)+'.csv', index=False)

SEED = 100
set_seed(SEED)

if __name__ == "__main__":
	graph = '0_0_3_1_X[0]_Z[0.4]'
	for i in range(11,21):
		PARAM = {
			# model
			'input_dim': 100,  # no features, this equal to num_nodes
			'hidden1_dim': 16,
			'hidden2_dim': 4,
			# train
			'num_epochs': 1000,
			'learning_rate': 0.005,
			# data
			'has_feature': False,
			'feature': 'data/gendt/'+graph+'/gendt_'+str(i)+'.csv',
			'network': 'data/gendt/'+graph+'/net_'+str(i)+'.csv',
			'num_nodes': 100,
		}
		main()
