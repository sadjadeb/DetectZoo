"""Test-time MMD statistic from NSG-VD's ``utils/mmd_utils.py`` (copied verbatim).

Upstream: https://github.com/ZSHsh98/NSG-VD (Apache-2.0).
"""

import torch


def Pdist2(x, y):
	"""compute the paired distance between x and y."""
	x_norm = (x ** 2).sum(1).view(-1, 1)
	if y is not None:
		y_norm = (y ** 2).sum(1).view(1, -1)
	else:
		y = x
		y_norm = x_norm.view(1, -1)
	Pdist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
	Pdist[Pdist<0]=0
	return Pdist


def MMD_batch2(Fea, len_s, Fea_org, sigma, sigma0=0.1, epsilon = 10**(-10), is_smooth=True, is_var_computed=True, use_1sample_U=True, coeff_xy=2):
	X = Fea[0:len_s, :] # 400,300
	Y = Fea[len_s:, :] # 2000,300
	if is_smooth:
		X_org = Fea_org[0:len_s, :] # 400,76800
		Y_org = Fea_org[len_s:, :]  # 2000,76800
	L = 1 # generalized Gaussian (if L>1)

	nx = X.shape[0] # 400
	ny = Y.shape[0] # 2000
	Dxx = Pdist2(X, X) # 400,400
	Dyy = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
	# Dyy = Pdist2(Y, Y) # 2000, 1  0
	Dxy = Pdist2(X, Y).transpose(0,1) # 400，2000 => 2000, 400
	if is_smooth:
		Dxx_org = Pdist2(X_org, X_org)
		Dyy_org = torch.zeros(Fea.shape[0] - len_s, 1).to(Dxx.device) # 2000,1  0
		# Dyy_org = Pdist2(Y_org, Y_org) # 1，1  0
		Dxy_org = Pdist2(X_org, Y_org).transpose(0,1)  # 400，2000 => 2000, 400
	# K_Ix = torch.eye(nx).cuda() # 创建单位矩阵
	# K_Iy = torch.eye(ny).cuda() 

	if is_smooth:
		Kx = (1-epsilon) * torch.exp(-(Dxx / sigma0)**L -Dxx_org / sigma) + epsilon * torch.exp(-Dxx_org / sigma) # 400,400
		Ky = (1-epsilon) * torch.exp(-(Dyy / sigma0)**L -Dyy_org / sigma) + epsilon * torch.exp(-Dyy_org / sigma) # 2000,1     值 1
		Kxy = (1-epsilon) * torch.exp(-(Dxy / sigma0)**L -Dxy_org / sigma) + epsilon * torch.exp(-Dxy_org / sigma)# 400,2000
	else:
		Kx = torch.exp(-Dxx / sigma0)
		Ky = torch.exp(-Dyy / sigma0)
		Kxy = torch.exp(-Dxy / sigma0)

	nx = Kx.shape[0] # 400
	# ny = 1 # 2000
	is_unbiased = False
	if 1:
		xx = torch.div((torch.sum(Kx)), (nx * nx)) # 400,400 => 1
		yy = Ky.reshape(-1) # 2000
		# yy = torch.div((torch.sum(Ky)), (ny * ny)) # 2000,1 => 2000 1  [1]
		# one-sample U-statistic.

		xy = torch.div(torch.sum(Kxy, dim = 1), (nx )) # 2000
		
		mmd2 = xx - 2 * xy + yy
	return mmd2
