import logging
import sys
import functools
import os

import numpy as np
from scipy import linalg # For numpy FID
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter as P
from torchvision.models.inception import inception_v3
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

from template_lib.d2.utils import comm

from .build import GAN_METRIC_REGISTRY

__all__ = ['PyTorchFIDISScore']

# Module that wraps the inception network to enable use with dataparallel and
# returning pool features and logits.
class WrapInception(nn.Module):
  def __init__(self, net):
    super(WrapInception,self).__init__()
    self.net = net
    self.mean = P(torch.tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1),
                  requires_grad=False)
    self.std = P(torch.tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1),
                 requires_grad=False)
  def forward(self, x):
    # Normalize x
    x = (x + 1.) / 2.0
    x = (x - self.mean) / self.std
    # Upsample if necessary
    if x.shape[2] != 299 or x.shape[3] != 299:
      if torch.__version__ in ['0.4.0']:
        x = F.upsample_bilinear(x, size=(299, 299))
      else:
        x = F.interpolate(x, size=(299, 299), mode='bilinear', align_corners=True)
    # 299 x 299 x 3
    x = self.net.Conv2d_1a_3x3(x)
    # 149 x 149 x 32
    x = self.net.Conv2d_2a_3x3(x)
    # 147 x 147 x 32
    x = self.net.Conv2d_2b_3x3(x)
    # 147 x 147 x 64
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 73 x 73 x 64
    x = self.net.Conv2d_3b_1x1(x)
    # 73 x 73 x 80
    x = self.net.Conv2d_4a_3x3(x)
    # 71 x 71 x 192
    x = F.max_pool2d(x, kernel_size=3, stride=2)
    # 35 x 35 x 192
    x = self.net.Mixed_5b(x)
    # 35 x 35 x 256
    x = self.net.Mixed_5c(x)
    # 35 x 35 x 288
    x = self.net.Mixed_5d(x)
    # 35 x 35 x 288
    x = self.net.Mixed_6a(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6b(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6c(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6d(x)
    # 17 x 17 x 768
    x = self.net.Mixed_6e(x)
    # 17 x 17 x 768
    # 17 x 17 x 768
    x = self.net.Mixed_7a(x)
    # 8 x 8 x 1280
    x = self.net.Mixed_7b(x)
    # 8 x 8 x 2048
    x = self.net.Mixed_7c(x)
    # 8 x 8 x 2048
    pool = torch.mean(x.view(x.size(0), x.size(1), -1), 2)
    # 1 x 1 x 2048
    logits = self.net.fc(F.dropout(pool, training=False).view(pool.size(0), -1))
    # 1000 (num_classes)
    return pool, logits


# A pytorch implementation of cov, from Modar M. Alfadly
# https://discuss.pytorch.org/t/covariance-and-gradient-support/16217/2
def torch_cov(m, rowvar=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    m -= torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


# Pytorch implementation of matrix sqrt, from Tsung-Yu Lin, and Subhransu Maji
# https://github.com/msubhransu/matrix-sqrt 
def sqrt_newton_schulz(A, numIters, dtype=None):
  with torch.no_grad():
    if dtype is None:
      dtype = A.type()
    batchSize = A.shape[0]
    dim = A.shape[1]
    normA = A.mul(A).sum(dim=1).sum(dim=1).sqrt()
    Y = A.div(normA.view(batchSize, 1, 1).expand_as(A));
    I = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    Z = torch.eye(dim,dim).view(1, dim, dim).repeat(batchSize,1,1).type(dtype)
    for i in range(numIters):
      T = 0.5*(3.0*I - Z.bmm(Y))
      Y = Y.bmm(T)
      Z = T.bmm(Z)
    sA = Y*torch.sqrt(normA).view(batchSize, 1, 1).expand_as(A)
  return sA


# FID calculator from TTUR--consider replacing this with GPU-accelerated cov
# calculations using torch?
def numpy_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Numpy implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """

  mu1 = np.atleast_1d(mu1)
  mu2 = np.atleast_1d(mu2)

  sigma1 = np.atleast_2d(sigma1)
  sigma2 = np.atleast_2d(sigma2)

  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2

  # Product might be almost singular
  covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
  if not np.isfinite(covmean).all():
    msg = ('fid calculation produces singular product; '
           'adding %s to diagonal of cov estimates') % eps
    print(msg)
    offset = np.eye(sigma1.shape[0]) * eps
    covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

  # Numerical error might give slight imaginary component
  if np.iscomplexobj(covmean):
    print('wat')
    if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
      m = np.max(np.abs(covmean.imag))
      raise ValueError('Imaginary component {}'.format(m))
    covmean = covmean.real  

  tr_covmean = np.trace(covmean) 

  out = diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
  return out


def torch_calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
  """Pytorch implementation of the Frechet Distance.
  Taken from https://github.com/bioinf-jku/TTUR
  The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
  and X_2 ~ N(mu_2, C_2) is
          d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
  Stable version by Dougal J. Sutherland.
  Params:
  -- mu1   : Numpy array containing the activations of a layer of the
             inception net (like returned by the function 'get_predictions')
             for generated samples.
  -- mu2   : The sample mean over activations, precalculated on an 
             representive data set.
  -- sigma1: The covariance matrix over activations for generated samples.
  -- sigma2: The covariance matrix over activations, precalculated on an 
             representive data set.
  Returns:
  --   : The Frechet Distance.
  """


  assert mu1.shape == mu2.shape, \
    'Training and test mean vectors have different lengths'
  assert sigma1.shape == sigma2.shape, \
    'Training and test covariances have different dimensions'

  diff = mu1 - mu2
  # Run 50 itrs of newton-schulz to get the matrix sqrt of sigma1 dot sigma2
  covmean = sqrt_newton_schulz(sigma1.mm(sigma2).unsqueeze(0), 50).squeeze()  
  out = (diff.dot(diff) +  torch.trace(sigma1) + torch.trace(sigma2)
         - 2 * torch.trace(covmean))
  return out


# Calculate Inception Score mean + std given softmax'd logits and number of splits
def calculate_inception_score(pred, num_splits=10):
  scores = []
  for index in range(num_splits):
    pred_chunk = pred[index * (pred.shape[0] // num_splits): (index + 1) * (pred.shape[0] // num_splits), :]
    kl_inception = pred_chunk * (np.log(pred_chunk) - np.log(np.expand_dims(np.mean(pred_chunk, 0), 0)))
    kl_inception = np.mean(np.sum(kl_inception, 1))
    scores.append(np.exp(kl_inception))
  return np.mean(scores), np.std(scores)






# Load and wrap the Inception model
def load_inception_net(parallel=False):
  inception_model = inception_v3(pretrained=True, transform_input=False)
  inception_model = WrapInception(inception_model.eval()).cuda()
  if parallel:
    inception_model = nn.DataParallel(inception_model)
  inception_model.eval()
  return inception_model


@GAN_METRIC_REGISTRY.register()
class PyTorchFIDISScore(object):

  def __init__(self, cfg):
    """

    """
    self.torch_fid_stat                   = cfg.GAN_metric.torch_fid_stat
    self.num_inception_images             = getattr(cfg.GAN_metric, 'num_inception_images', 50000)
    self.IS_splits                        = getattr(cfg.GAN_metric, 'IS_splits', 10)
    self.torch_inception_net_ddp          = getattr(cfg.GAN_metric, 'torch_inception_net_ddp', False)
    self.calculate_FID_use_torch          = getattr(cfg.GAN_metric, 'calculate_FID_use_torch', False)
    self.no_FID                           = getattr(cfg.GAN_metric, 'no_FID', False)

    self.logger = logging.getLogger('tl')
    self.data_mu = np.load(self.torch_fid_stat)['mu']
    self.data_sigma = np.load(self.torch_fid_stat)['sigma']

    # Load inception_v3 network
    self.parallel = self.torch_inception_net_ddp
    if self.parallel and comm.get_world_size() <= 1:
      self.parallel = False
    self.inception_net = self.load_inception_net(parallel=self.parallel)

    if self.parallel:
      ws = comm.get_world_size()
      self.num_inception_images = self.num_inception_images // ws
    pass

  @staticmethod
  def load_inception_net(parallel, device='cuda'):
    net = load_inception_net(parallel=False)
    net = net.to(device)

    if parallel:
      if not comm.get_world_size() > 1:
        return net
      pg = torch.distributed.new_group(range(torch.distributed.get_world_size()))
      net = DistributedDataParallel(
        net, device_ids=[dist.get_rank()], broadcast_buffers=False,
        process_group=pg, check_reduction=False
      )
    return net

  def accumulate_inception_activations(self,
                                       sample_func, net, num_inception_images,
                                       show_process=True, stdout=sys.stdout):

    pool, logits = [], []
    count = 0
    net.eval()
    while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
      if show_process:
        print('\r',
              end='PyTorch FID IS Score forwarding: [%d/%d]' % (count, num_inception_images),
              file=stdout, flush=True)
      with torch.no_grad():
        images = sample_func()
        pool_val, logits_val = net(images.float())
        pool += [pool_val]
        logits += [F.softmax(logits_val, 1)]
        count += images.size(0)
    if show_process:
      print('', file=stdout)
    return torch.cat(pool, 0), torch.cat(logits, 0)

  def __call__(self, sample_func, stdout=sys.stdout):
    start_time = time.time()

    pool, logits = self.accumulate_inception_activations(
      sample_func, net=self.inception_net, num_inception_images=self.num_inception_images, stdout=stdout)

    if self.parallel:
      pool, logits = self.gather_pool_logits(pool, logits)

    if comm.is_main_process():
      IS_mean_torch, IS_std_torch, FID_torch = self.calculate_FID_IS(
        logits=logits, pool=pool, num_splits=self.IS_splits,
        no_fid=self.no_FID, use_torch=self.calculate_FID_use_torch)
    else:
      IS_mean_torch = IS_std_torch = FID_torch = 0

    elapsed_time = time.time() - start_time
    time_str = time.strftime('%H:%M:%S', time.gmtime(elapsed_time))
    self.logger.info('Elapsed time: %s' % (time_str))
    del pool, logits
    comm.synchronize()
    return IS_mean_torch, IS_std_torch, FID_torch

  @staticmethod
  def gather_pool_logits(pool, logits):
    pool_list = comm.gather(data=pool)
    logits_list = comm.gather(data=logits)
    if len(pool_list) > 0:
      pool = torch.cat(pool_list, dim=0).to('cuda')
    if len(logits_list) > 0:
      logits = torch.cat(logits_list, dim=0).to('cuda')
    return pool, logits

  def calculate_FID_IS(self, logits, pool, num_splits=10, no_fid=False, use_torch=False,):
    # if prints:
    #   print('Calculating Inception Score...')
    IS_mean, IS_std = calculate_inception_score(
      logits.cpu().numpy(), num_splits)
    if no_fid:
      FID = 9999.0
    else:
      # if prints:
      #   print('Calculating means and covariances...')
      if use_torch:
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
      else:
        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), \
                    np.cov(pool.cpu().numpy(), rowvar=False)
      # if prints:
      #   print('Covariances calculated, getting FID...')
      if use_torch:
        FID = torch_calculate_frechet_distance(
          mu, sigma,
          torch.tensor(self.data_mu).float().cuda(),
          torch.tensor(self.data_sigma).float().cuda())
        FID = float(FID.cpu().numpy())
      else:
        FID = numpy_calculate_frechet_distance(
          mu, sigma, self.data_mu, self.data_sigma)
    # Delete mu, sigma, pool, logits, and labels, just in case
    del mu, sigma, pool, logits

    return IS_mean, IS_std, FID

  @staticmethod
  def sample(G, z, parallel):
    """
    # Sample function for use with inception metrics
    :param z:
    :param parallel:
    :return:
    """
    with torch.no_grad():
      z.sample_()
      G.eval()

      if parallel:
        G_z = nn.parallel.data_parallel(G, (z,))
      else:
        G_z = G(z)

      G.train()
      return G_z


class InceptionMetricsCond(object):
  # This produces a function which takes in an iterator which returns a set number of samples
  # and iterates until it accumulates config['num_inception_images'] images.
  # The iterator can return samples with a different batch size than used in
  # training, using the setting confg['inception_batchsize']
  def __init__(self, saved_inception_moments, parallel=True, no_fid=False):
    """
    # Load metrics; this is intentionally not in a try-except loop so that
    # the script will crash here if it cannot find the Inception moments.
    # By default, remove the "hdf5" from dataset
    :param saved_inception_moments:
    :param parallel:
    :param no_fid:
    :param show_process:
    """

    self.no_fid = no_fid
    saved_inception_moments = os.path.expanduser(saved_inception_moments)
    self.data_mu = np.load(saved_inception_moments)['mu']
    self.data_sigma = np.load(saved_inception_moments)['sigma']
    # Load network
    self.net = load_inception_net(parallel)

  def __call__(self, G, z, y, num_inception_images, num_splits=10,
               prints=True, show_process=False, use_torch=False,
               parallel=False):
    if prints:
      print('Gathering activations...')

    sample_func = functools.partial(self.sample, G=G, z=z, y=y,
                                    parallel=parallel)
    pool, logits, labels = self.accumulate_inception_activations(
      sample_func, net=self.net,
      num_inception_images=num_inception_images,
      show_process=show_process)
    if prints:
      print('Calculating Inception Score...')
    IS_mean, IS_std = calculate_inception_score(
      logits.cpu().numpy(), num_splits)
    if self.no_fid:
      FID = 9999.0
    else:
      if prints:
        print('Calculating means and covariances...')
      if use_torch:
        mu, sigma = torch.mean(pool, 0), torch_cov(pool, rowvar=False)
      else:
        mu, sigma = np.mean(pool.cpu().numpy(), axis=0), \
                    np.cov(pool.cpu().numpy(), rowvar=False)
      if prints:
        print('Covariances calculated, getting FID...')
      if use_torch:
        FID = torch_calculate_frechet_distance(
          mu, sigma,
          torch.tensor(self.data_mu).float().cuda(),
          torch.tensor(self.data_sigma).float().cuda())
        FID = float(FID.cpu().numpy())
      else:
        FID = numpy_calculate_frechet_distance(
          mu, sigma, self.data_mu, self.data_sigma)
    # Delete mu, sigma, pool, logits, and labels, just in case
    del mu, sigma, pool, logits, labels
    return IS_mean, IS_std, FID

  @staticmethod
  def sample(G, z, y, parallel):
    with torch.no_grad():
      if isinstance(G, functools.partial):
        G.func.eval()
      else:
        G.eval()
      z.sample_()
      y.sample_()
      if parallel:
        G_z = nn.parallel.data_parallel(G, (z, G.shared(y)))
      else:
        G_z = G(z, G.shared(y))
      if isinstance(G, functools.partial):
        G.func.train()
      else:
        G.train()
      return G_z, y

  @staticmethod
  def accumulate_inception_activations(sample, net, num_inception_images=50000, show_process=False):
    """
    # Loop and run the sampler and the net until it accumulates num_inception_images
    # activations. Return the pool, the logits, and the labels (if one wants
    # Inception Accuracy the labels of the generated class will be needed)
    :param sample:
    :param net:
    :param num_inception_images:
    :param show_process:
    :return:
    """
    pool, logits, labels = [], [], []
    count = 0
    while (torch.cat(logits, 0).shape[0] if len(logits) else 0) < num_inception_images:
      if show_process:
        print('accumulate_inception_activations: [%d/%d]' % (count, num_inception_images))
      with torch.no_grad():
        images, labels_val = sample()
        pool_val, logits_val = net(images.float())
        pool += [pool_val]
        logits += [F.softmax(logits_val, 1)]
        labels += [labels_val]
        count += labels_val.size(0)
    return torch.cat(pool, 0), torch.cat(logits, 0), torch.cat(labels, 0)