import torch.autograd as autograd
from torch.autograd import grad
from torch.autograd import Variable
import torch
import torch.nn.functional as F


def wgan_agp_gradient_penalty(x, y, f):
  # Adaptive gradient penalty
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape, device='cuda')
  z = x + alpha * (y - x)

  # gradient penalty
  z = Variable(z, requires_grad=True).cuda()
  o = f(z)
  g = grad(o, z, grad_outputs=torch.ones(o.size(), device='cuda'), create_graph=True)[0].view(z.size(0), -1)
  with torch.no_grad():
    g_norm_mean = g.norm(p=2, dim=1).mean().item()
  gp = ((g.norm(p=2, dim=1) - g_norm_mean)**2).mean()
  return gp, g_norm_mean


def wgan_gp_gradient_penalty(x, y, f):
  # interpolation
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape, device='cuda')
  z = x + alpha * (y - x)

  # gradient penalty
  z = Variable(z, requires_grad=True).cuda()
  o = f(z)
  g = grad(o, z, grad_outputs=torch.ones(o.size(), device='cuda'), create_graph=True)[0].view(z.size(0), -1)
  gp = ((g.norm(p=2, dim=1) - 1)**2).mean()

  return gp


def wgan_div_gradient_penalty(real_imgs, fake_imgs, real_validity, fake_validity):
  # Compute W-div gradient penalty
  k = 2
  p = 6

  real_grad_out = Variable(torch.cuda.FloatTensor(real_validity.size()).fill_(1.0), requires_grad=False)
  real_grad = autograd.grad(
    real_validity, real_imgs, real_grad_out, create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  real_grad_norm = real_grad.view(real_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

  fake_grad_out = Variable(torch.cuda.FloatTensor(fake_validity.size()).fill_(1.0), requires_grad=False)
  fake_grad = autograd.grad(
    fake_validity, fake_imgs, fake_grad_out, create_graph=True, retain_graph=True, only_inputs=True,
  )[0]
  fake_grad_norm = fake_grad.view(fake_grad.size(0), -1).pow(2).sum(1) ** (p / 2)

  div_gp = torch.mean(real_grad_norm + fake_grad_norm) * k / 2
  return div_gp


def compute_grad2(d_out, x_in):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  grad_dout2 = grad_dout.pow(2)
  assert (grad_dout2.size() == x_in.size())
  reg = grad_dout2.view(batch_size, -1).sum(1)
  reg_mean = reg.mean()
  return reg, reg_mean


def agp_real(d_out, x_in):
  batch_size = x_in.size(0)
  grad_dout = autograd.grad(
    outputs=d_out.sum(), inputs=x_in,
    create_graph=True, retain_graph=True, only_inputs=True
  )[0]
  grad_dout = grad_dout.view(batch_size, -1)
  with torch.no_grad():
    g_norm_mean = grad_dout.norm(p=2, dim=1).mean().item()
  gp = ((grad_dout.norm(p=2, dim=1) - g_norm_mean) ** 2).mean()

  # grad_dout2 = grad_dout.pow(2)
  # assert (grad_dout2.size() == x_in.size())
  # reg = grad_dout2.view(batch_size, -1).sum(1)
  # reg_mean = reg.mean()
  return gp, g_norm_mean


def wgan_gp_gradient_penalty_cond(x, G_z, gy, f):
  """
  gradient penalty for conditional discriminator
  :param x:
  :param G_z:
  :param gy: label for x * alpha + (1 - alpha) * G_z
  :param f:
  :return:
  """
  # interpolation
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape).cuda()
  z = x + alpha * (G_z - x)

  # gradient penalty
  z.requires_grad_()
  o = torch.nn.parallel.data_parallel(f, (z, gy))
  g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
  gp = ((g.norm(p=2, dim=1) - 1) ** 2).mean()
  return gp


def wgan_agp_gradient_penalty_cond(x, G_z, gy, f):
  """
  gradient penalty for conditional discriminator
  :param x:
  :param G_z:
  :param gy: label for x * alpha + (1 - alpha) * G_z
  :param f:
  :return:
  """
  # interpolation
  shape = [x.size(0)] + [1] * (x.dim() - 1)
  alpha = torch.rand(shape).cuda()
  z = x + alpha * (G_z - x)

  # gradient penalty
  z.requires_grad_()
  o = torch.nn.parallel.data_parallel(f, (z, gy))
  g = torch.autograd.grad(o, z, grad_outputs=torch.ones(o.size()).cuda(), create_graph=True)[0].view(z.size(0), -1)
  with torch.no_grad():
    g_norm_mean = g.norm(p=2, dim=1).mean().item()
  gp = ((g.norm(p=2, dim=1) - g_norm_mean) ** 2).mean()
  return gp, g_norm_mean


def wgan_discriminator_loss(r_logit, f_logit):
  """
  d_loss = -wd + gp * 10.0
  :param r_logit:
  :param f_logit:
  :return:
  """
  r_logit_mean = r_logit.mean()
  f_logit_mean = f_logit.mean()

  # Wasserstein-1 Distance
  wd = r_logit_mean - f_logit_mean
  D_loss = -wd
  return r_logit_mean, f_logit_mean, wd, D_loss


def wgan_generator_loss(f_logit):
  G_fake_mean = f_logit.mean()
  G_loss = - G_fake_mean
  return G_fake_mean, G_loss


def hinge_loss_discriminator(r_logit, f_logit):
  r_logit_mean = r_logit.mean()
  f_logit_mean = f_logit.mean()

  loss_real = torch.mean(F.relu(1. - r_logit))
  loss_fake = torch.mean(F.relu(1. + f_logit))
  D_loss = loss_real + loss_fake
  return r_logit_mean, f_logit_mean, D_loss


def hinge_loss_generator(f_logit):
  f_logit_mean = f_logit.mean()
  G_loss = - f_logit_mean
  return f_logit_mean, G_loss