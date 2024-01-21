import torch
from codebase import utils as ut
from codebase.models import nns
from torch import nn
from torch.nn import functional as F

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        # Small note: unfortunate name clash with torch.nn
        # nn here refers to the specific architecture file found in
        # codebase/models/nns/*.py
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim)
        self.dec = nn.Decoder(self.z_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # 待办：在这里修改/完善代码
        # 计算负的证据下界（ELBO）及其 KL 分解和重构（Rec）分解
        #
        # 注意 nelbo = kl + rec
        #
        # 输出结果应该都是标量
        ################################################################################
       # 对输入数据 x 进行编码，得到编码的均值 m 和方差 v
        m, v = self.enc.encode(x)
        # 使用重参数化技巧从编码中采样得到潜在变量 z
        z = ut.sample_gaussian(m, v)
        # 使用解码器将潜在变量 z 解码为生成的数据 z
        z = self.dec.decode(z)
        # 计算重构误差，即生成数据与输入数据的负对数伯努利交叉熵的平均值
        rec = torch.mean(-ut.log_bernoulli_with_logits(x, z))
        # 计算 KL 散度，衡量编码分布与先验分布之间的差异
        kl = torch.mean(ut.kl_normal(m, v, self.z_prior_m, self.z_prior_v))
        # 计算负 ELBO（Negative Evidence Lower Bound），即 KL 散度和重构误差之和
        nelbo = kl + rec

        ################################################################################
        # 代码修改结束
        ################################################################################

        return nelbo, kl, rec

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # 待办：在这里修改/完善代码
        # 使用 iw 重要性样本计算 niwae(negative IWAE)以及证据下界（ELBO）的 KL 分解和重构（Rec）分解
        #
        # 输出结果应该都是标量
        ################################################################################

        ################################################################################
        # 代码修改结束
        ################################################################################

        return niwae, kl, rec

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
