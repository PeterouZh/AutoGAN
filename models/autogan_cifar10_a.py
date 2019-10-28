# -*- coding: utf-8 -*-
# @Date    : 2019-07-31
# @Author  : Xinyu Gong (xy_gong@tamu.edu)
# @Link    : None
# @Version : 0.0

from torch import nn
from models.building_blocks import Cell, DisBlock, OptimizedDisBlock


class Generator(nn.Module):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.args = args
        self.ch = args.gf_dim
        self.bottom_width = args.bottom_width
        self.l1 = nn.Linear(args.latent_dim, (self.bottom_width ** 2) * args.gf_dim)
        self.cell1 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=0, short_cut=True)
        self.cell2 = Cell(args.gf_dim, args.gf_dim, 'bilinear', num_skip_in=1, short_cut=True)
        self.cell3 = Cell(args.gf_dim, args.gf_dim, 'nearest', num_skip_in=2, short_cut=False)
        self.to_rgb = nn.Sequential(
            nn.BatchNorm2d(args.gf_dim),
            nn.ReLU(),
            nn.Conv2d(args.gf_dim, 3, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        h = self.l1(z).view(-1, self.ch, self.bottom_width, self.bottom_width)
        h1_skip_out, h1 = self.cell1(h)
        h2_skip_out, h2 = self.cell2(h1, (h1_skip_out, ))
        _, h3 = self.cell3(h2, (h1_skip_out, h2_skip_out))
        output = self.to_rgb(h3)

        return output


class Discriminator(nn.Module):
    def __init__(self, args, activation=nn.ReLU()):
        super(Discriminator, self).__init__()
        self.ch = args.df_dim
        self.activation = activation
        self.block1 = OptimizedDisBlock(args, 3, self.ch)
        self.block2 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=True)
        self.block3 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.block4 = DisBlock(
            args,
            self.ch,
            self.ch,
            activation=activation,
            downsample=False)
        self.l5 = nn.Linear(self.ch, 1, bias=False)
        if args.d_spectral_norm:
            self.l5 = nn.utils.spectral_norm(self.l5)

    def forward(self, x):
        h = x
        layers = [self.block1, self.block2, self.block3]
        model = nn.Sequential(*layers)
        h = model(h)
        h = self.block4(h)
        h = self.activation(h)
        # Global average pooling
        h = h.sum(2).sum(2)
        output = self.l5(h)

        return output


def test_autogan_cifar10_a_Generator(args1, myargs):
    import cfg, os, torch
    import numpy as np
    myargs.config = getattr(myargs.config, 'train_autogan_cifar10_a')
    myargs.args = args1
    args = cfg.parse_args()
    for k, v in myargs.config.items():
        setattr(args, k, v)

    args.tf_inception_model_dir = os.path.expanduser(
        args.tf_inception_model_dir)
    args.fid_stat = os.path.expanduser(args.fid_stat)
    args.data_path = os.path.expanduser(args.data_path)

    gen_net = Generator(args=args).cuda()
    z = torch.cuda.FloatTensor(
        np.random.normal(0, 1, (16, args.latent_dim)))
    x = gen_net(z)

    import torchviz
    g = torchviz.make_dot(x)
    g.view()
    pass


def test_autogan_cifar10_a_Discriminator(args1, myargs):
    import cfg, os, torch
    import numpy as np
    args = getattr(myargs.config, 'test_autogan_cifar10_a_Discriminator')

    dis_net = Discriminator(args=args).cuda()
    x = torch.cuda.FloatTensor(np.random.normal(0, 1, (16, 3, 32, 32)))
    x = dis_net(x)

    import torchviz
    g = torchviz.make_dot(x)
    g.view()
    pass

