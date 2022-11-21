import torch
import torch.nn.functional as F
from torch.autograd import Variable

import networks
import numpy as np
import math
import functools

class AugmentedCycleGAN(object):

    def __init__(self, opt, testing=False):

        self.old_lr = opt.lr
        opt.use_sigmoid = opt.no_lsgan

        self.opt = opt

        self.netG_A_B = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                    output_nc=opt.output_nc, ngf=opt.ngf,
                                                    which_model_netG=opt.which_model_netG,
                                                    norm=opt.norm, use_dropout=opt.use_dropout,
                                                    gpu_ids=opt.gpu_ids)

        self.netG_B_A = networks.define_G(input_nc=opt.output_nc,
                                            output_nc=opt.input_nc, ngf=opt.ngf,
                                            which_model_netG=opt.which_model_netG,
                                            norm=opt.norm, use_dropout=opt.use_dropout,
                                            gpu_ids=opt.gpu_ids)

        enc_input_nc = opt.output_nc
        if opt.enc_A_B:
            enc_input_nc += opt.input_nc

        self.netE_B = networks.define_E(nlatent=opt.nlatent, input_nc=enc_input_nc,
                                        nef=opt.nef, norm='batch', gpu_ids=opt.gpu_ids)

        self.netD_A = networks.define_D_A(input_nc=opt.input_nc,
                                            ndf=32, which_model_netD=opt.which_model_netD,
                                            norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        self.netD_B = networks.defineD_B(input_nc=opt.output_nc,
                                            ndf=opt.ndf, which_model_netD=opt.which_model_netD,
                                            norm=opt.norm, use_sigmoid=opt.use_sigmoid, gpu_ids=opt.gpu_ids)

        self.netD_z_B = networks.define_LAT_D(nlatent=opt.nlatent, ndf=opt.ndf,
                                                use_sigmoid=opt.use_sigmoid,
                                                gpu_ids=opt.gpu_ids)

        self.optimizer_G_A = torch.optim.Adam(self.netG_B_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_G_B = torch.optim.Adam(itertools.chain(self.netG_A_B.parameters(),
                                                                self.netE_B.parameters()),
                                              lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(),
                                                lr=opt.lr/5., betas=(opt.beta1, 0.999))
        self.optimizer_D_B = torch.optim.Adam(itertools.chain(self.netD_B.parameters(),
                                                                self.netD_z_B.parameters(),
                                                                ),
                                                lr=opt.lr/5., betas=(opt.beta1, 0.999))
        self.criterionGAN = functools.partial(criterion_GAN, use_sigmoid=opt.use_sigmoid)
        self.criterionCycle = F.l1_loss

        if not testing:
            with open("%s/nets.txt" % opt.expr_dir, 'w') as nets_f:
                networks.print_network(self.netG_A_B, nets_f)
                networks.print_network(self.netG_B_A, nets_f)
                networks.print_network(self.netD_A, nets_f)
                networks.print_network(self.netD_B, nets_f)
                networks.print_network(self.netD_z_B, nets_f)
                networks.print_network(self.netE_B, nets_f)

    def train_instance(self, real_A, real_B, prior_z_B):

        fake_B = self.netG_A_B.forward(real_A, prior_z_B)

        fake_A = self.netG_B_A.forward(real_B)

        if self.opt.enc_A_B:
            concat_B_A = torch.cat((fake_A, real_B),1)
            mu_z_realB, logvar_z_realB = self.netE_B.forward(concat_B_A)
        else:
            mu_z_realB, logvar_z_realB = self.netE_B.froward(real_B)

        if self.opt.
