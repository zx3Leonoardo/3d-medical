from torch.nn import init
from torch import nn

def weight_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    if init_type=='normal':
        net.apply(weight_init_normal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

def init_model(net):
    if isinstance(net, nn.Conv3d) or isinstance(net, nn.ConvTranspose3d):
        nn.init.kaiming_normal_(net.weight.data, 0.25)
        nn.init.constant_(net.bias.data, 0)