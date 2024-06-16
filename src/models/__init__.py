from . import UNet, UNet_ResNet, FCN8
from torch.nn import init


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def get_model(model_name):

    if model_name == "unet":
        model = UNet.UNet(n_channels=3, n_classes=2)
    elif model_name == "fcn8":
        model = FCN8.FCN8(n_classes=2)
    elif model_name == "unet_resnet":
        model = UNet_ResNet.UNet(n_classes=2)
    elif model_name == "unet_cgm":
        model = UNet_ResNet.UNet_CGM(n_classes=2)
    elif model_name == "attention_unet":
        model = UNet_ResNet.Attention_UNet(n_classes=2)
    elif model_name == "attention_unet_cgm":
        model = UNet_ResNet.Attention_UNet_CGM(n_classes=2)
    elif model_name == "unet_deepsup":
        model = UNet_ResNet.UNet_DeepSup(n_classes=2)
    else:
        raise RuntimeError("Model with name %s not found" % model_name)

    return model


def params_init(args):
    with_cgm = False
    with_deepsup = False
    lambda_cgm = 0.

    if args.network == "unet_cgm" or args.network == "attention_unet_cgm":
        with_cgm = True
        lambda_cgm = 0.1
    if args.network == "unet_deepsup":
        with_deepsup = True

    params = {
        "with_cgm": with_cgm,
        "with_deepsup": with_deepsup,
        "lambda_cgm": lambda_cgm
    }

    return params
