

model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth',
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'
}

rootdir = '/home/keisoku/work/pretrainedmodel/'
model_files = {
    'alexnet'       : rootdir + 'alexnet-owt-4df8aa71.pth', #broken?
    'mobilenet_v2'  : rootdir + 'mobilenet_v2-b0353104.pth',
    'mobilenet_v3_large'  : rootdir + 'mobilenetv3-large-1cd25616.pth',
    'mobilenet_v3_small'  : rootdir + 'mobilenetv3-small-55df8e1f.pth',
    'resnet18'      : rootdir + 'resnet18-5c106cde.pth',
    'resnet34'      : rootdir + 'resnet34-333f7ec4.pth',
    'resnet50'      : rootdir + 'resnet50-19c8e357.pth',
    'resnet101'     : rootdir + 'resnet101-5d3b4d8f.pth',
    'resnet152'     : rootdir + 'resnet152-b121ed2d.pth',
    'resnext50_32x4d': rootdir + 'resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': rootdir + 'resnext101_32x8d-8ba56ff5.pth',
    'vgg11'         : rootdir + 'vgg11-bbd30ac9.pth',  #broken?
    'vgg13'         : rootdir + 'vgg13-c768596a.pth', #broken?
    'vgg16'         : rootdir + 'vgg16-397923af.pth', 
    'vgg19'         : rootdir + 'vgg19-dcbb9e9d.pth',
    'vgg11_bn'      : rootdir + 'vgg11_bn-6002323d.pth',
    'vgg13_bn'      : rootdir + 'vgg13_bn-abd245e5.pth',
    'vgg16_bn'      : rootdir + 'vgg16_bn-6c64b313.pth',
    'vgg19_bn'      : rootdir + 'vgg19_bn-c79401a0.pth'
    }

def get_model_urls(netname):
    url = model_urls.get(netname)
    assert url != None, 'Netwrok {0} Not Found'.format(netname)
    return url

def get_model_files(netname):
    filename = model_files.get(netname)
    assert filename != None, 'Netwrok {0} Not Found'.format(netname)
    return filename

