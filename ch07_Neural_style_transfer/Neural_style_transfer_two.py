# encoding:utf-8
import torchvision as tv
from torchvision import models, datasets
from torch.utils import data

import torchnet

import torch
from torch import nn
from torch.nn import functional
from torch.autograd import Variable

import numpy as np
from collections import namedtuple

import time
import tqdm
import argparse
import os

"""
class Visualizer():
    '''
    封装了visdom的基本操作，但是你仍然可以通过`self.vis.function`
    调用原生的visdom接口
    '''

    def __init__(self, env='default', **kwargs):
        import visdom
        self.vis = visdom.Visdom(env=env, **kwargs)

        # 画的第几个数，相当于横座标
        # 保存（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        '''
        修改visdom的配置
        '''
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        '''
        一次plot多个
        @params d: dict (name,value) i.e. ('loss',0.11)
        '''
        for k, v in d.items():
            self.plot(k, v)

    def img_many(self, d):
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y):
        '''
        self.plot('loss',1.00)
        '''
        x = self.index.get(name, 0)
        self.vis.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update=None if x == 0 else 'append'
                      )
        self.index[name] = x + 1

    def img(self, name, img_):
        '''
        self.img('input_img',t.Tensor(64,64))
        '''

        if len(img_.size()) < 3:
            img_ = img_.cpu().unsqueeze(0)
        self.vis.image(img_.cpu(),
                       win=name,
                       opts=dict(title=name)
                       )

    def img_grid_many(self, d):
        for k, v in d.items():
            self.img_grid(k, v)

    def img_grid(self, name, input_3d):
        '''
        一个batch的图片转成一个网格图，i.e. input（36，64，64）
        会变成 6*6 的网格图，每个格子大小64*64
        '''
        self.img(name, tv.utils.make_grid(
            input_3d.cpu()[0].unsqueeze(1).clamp(max=1, min=0)))

    def log(self, info, win='log_text'):
        '''
        self.log({'loss':1,'lr':0.0001})
        '''
        self.log_text += ('[{time}] {info} <br>'.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win='log_text')

    def __getattr__(self, name):
        return getattr(self.vis, name)
"""


#  设置网络
class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        features = list(models.vgg16(pretrained=True).features)[:23]
        # features的第3，8，15，22层分别是: relu1_2,relu2_2,relu3_3,relu4_3
        self.features = nn.ModuleList(features).eval()

    def forward(self, x):
        results = []
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii in {3, 8, 15, 22}:
                results.append(x)

        vgg_outputs = namedtuple("VggOutputs", ['relu1_2', 'relu2_2', 'relu3_3', 'relu4_3'])
        return vgg_outputs(*results)

class ConvLayer(nn.Module):
    '''
    add ReflectionPad for Conv
    默认的卷积的padding操作是补0，这里使用边界反射填充
    '''
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out

class UpsampleConvLayer(nn.Module):
    """UpsampleConvLayer
    默认的卷积的padding操作是补0，这里使用边界反射填充
    先上采样，然后做一个卷积(Conv2d)，而不是采用ConvTranspose2d，这种效果更好，参见
    ref: http://distill.pub/2016/deconv-checkerboard/
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample)
            #self.upsample_layer = nn.functional.interpolate(input=,scale_factor=upsample)
        reflection_padding = int(np.floor(kernel_size / 2))
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out

class ResidualBlock(nn.Module):
    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = nn.InstanceNorm2d(channels, affine=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out = out + residual
        return out

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # 下卷积层
        self.initial_layers = nn.Sequential(
            ConvLayer(3, 32, kernel_size=9, stride=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 64, kernel_size=3, stride=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            ConvLayer(64, 128, kernel_size=3, stride=2),
            nn.InstanceNorm2d(128, affine=True),
            nn.ReLU(True),
        )

        # Residual layers(残差层)
        self.res_layers = nn.Sequential(
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128),
            ResidualBlock(128)
        )

        # Upsampling Layers(上卷积层)
        self.upsample_layers = nn.Sequential(
            UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(64, affine=True),
            nn.ReLU(True),
            UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(True),
            ConvLayer(32, 3, kernel_size=9, stride=1)
        )

    def forward(self, x):
        x = self.initial_layers(x)
        x = self.res_layers(x)
        x = self.upsample_layers(x)
        return x


# 4 预处理函数
def normalize_batch(batch, IMAGENET_MEAN = [0.485, 0.456, 0.406],IMAGENET_STD = [0.229, 0.224,  0.225]):
    '''
    输入: b,ch,h,w  0~255
    输出: b,ch,h,w  -2~2
    '''
    mean = batch.data.new(IMAGENET_MEAN).view(1,-1,1,1)
    std = batch.data.new(IMAGENET_STD).view(1,-1,1,1)
    mean = torch.autograd.Variable(mean.expand_as(batch.data))
    std = torch.autograd.Variable(std.expand_as(batch.data))
    return (batch/255.0 - mean) / std

## 获取风格图片的数据
def get_style_data(path, IMAGENET_MEAN = [0.485, 0.456, 0.406],IMAGENET_STD = [0.229, 0.224,  0.225]):
    '''
    加载风格图片，
    输入： path， 文件路径
    返回： 形状 1*c*h*w， 分布 -2~2
    '''
    style_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean = IMAGENET_MEAN,std = IMAGENET_STD),
    ])

    style_image = tv.datasets.folder.default_loader(path)
    style_tensor = style_transform(style_image)
    return style_tensor.unsqueeze(0)

# 风格图片的gram矩阵
def gram_matrix(y):
    '''
    输入 b,c,h,w
    输出 b,c,c
    '''
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram


def train(opt):
    #vis = Visualizer(opt.env)

    # 数据加载
    transfroms = tv.transforms.Compose([
        tv.transforms.Resize(size=opt.image_size),
        tv.transforms.CenterCrop(opt.image_size),
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x * 255)
    ])
    dataset = tv.datasets.ImageFolder(opt.data_root, transfroms)
    dataloader = data.DataLoader(dataset, opt.batch_size)

    # 转换网络
    transformer = TransformerNet()
    if opt.model_path:
        transformer.load_state_dict(torch.load(opt.model_path, map_location=lambda _s, _: _s))

    # 网络 Vgg16
    vgg = Vgg16().eval()

    # 优化器
    optimizer = torch.optim.Adam(transformer.parameters(), opt.lr)

    # 获取风格图片的数据
    style = get_style_data(opt.style_path)
    # vis.img('style', (style[0] * 0.225 + 0.45).clamp(min=0, max=1))

    if opt.use_gpu:
        transformer.cuda()
        style = style.cuda()
        vgg.cuda()

    # 风格图片的gram矩阵
    style_v = Variable(style, volatile=True)
    features_style = vgg(style_v)
    gram_style = [Variable(gram_matrix(y.data)) for y in features_style]

    # 损失统计
    style_meter = torchnet.meter.AverageValueMeter()
    content_meter = torchnet.meter.AverageValueMeter()

    for epoch in range(opt.epoches):
        content_meter.reset()
        style_meter.reset()

        for ii, (x, _) in tqdm.tqdm(enumerate(dataloader)):
            # 训练
            optimizer.zero_grad()
            if opt.use_gpu:
                x = x.cuda()
            x = Variable(x)
            y = transformer(x)
            y = normalize_batch(y)
            x = normalize_batch(x)
            features_y = vgg(y)
            features_x = vgg(x)

            # content loss
            content_loss = opt.content_weight * functional.mse_loss(features_y.relu2_2, features_x.relu2_2)

            # style loss
            style_loss = 0.
            for ft_y, gm_s in zip(features_y, gram_style):
                gram_y = gram_matrix(ft_y)
                style_loss += functional.mse_loss(gram_y, gm_s.expand_as(gram_y))
            style_loss *= opt.style_weight

            total_loss = content_loss + style_loss
            total_loss.backward()
            optimizer.step()

            # 损失平滑
            content_meter.add(content_loss.data[0])
            style_meter.add(style_loss.data[0])

            # if (ii + 1) % opt.plot_every == 0:
            #     if os.path.exists(opt.debug_file):
            #         ipdb.set_trace()
            #
            #     # 可视化
            #     vis.plot('content_loss', content_meter.value()[0])
            #     vis.plot('style_loss', style_meter.value()[0])
            #     # 因为x和y经过标准化处理(utils.normalize_batch)，所以需要将它们还原
            #     vis.img('output', (y.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))
            #     vis.img('input', (x.data.cpu()[0] * 0.225 + 0.45).clamp(min=0, max=1))

        # 保存visdom和模型
        # vis.save([opt.env])
        torch.save(transformer.state_dict(), '/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/checkpoints/%s_style.pth' % epoch)



def stylize(opt):
    # 图片处理
    content_image = tv.datasets.folder.default_loader(opt.content_path)
    content_transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Lambda(lambda x: x.mul(255))
    ])
    content_image = content_transform(content_image)
    content_image = content_image.unsqueeze(0)
    content_image = Variable(content_image, volatile=True)

    # 模型
    style_model = TransformerNet().eval()
    style_model.load_state_dict(torch.load(opt.model_path, map_location=lambda _s, _: _s))

    if opt.use_gpu:
        content_image = content_image.cuda()
        style_model.cuda()

    # 风格迁移与保存
    output = style_model(content_image)
    output_data = output.cpu().data[0]
    tv.utils.save_image(((output_data / 255)).clamp(min=0, max=1), opt.result_path)



if __name__=='__main__':
    # 配置参数
    class Config(object):
        image_size = 256  # 图片大小
        batch_size = 8
        data_root = 'data/'  # 数据集存放路径：data/coco/a.jpg
        num_workers = 4  # 多线程加载数据
        use_gpu = True  # 使用GPU

        style_path = '/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/png/style.jpg'  # 风格图片存放路径
        lr = 1e-3  # 学习率

        env = 'neural-style'  # visdom env
        plot_every = 10  # 每10个batch可视化一次

        epoches = 2  # 训练epoch

        content_weight = 1e5  # content_loss 的权重
        style_weight = 1e10  # style_loss的权重

        model_path = None  # 预训练模型的路径
        debug_file = '/tmp/debugnn'  # touch $debug_fie 进入调试模式

        content_path = 'input.png'  # 需要进行分割迁移的图片
        result_path = '/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/result/output.png'  # 风格迁移结果的保存路径

    parser = argparse.ArgumentParser(description='Neural style transfer with pytorch')
    parser.add_argument('train', type=str)
    # parser.add_argument('--use_gpu', type=int)
    parser.add_argument('--data_root', type=str)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--content_path', type=str)
    parser.add_argument('--result_path', type=str)
    arg = parser.parse_args()

    opt = Config()
    if arg.train=='train':
        for k, v in [('data_root', arg.data_root), ('batch_size', arg.batch_size)]:
            setattr(opt, k, v)  # 对opt类中k属性值改为v
        train(opt=opt)
    if arg.train=='style':
        for k, v in [('model_path', arg.model_path), ('content_path', arg.content_path), ('result_path',arg.result_path)]:
            setattr(opt, k, v)

        stylize(opt=opt)


    """
    # 训练
    python /home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/Neural_style_transfer_two.py train --data_root=/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/data --batch_size=2
    
    # 生成图片
    python /home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/Neural_style_transfer_two.py style  --model_path=/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/checkpoints/1_style.pth --content_path=/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/png/content.png --result_path=/home/zt/Documents/PytorchLearning/ch07_Neural_style_transfer/result/output2.png
    
    """