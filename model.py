import torch
from torch import nn
from torch.utils import model_zoo


class RGBLNet(nn.Module):
    def __init__(self):
        super(RGBLNet, self).__init__()
        self.vgg = VGG()
        self.load_vgg()
        self.dmp = BackEnd()

        self.conv_out = BaseConv(64, 1, 1, 1, activation=None, use_bn=False)

    def forward(self, input):
        input = self.vgg(input)
        dmp_out = self.dmp(*input)
        dmp_out = self.conv_out(dmp_out)

        return dmp_out

    def load_vgg(self):
        state_dict = model_zoo.load_url('https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
        old_name = [0, 2, 5, 7, 10, 12, 14, 16, 19, 21, 23, 25, 28, 30, 32, 34]
        new_name = ['1_1', '1_2', '2_1', '2_2', '3_1', '3_2', '3_3', '3_4', '4_1', '4_2', '4_3', '4_4', '5_1', '5_2', '5_3', '5_4']
        new_dict = {}
        for j in range(15):
            i = j + 1
            new_dict['conv' + new_name[i] + '.conv.weight'] = \
                state_dict['features.' + str(old_name[i]) + '.weight']
            new_dict['conv' + new_name[i] + '.conv.bias'] = \
                state_dict['features.' + str(old_name[i]) + '.bias']
        self.vgg.load_state_dict(new_dict, False)


class VGG(nn.Module):
    def __init__(self):
        super(VGG, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1_1 = BaseConv(4, 64, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv1_2 = BaseConv(64, 64, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv2_1 = BaseConv(64, 128, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv2_2 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv3_1 = BaseConv(128, 256, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv3_2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv3_3 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv3_4 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv4_1 = BaseConv(256, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv4_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv4_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv4_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv5_1 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv5_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv5_3 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)
        self.conv5_4 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False)

    def forward(self, input):
        input = self.conv1_1(input)
        input = self.conv1_2(input)
        input = self.pool(input)
        input = self.conv2_1(input)
        conv2_2 = self.conv2_2(input)

        input = self.pool(conv2_2)
        input = self.conv3_1(input)
        input = self.conv3_2(input)
        input = self.conv3_3(input)
        conv3_4 = self.conv3_4(input)

        input = self.pool(conv3_4)
        input = self.conv4_1(input)
        input = self.conv4_2(input)
        input = self.conv4_3(input)
        conv4_4 = self.conv4_4(input)

        input = self.pool(conv4_4)
        input = self.conv5_1(input)
        input = self.conv5_2(input)
        input = self.conv5_3(input)
        conv5_4 = self.conv5_4(input)

        return conv2_2, conv3_4, conv4_4, conv5_4


class BackEnd(nn.Module):
    def __init__(self):
        super(BackEnd, self).__init__()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)

        self.conv0_1 = BaseConv(512, 512, 1, 1, activation=nn.ReLU(), use_bn=False, dilation=False)
        self.conv0_2 = BaseConv(512, 512, 3, 1, activation=nn.ReLU(), use_bn=False, dilation=True)

        self.conv1 = BaseConv(1024, 256, 1, 1, activation=nn.ReLU(), use_bn=False, dilation=False)
        self.conv2 = BaseConv(256, 256, 3, 1, activation=nn.ReLU(), use_bn=False, dilation=True)

        self.conv3 = BaseConv(512, 128, 1, 1, activation=nn.ReLU(), use_bn=False, dilation=False)
        self.conv4 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=False, dilation=True)

        self.conv5 = BaseConv(256, 128, 1, 1, activation=nn.ReLU(), use_bn=False, dilation=False)
        self.conv6 = BaseConv(128, 128, 3, 1, activation=nn.ReLU(), use_bn=False, dilation=True)
        self.conv7 = BaseConv(128, 64, 3, 1, activation=nn.ReLU(), use_bn=False, dilation=True)

    def forward(self, *input):
        conv2_2, conv3_4, conv4_4, conv5_4 = input

        input = self.conv0_1(conv5_4)
        input = self.conv0_2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv4_4], 1)
        input = self.conv1(input)
        input = self.conv2(input)
        input = self.upsample(input)

        input = torch.cat([input, conv3_4], 1)
        input = self.conv3(input)
        input = self.conv4(input)
        input = self.upsample(input)

        input = torch.cat([input, conv2_2], 1)
        input = self.conv5(input)
        input = self.conv6(input)
        input = self.conv7(input)

        return input


class BaseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel, stride=1, activation=None, use_bn=False, dilation=False):
        super(BaseConv, self).__init__()
        if dilation:
            p_rate = 2
            d_rate = 2
        else:
            p_rate = kernel // 2
            d_rate = 1
        self.use_bn = use_bn
        self.activation = activation
        self.conv = nn.Conv2d(in_channels, out_channels, kernel, stride, padding=p_rate, dilation=d_rate)
        self.conv.weight.data.normal_(0, 0.01)
        self.conv.bias.data.zero_()
        self.bn = nn.BatchNorm2d(out_channels)
        self.bn.weight.data.fill_(1)
        self.bn.bias.data.zero_()

    def forward(self, input):
        input = self.conv(input)
        if self.use_bn:
            input = self.bn(input)
        if self.activation:
            input = self.activation(input)

        return input


if __name__ == '__main__':
    input = torch.randn(8, 4, 400, 400).cuda()
    model = Model().cuda()
    output = model(input)
    print(input.size())
    print(output.size())
