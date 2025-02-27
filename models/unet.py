import torch
import torch.nn as nn
import torch.nn.init as init

class double_conv(nn.Module):
    ''' Conv => Batch_Norm => ReLU => Conv2d => Batch_Norm => ReLU
    '''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.conv.apply(self.init_weights)

    def forward(self, x):
        x = self.conv(x)
        return x

    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


class down(nn.Module):
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    ''' up path
        conv_transpose => double_conv
    '''
    def __init__(self, in_ch, out_ch, Transpose=False):
        super(up, self).__init__()
        if Transpose:
            self.up = nn.ConvTranspose2d(in_ch, in_ch // 2, 2, stride=2)
        else:

            self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                                    nn.Conv2d(in_ch, in_ch // 2, kernel_size=1, padding=0),
                                    nn.ReLU(inplace=True))
        self.conv = double_conv(in_ch, out_ch)
        self.up.apply(self.init_weights)
    def forward(self, x1, x2):
        '''
            conv output shape = (input_shape - Filter_shape + 2 * padding)/stride + 1
        '''
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = nn.functional.pad(x1, (diffX // 2, diffX - diffX // 2,
                                    diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x
    @staticmethod
    def init_weights(m):
        if type(m) == nn.Conv2d:
            init.xavier_normal(m.weight)
            init.constant(m.bias, 0)


class inconv(nn.Module):
    ''' input conv layer
        let input channels image to 64 channels
        The oly difference between `inconv` and `down` is maxpool layer
    '''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class outconv(nn.Module):
    ''' input conv layer
        let input 3 channels image to 64 channels
        The oly difference between `inconv` and `down` is maxpool layer
    '''
    def __init__(self, in_ch, out_ch, sigmoid=True):
        super(outconv, self).__init__()
        self.cout = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        if sigmoid:
            self.relu = nn.Sigmoid()
        else:
            self.relu = nn.Identity()

    def forward(self, x):
        x = self.cout(x)
        x = self.relu(x)
        return x


class Unet(nn.Module):
    def __init__(self, in_ch, out_ch, hidden=64,
                 increase=[1,2,4,8,16], dropout_rate=0.5,
                 sigmoid=True, varhead=False, var_de=False):
        super(Unet, self).__init__()
        #self.gpu_ids = gpu_ids
        #self.device = torch.device('cuda:{}'.format(0)) if torch.cuda.is_available() else torch.device('cpu')
        self.inc = inconv(in_ch, hidden*increase[0])
        self.down1 = down(hidden*increase[0], hidden*increase[1])
        # print(list(self.down1.parameters()))
        self.down2 = down(hidden*increase[1], hidden*increase[2])
        self.down3 = down(hidden*increase[2], hidden*increase[3])
        self.down4 = down(hidden*increase[3], hidden*increase[4])
        self.up1 = up(hidden*increase[4], hidden*increase[3], False)
        self.up2 = up(hidden*increase[3], hidden*increase[2], False)
        self.up3 = up(hidden*increase[2], hidden*increase[1], False)
        self.up4 = up(hidden*increase[1], hidden*increase[0], False)
        self.outc = outconv(hidden*increase[0], out_ch, sigmoid)

        self.drop_d2 = nn.Dropout2d(dropout_rate) #before down layer
        self.drop_d3 = nn.Dropout2d(dropout_rate)
        self.drop_d4 = nn.Dropout2d(dropout_rate)
        self.drop_u1 = nn.Dropout2d(dropout_rate) #before up layer
        self.drop_u2 = nn.Dropout2d(dropout_rate)
        self.drop_u3 = nn.Dropout2d(dropout_rate)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x2 = self.drop_d2(x2)
        
        x3 = self.down2(x2)
        x3 = self.drop_d3(x3)
        
        x4 = self.down3(x3)
        x4 = self.drop_d4(x4)
        
        x5 = self.down4(x4)
        x5 = self.drop_u1(x5)

        x = self.up1(x5, x4)
        x = self.drop_u2(x)
        
        x = self.up2(x, x3)
        x = self.drop_u3(x)
        
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        
        return out