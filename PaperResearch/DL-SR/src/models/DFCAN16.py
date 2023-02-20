from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Input, add, multiply, Lambda
from keras.layers.convolutional import Conv2D
from .common import fft2d, fftshift2d, gelu, pixel_shiffle, global_average_pooling2d


def FCALayer(input, channel, reduction=16, size_psc=128):
    absfft1 = Lambda(fft2d, arguments={'gamma': 0.8})(input)
    absfft1 = Lambda(fftshift2d, arguments={'size_psc': size_psc})(absfft1)
    absfft2 = Conv2D(channel, kernel_size=3, activation='relu', padding='same')(absfft1)
    W = Lambda(global_average_pooling2d)(absfft2)
    W = Conv2D(channel // reduction, kernel_size=1, activation='relu', padding='same')(W)
    W = Conv2D(channel, kernel_size=1, activation='sigmoid', padding='same')(W)
    mul = multiply([input, W])
    return mul


import torch
class FCALayer():
    def __init__(self, channels_in, channels_out, reductions=16, sizes=(128, 128), gammas=0.8):
        self.channels_in = channels_in
        self.channels_out = channels_out
        self.reductions = reductions
        self.sizes = sizes
        self.gammas = gammas
        self.conv1 = torch.nn.Conv2d(self.channels_in, self.channels_out, kernel_size=(3, 3), padding=1)
        self.conv2 = torch.nn.Conv2d(self.channels_out, self.channels_out // self.reductions, kernel_size=(1, 1))
        self.conv3 = torch.nn.Conv2d(self.channels_out // self.reductions, self.channels_out, kernel_size=(1, 1))
        self.relu = torch.nn.ReLU(inplace=True)
        self.sigmoid = torch.nn.Sigmoid()

    def forword(self, inputs):
        x = torch.fft.fft2(inputs)
        x = torch.pow(x.abs() + 1e-8, self.gammas)
        x = torch.fft.fftshift(x)
        x = torch.nn.UpsamplingNearest2d(self.sizes)(x)
        x = self.conv1(x)
        w = torch.mean(x, (-2, -1), keepdim=True)
        w = self.conv2(w)
        w = self.relu(w)
        w = self.conv3(w)
        w = self.sigmoid(w)
        outputs = torch.multiply(inputs, w)
        return outputs


def FCAB(input, channel, size_psc=128):
    conv = Conv2D(channel, kernel_size=3, padding='same')(input)
    conv = Lambda(gelu)(conv)
    conv = Conv2D(channel, kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    att = FCALayer(conv, channel, reduction=16, size_psc=size_psc)
    output = add([att, input])
    return output


def ResidualGroup(input, channel, size_psc=128):
    conv = input
    n_RCAB = 4
    for _ in range(n_RCAB):
        conv = FCAB(conv, channel=channel, size_psc=size_psc)
    conv = add([conv, input])
    return conv


def DFCAN(input_shape, scale=2, size_psc=128):
    inputs = Input(input_shape)
    conv = Conv2D(64, kernel_size=3, padding='same')(inputs)
    conv = Lambda(gelu)(conv)
    n_ResGroup = 4
    for _ in range(n_ResGroup):
        conv = ResidualGroup(conv, 64, size_psc)
    conv = Conv2D(64 * (scale ** 2), kernel_size=3, padding='same')(conv)
    conv = Lambda(gelu)(conv)
    upsampled = Lambda(pixel_shiffle, arguments={'scale': scale})(conv)
    conv = Conv2D(1, kernel_size=3, padding='same')(upsampled)
    output = Activation('sigmoid')(conv)
    model = Model(inputs=inputs, outputs=output)
    return model
