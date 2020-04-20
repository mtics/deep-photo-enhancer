import torch
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import skimage.color
from torch import autograd
from torch.autograd import Variable
from torchvision.utils import save_image
from scipy.misc import toimage

device = torch.device('cuda')  # Default CUDA device
Tensor_gpu = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
Tensor = torch.FloatTensor

### PARAMETERS
SIZE = 512
BETA1 = 0.5
BETA2 = 0.999
LAMBDA = 10
ALPHA = 1000
BATCH_SIZE = 5
NUM_EPOCHS_PRETRAIN = 50
NUM_EPOCHS_TRAIN = 100
LATENT_DIM = 100
LEARNING_RATE = 0.00001


### GENERATOR NETWORK
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        #  Convolutional layers

        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv1_bn = nn.BatchNorm2d(16)

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv2_bn = nn.BatchNorm2d(32)

        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3_bn = nn.BatchNorm2d(64)

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4_bn = nn.BatchNorm2d(128)

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.conv5_bn = nn.BatchNorm2d(128)

        # convs for global features
        # input 32x32x128 output 16x16x128
        self.conv51 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 16x16x128 output 8x8x128
        self.conv52 = nn.Conv2d(128, 128, 5, stride=2, padding=2)

        # input 8x8x128 output 1x1x128
        self.conv531 = nn.Conv2d(128, 128, 5, stride=2, padding=1)

        # input 1x1x128 output 1x1x128
        self.conv532 = nn.Conv2d(128, 128, 5, stride=2, padding=1)

        # input 32x32x128 output 32x32x128
        # the global features should be concatenated to the feature map aftere this layer
        # the output after concat would be 32x32x256
        self.conv6 = nn.Conv2d(128, 128, 5, stride=1, padding=2)

        # input 32x32x256 output 32x32x128
        self.conv7 = nn.Conv2d(256, 128, 5, stride=1, padding=2)

        # deconvolutional layers
        # input 32x32x128 output 64x64x128
        self.dconv1 = nn.ConvTranspose2d(128, 128, 4, stride=2, padding=1)
        self.dconv1_bn = nn.BatchNorm2d(128)

        # input 64x64x256 ouput 128x128x128
        self.dconv2 = nn.ConvTranspose2d(256, 128, 4, stride=2, padding=1)
        self.dconv2_bn = nn.BatchNorm2d(256)

        # input 128x128x192 output 256x256x64
        self.dconv3 = nn.ConvTranspose2d(192, 64, 4, stride=2, padding=1)
        self.dconv3_bn = nn.BatchNorm2d(192)

        # input 256x256x96 ouput 512x512x32
        self.dconv4 = nn.ConvTranspose2d(96, 32, 4, stride=2, padding=1)
        self.dconv4_bn = nn.BatchNorm2d(96)

        # final convolutional layers
        # input 512x512x48 output 512x512x16
        self.conv8 = nn.Conv2d(48, 16, 5, stride=1, padding=2)
        self.conv8_bn = nn.BatchNorm2d(48)

        # input 512x512x16 output 512x512x3
        self.conv9 = nn.Conv2d(16, 3, 5, stride=1, padding=2)
        self.conv9_bn = nn.BatchNorm2d(16)
        # SELU

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x0 = self.conv1_bn(F.selu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x1 = self.conv2_bn(F.selu(self.conv2(x0)))

        # input 256x256x32 to output 128x128x64
        x2 = self.conv3_bn(F.selu(self.conv3(x1)))

        # input 128x128x64 to output 64x64x128
        x3 = self.conv4_bn(F.selu(self.conv4(x2)))

        # input 64x64x128 to output 32x32x128
        x4 = self.conv5_bn(F.selu(self.conv5(x3)))

        # convolutions for global features
        # input 32x32x128 to output 16x16x128
        x51 = self.conv51(x4)

        # input 16x16x128 to output 8x8x128
        x52 = self.conv52(x51)

        # input 8x8x128 to output 1x1x128
        x53 = self.conv532(F.selu(self.conv531(x52)))
        x53_temp = torch.cat([x53] * 32, dim=2)
        x53_temp = torch.cat([x53_temp] * 32, dim=3)

        # input 32x32x256 to output 32x32x128
        x5 = self.conv6(x4)

        # input 32x32x128 to output 32x32x128
        x5 = self.conv7(torch.cat([x5, x53_temp], dim=1))

        # input 32x32x128 to output 64x64x128
        xd = self.dconv1(self.dconv1_bn(F.selu(x5)))

        # input 64x64x256 to output 128x128x128
        xd = self.dconv2(self.dconv2_bn(F.selu(torch.cat([xd, x3], dim=1))))

        # input 128x128x192 to output 256x256x64
        xd = self.dconv3(self.dconv3_bn(F.selu(torch.cat([xd, x2], dim=1))))

        # input 256x256x64 to output 512x512x32
        xd = self.dconv4(self.dconv4_bn(F.selu(torch.cat([xd, x1], dim=1))))

        # input 512x512x48 to output 512x512x16
        xd = self.conv8(self.conv8_bn(F.selu(torch.cat([xd, x0], dim=1))))

        # input 512x512x16 to output 512x512x3
        xd = self.conv9(self.conv9_bn(F.selu((xd))))

        # Residuals
        xd = xd + x
        return xd


### DISCRIMINATOR NETWORK
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        #  Convolutional layers

        # input 512x512x3  output 512x512x16
        self.conv1 = nn.Conv2d(3, 16, 5, stride=1, padding=2)
        self.conv1_in = nn.InstanceNorm2d(16)

        # input 512x512x16  output 256x256x32
        self.conv2 = nn.Conv2d(16, 32, 5, stride=2, padding=2)
        self.conv2_in = nn.InstanceNorm2d(32)

        # input 265x256x32  output 128x128x64
        self.conv3 = nn.Conv2d(32, 64, 5, stride=2, padding=2)
        self.conv3_in = nn.InstanceNorm2d(64)

        # input 128x128x64  output 64x64x128
        self.conv4 = nn.Conv2d(64, 128, 5, stride=2, padding=2)
        self.conv4_in = nn.InstanceNorm2d(128)

        # input 64x64x128  output 32x32x128
        # the output of this layer we need layers for global features
        self.conv5 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.conv5_in = nn.InstanceNorm2d(128)

        # input 32x32x128  output 16x16x128
        # the output of this layer we need layers for global features
        self.conv6 = nn.Conv2d(128, 128, 5, stride=2, padding=2)
        self.conv6_in = nn.InstanceNorm2d(128)

        # input 16x16x128  output 1x1x1
        # the output of this layer we need layers for global features
        self.conv7 = nn.Conv2d(128, 1, 16)
        self.conv7_in = nn.InstanceNorm2d(1)

    def forward(self, x):
        # input 512x512x3 to output 512x512x16
        x = self.conv1_in(F.leaky_relu(self.conv1(x)))

        # input 512x512x16 to output 256x256x32
        x = self.conv2_in(F.leaky_relu(self.conv2(x)))

        # input 256x256x32 to output 128x128x64
        x = self.conv3_in(F.leaky_relu(self.conv3(x)))

        # input 128x128x64 to output 64x64x128
        x = self.conv4_in(F.leaky_relu(self.conv4(x)))

        # input 64x64x128 to output 32x32x128
        x = self.conv5_in(F.leaky_relu(self.conv5(x)))

        # input 32x32x128 to output 16x16x128
        x = self.conv6_in(F.leaky_relu(self.conv6(x)))

        # input 16x16x128 to output 1x1x1
        x = self.conv7(x)
        x = F.leaky_relu(x)

        return x


# Creating generator and discriminator
generator1 = Generator()
generator1.load_state_dict(torch.load('./gan1_pretrain_49_449.pth'))
generator1.train()

discriminator = Discriminator()

if torch.cuda.is_available():
    generator1.to(device)
    discriminator.to(device)

### Loading Training and Test Set Data

# Converting the images for PILImage to tensor, so they can be accepted as the input to the network
print("Loading Dataset")
transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation=2), transforms.ToTensor()])

trainset_1_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training1/', transform=transform,
                                                 target_transform=None)
trainset_2_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training2/', transform=transform,
                                                 target_transform=None)
testset_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Testing/', transform=transform,
                                              target_transform=None)
trainset_1_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training1/', transform=transform,
                                                  target_transform=None)
trainset_2_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training2/', transform=transform,
                                                  target_transform=None)
testset_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Testing/', transform=transform,
                                               target_transform=None)


### BUILDING DATA LOADERS
class ConcatDataset(torch.utils.data.Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)


trainLoader1 = torch.utils.data.DataLoader(
    ConcatDataset(
        trainset_1_gt,
        trainset_1_inp
    ),
    batch_size=BATCH_SIZE, shuffle=True, )

trainLoader2 = torch.utils.data.DataLoader(
    ConcatDataset(
        trainset_2_gt,
        trainset_2_inp
    ),
    batch_size=BATCH_SIZE, shuffle=True, )

trainLoader_cross = torch.utils.data.DataLoader(
    ConcatDataset(
        trainset_2_inp,
        trainset_1_gt
    ),
    batch_size=BATCH_SIZE, shuffle=True, )

testLoader = torch.utils.data.DataLoader(
    ConcatDataset(
        testset_gt,
        testset_inp
    ),
    batch_size=BATCH_SIZE, shuffle=True, )
print("Finished loading dataset")

### MSE Loss and Optimizer
criterion = nn.MSELoss()

optimizer_g1 = optim.Adam(generator1.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))


### Gradient Penalty
def computeGradientPenalty(D, realSample, fakeSample):
    alpha = Tensor_gpu(np.random.random((realSample.shape)))
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = D(interpolates)
    fakeOutput = Variable(Tensor_gpu(realSample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=dInterpolation,
        inputs=interpolates,
        grad_outputs=fakeOutput,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    ## Use Adadpative weighting scheme
    gradients = gradients.view(gradients.size(0), -1)
    maxVals = []
    normGradients = gradients.norm(2, dim=1) - 1
    for i in range(len(normGradients)):
        if (normGradients[i] > 0):
            maxVals.append(Variable(normGradients[i].type(Tensor)).detach().numpy())
        else:
            maxVals.append(0)

    gradientPenalty = np.mean(maxVals)
    return gradientPenalty


### Generator Loss
def generatorAdversarialLoss(output_images):
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    return gen_adv_loss


def computeGeneratorLoss(inputs, outputs_g1):
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1)
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA * i_loss

    return gen_loss


### Discriminator Loss
def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA * gradPenalty)


### Training Network
dataiter = iter(testLoader)
gt_test, data_test = dataiter.next()
input_test, dummy = data_test
testInput = Variable(input_test.type(Tensor_gpu))
batches_done = 0
generator_loss = []
discriminator_loss = []
for epoch in range(NUM_EPOCHS_TRAIN):
    for i, (data, gt1) in enumerate(trainLoader_cross, 0):
        input, dummy = data
        groundTruth, dummy = gt1
        trainInput = Variable(input.type(Tensor_gpu))
        realImgs = Variable(groundTruth.type(Tensor_gpu))

        ### TRAIN DISCRIMINATOR
        optimizer_d.zero_grad()
        fake_imgs = generator1(trainInput)

        # # Real Images
        realValid = discriminator(realImgs)
        # Fake Images
        fakeValid = discriminator(fake_imgs)

        gradientPenalty = computeGradientPenalty(discriminator, realImgs.data, fake_imgs.data)
        dLoss = discriminatorLoss(realValid, fakeValid, gradientPenalty)
        dLoss.backward()
        optimizer_d.step()

        if batches_done % 50 == 0:
            ### TRAIN GENERATOR
            optimizer_g1.zero_grad()
            gLoss = computeGeneratorLoss(trainInput, fake_imgs)
            gLoss.backward(retain_graph=True)
            optimizer_g1.step()

        print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
        epoch, NUM_EPOCHS_TRAIN, i, len(trainLoader_cross), dLoss.item(), gLoss.item()))
        f = open("./models/log_Train.txt", "a+")
        f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
        epoch, NUM_EPOCHS_TRAIN, i, len(trainLoader_cross), dLoss.item(), gLoss.item()))
        f.close()

        if batches_done % 200 == 0:
            save_image(fake_imgs.data[:25], "./models/train_images/1Way_Train_%d.png" % batches_done, nrow=5,
                       normalize=True)
            torch.save(generator1.state_dict(),
                       './models/train_checkpoint/gan1_train_' + str(epoch) + '_' + str(i) + '.pth')
            torch.save(discriminator.state_dict(),
                       './models/train_checkpoint/discriminator_train_' + str(epoch) + '_' + str(i) + '.pth')
            fake_test_imgs = generator1(testInput)
            save_image(fake_test_imgs.data[:25], "./model/train_test_images/1Way_Train_Test_%d.png" % batches_done,
                       nrow=5, normalize=True)

        batches_done += 1
        print("Done training discriminator on iteration: %d" % (i))

### TEST NETWORK
batches_done = 0
with torch.no_grad():
    psnrAvg = 0.0
    for j, (gt, data) in enumerate(testLoader, 0):
        input, dummy = data
        groundTruth, dummy = gt
        trainInput = Variable(input.type(Tensor_gpu))
        realImgs = Variable(groundTruth.type(Tensor_gpu))
        output = generator1(trainInput)
        loss = criterion(output, realImgs)
        psnr = 10 * torch.log10(1 / loss)
        psnrAvg += psnr
        save_image(output.data[:25], "./models/test_images/test_%d_%d.png" % (batches_done, j), nrow=5, normalize=True)
        save_image(realImgs.data[:25], "./models/gt_images/gt_%d_%d.png" % (batches_done, j), nrow=5, normalize=True)
        save_image(trainInput.data[:25], "./models/input_images/input_%d_%d.png" % (batches_done, j), nrow=5,
                   normalize=True)
        batches_done += 5
        print("Loss loss: %f" % loss)
        print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "a+")
        f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
    f = open("./models/psnr_Score.txt", "a+")
    f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
    print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
