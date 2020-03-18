import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable

from libs.constant import *
from libs.model import *


# device = torch.device('cuda')  # Default CUDA device
# Tensor_gpu = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
# Tensor = torch.FloatTensor


def dataLoader():
    # Converting the images for PILImage to tensor, so they can be accepted as the input to the network
    print("Loading Dataset")
    transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation=2), transforms.ToTensor()])

    expert_c_folder = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/', transform=transform)

    input_folder = torchvision.datasets.ImageFolder(root='./images_LR/input/', transform=transform)

    testset_gt = expert_c_folder[0]
    trainset_1_gt = expert_c_folder[1]
    trainset_2_gt = expert_c_folder[2]

    testset_inp = input_folder[0]
    trainset_1_inp = input_folder[1]
    trainset_2_inp = input_folder[2]

    trainLoader1 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_1_gt,
            trainset_1_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    trainLoader2 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_gt,
            trainset_2_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    trainLoader_cross = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_inp,
            trainset_1_gt
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    testLoader = torch.utils.data.DataLoader(
        ConcatDataset(
            testset_gt,
            testset_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )
    print("Finished loading dataset")

    return trainLoader1, trainLoader2, trainLoader_cross, testLoader


# Gradient Penalty
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


# Generator Loss
def generatorAdversarialLoss(output_images, discriminator):
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    return gen_adv_loss


# Discriminator Loss
def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA * gradPenalty)


def computeGeneratorLoss(inputs, outputs_g1, discriminator, criterion):
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1, discriminator)
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA * i_loss

    return gen_loss
