import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable

from libs.constant import *
from libs.model import *


def data_loader():
    """
    Converting the images for PILImage to tensor,
    so they can be accepted as the input to the network
    :return :
    """
    print("Loading Dataset")
    transform = transforms.Compose([transforms.Resize((SIZE, SIZE), interpolation=2), transforms.ToTensor()])

    testset_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Testing/', transform=transform)
    trainset_1_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training1/', transform=transform)
    trainset_2_gt = torchvision.datasets.ImageFolder(root='./images_LR/Expert-C/Training2/', transform=transform)

    testset_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Testing/', transform=transform)
    trainset_1_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training1/', transform=transform)
    trainset_2_inp = torchvision.datasets.ImageFolder(root='./images_LR/input/Training2/', transform=transform)

    train_loader_1 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_1_gt,
            trainset_1_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    train_loader_2 = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_gt,
            trainset_2_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    train_loader_cross = torch.utils.data.DataLoader(
        ConcatDataset(
            trainset_2_inp,
            trainset_1_gt
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        ConcatDataset(
            testset_gt,
            testset_inp
        ),
        batch_size=BATCH_SIZE * GPUS_NUM,  # Enlarge batch_size by a factor of len(device_ids)
        shuffle=True,
    )
    print("Finished loading dataset")

    return train_loader_1, train_loader_2, train_loader_cross, test_loader


# Gradient Penalty
def compute_gradient_penalty(discriminator, real_sample, fake_sample):
    """
    This function used to compute Gradient Penalty
    The equation is Equation(4) in Chp5
    :param discriminator: stands for D_Y
    :param real_sample: stands for Y
    :param fake_sample: stands for Y'
    :return gradient_penalty: instead of the global parameter LAMBDA
    """
    alpha = Tensor_gpu(np.random.random(real_sample.shape))
    interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample)).requires_grad_(True)  # stands for y^
    d_interpolation = discriminator(interpolates)  # stands for D_Y(y^)
    fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolation,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    # Use Adaptive weighting scheme
    # The following codes stand for the Equation(4) in Chp5
    gradients = gradients.view(gradients.size(0), -1)
    max_vals = []
    norm_gradients = gradients.norm(2, dim=1) - 1
    for i in range(len(norm_gradients)):
        if norm_gradients[i] > 0:
            max_vals.append(Variable(norm_gradients[i].type(Tensor)).detach().numpy())
        else:
            max_vals.append(0)

    gradient_penalty = np.mean(max_vals)
    return gradient_penalty


def generatorAdversarialLoss(output_images, discriminator):
    """
    This function is used to compute Generator Adversarial Loss
    :param output_images:
    :param discriminator:
    :return: the value of Generator Adversarial Loss
    """
    validity = discriminator(output_images)
    gen_adv_loss = torch.mean(validity)
    return gen_adv_loss


def discriminatorLoss(d1Real, d1Fake, gradPenalty):
    """
    This function is used to compute Discriminator Loss E[D(x)]
    :param d1Real:
    :param d1Fake:
    :param gradPenalty:
    :return:
    """
    return (torch.mean(d1Fake) - torch.mean(d1Real)) + (LAMBDA * gradPenalty)


def computeGeneratorLoss(inputs, outputs_g1, discriminator, criterion):
    """
    This function is used to compute Generator Loss
    :param inputs:
    :param outputs_g1:
    :param discriminator:
    :param criterion:
    :return:
    """
    gen_adv_loss1 = generatorAdversarialLoss(outputs_g1, discriminator)
    i_loss = criterion(inputs, outputs_g1)
    gen_loss = -gen_adv_loss1 + ALPHA * i_loss

    return gen_loss


def computeIdentityMappingLoss(x, x1, y, y1):
    """
    This function is used to compute the identity mapping loss
    The equation is Equation(5) in Chp6
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    i_loss = criterion(x, y1) + criterion(y, x1)

    return i_loss


def computeCycleConsistencyLoss(x, x2, y, y2):
    """
    This function is used to compute the cycle consistency loss
    The equation is Equation(6) in Chp6
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    c_loss = criterion(x, y2) + criterion(y, x2)

    return c_loss


def computeAdversarialLosses(discriminator, x, x1, y, y1):
    """
    This function is used to compute the adversarial losses
    for the discriminator and the generator
    The equations are Equation(7)(8)(9) in Chp6
    :param discriminator:
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    dx = discriminator(x)
    dx1 = discriminator(x1)
    dy = discriminator(y)
    dy1 = discriminator(y1)

    ad = criterion(dx) - criterion(dx1) + \
         criterion(dy) - criterion(dy1)
    ag = criterion(dx1) + criterion(dy1)

    return ad, ag


def computeGradientPenaltyFor2Way(generator, discriminator, x, x1, y, y1):
    """
    This function is used to compute the gradient penalty for 2-Way GAN
    The equations are Equation(10)(11) in Chp6
    :param generator:
    :param discriminator:
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """
    gradient_penalty = compute_gradient_penalty(discriminator, x.data, y1.data) + \
                       compute_gradient_penalty(discriminator, y.data, x1.data)

    return gradient_penalty


def computeDiscriminatorLossFor2WayGan(ad, penalty):
    return ad - LAMBDA * penalty


def computeGeneratorLossFor2WayGan(ag, i_loss, c_loss):
    return -ag + ALPHA * i_loss + 10 * ALPHA * c_loss
