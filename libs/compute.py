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
# def compute_gradient_penalty(discriminator, real_sample, fake_sample):
#     """
#     This function used to compute Gradient Penalty
#     The equation is Equation(4) in Chp5
#     :param discriminator: stands for D_Y
#     :param real_sample: stands for Y
#     :param fake_sample: stands for Y'
#     :return gradient_penalty: instead of the global parameter LAMBDA
#     """
#     alpha = Tensor_gpu(np.random.random(real_sample.shape))
#     interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample)).requires_grad_(True)  # stands for y^
#     d_interpolation = discriminator(interpolates)  # stands for D_Y(y^)
#     fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)
#
#     gradients = autograd.grad(
#         outputs=d_interpolation,
#         inputs=interpolates,
#         grad_outputs=fake_output,
#         create_graph=True,
#         retain_graph=True,
#         only_inputs=True)[0]
#
#     # Use Adaptive weighting scheme
#     # The following codes stand for the Equation(4) in Chp5
#     gradients = gradients.view(gradients.size(0), -1)
#     max_vals = []
#     norm_gradients = gradients.norm(2, dim=1) - 1
#     for i in range(len(norm_gradients)):
#         if norm_gradients[i] > 0:
#             max_vals.append(Variable(norm_gradients[i].type(Tensor)).detach().numpy())
#         else:
#             max_vals.append(0)
#
#     tensor_max_vals = torch.as_tensor(max_vals, dtype=torch.float64, device=device)
#
#     # gradient_penalty = np.mean(max_vals)
#     gradient_penalty = torch.mean(tensor_max_vals)
#     return gradient_penalty


def computeGradientPenaltyFor1WayGAN(discriminator, realSample, fakeSample):
    alpha = Tensor_gpu(np.random.random((realSample.shape)))
    interpolates = (alpha * realSample + ((1 - alpha) * fakeSample)).requires_grad_(True)
    dInterpolation = discriminator(interpolates)
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
            # temp_data = Variable(norm_gradients[i].type(Tensor)).detach().item()
            temp_data = Variable(norm_gradients[i].type(Tensor)).item()
            max_vals.append(temp_data)
        else:
            max_vals.append(0)

    tensor_max_vals = torch.tensor(max_vals, dtype=torch.float64, device=device, requires_grad=True)

    # gradient_penalty = np.mean(max_vals)
    gradient_penalty = torch.mean(tensor_max_vals)
    # gradient_penalty.backward(retain_graph=True)
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


def computeDiscriminatorLoss(d1Real, d1Fake, gradPenalty):
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
    :param x2:
    :param y:
    :param y2:
    :return:
    """
    # MSE Loss and Optimizer
    criterion = nn.MSELoss()
    c_loss = criterion(x, x2) + criterion(y, y2)

    return c_loss


def computeAdversarialLosses(dx, dx1, dy, dy1):
    """
    This function is used to compute the adversarial losses
    for the discriminator and the generator
    The equations are Equation(7)(8)(9) in Chp6
    :param x:
    :param x1:
    :param y:
    :param y1:
    :return:
    """

    ad = torch.mean(dx) - torch.mean(dx1) + \
         torch.mean(dy) - torch.mean(dy1)
    ag = torch.mean(dx1) + torch.mean(dy1)

    return ad, ag


def computeDiscriminatorLossFor2WayGan(ad, penalty):
    return ad - LAMBDA * penalty


def computeGeneratorLossFor2WayGan(ag, i_loss, c_loss):
    return -ag + ALPHA * i_loss + 10 * ALPHA * c_loss


def adjustLearningRate(learning_rate, decay_rate, epoch_num):
    """
    Adjust Learning rate to get better performance
    :param learning_rate:
    :param decay_rate:
    :param epoch_num:
    :return:
    """
    return learning_rate / (1 + decay_rate * epoch_num)
