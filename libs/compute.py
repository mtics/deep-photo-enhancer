import numpy as np
import torchvision
import torchvision.transforms as transforms
from torch import autograd
from torch.autograd import Variable

from libs.constant import *
from libs.my_model import *


def data_loader():
    # Converting the images for PILImage to tensor, so they can be accepted as the input to the network
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
def compute_gradient_penalty(D, real_sample, fake_sample):
    alpha = Tensor_gpu(np.random.random(real_sample.shape))
    interpolates = (alpha * real_sample + ((1 - alpha) * fake_sample)).requires_grad_(True)
    d_interpolation = D(interpolates)
    fake_output = Variable(Tensor_gpu(real_sample.shape[0], 1, 1, 1).fill_(1.0), requires_grad=False)

    gradients = autograd.grad(
        outputs=d_interpolation,
        inputs=interpolates,
        grad_outputs=fake_output,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]

    # Use Adaptive weighting scheme
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
