import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *
# from libs.old_model import *
import itertools
import os
from libs.data import *

clip_value = 1e8

if __name__ == "__main__":

    start_time = datetime.now()

    # delete old logs and create new logs
    # if os.path.exists('./models/log/log_Train.txt'):
    #     os.remove('./models/log/log_Train.txt')
    #     os.mknod('./models/log/log_Train.txt')

    # Creating generator and discriminator
    generator_xy = Generator()
    generator_xy.load_state_dict(torch.load('./gan2_train_299_xy.pth'))
    generator_xy = nn.DataParallel(generator_xy)

    generator_yx = Generator()
    generator_yx.load_state_dict(torch.load('./gan2_train_299_yx.pth'))
    generator_yx = nn.DataParallel(generator_yx)

    discriminator_x = Discriminator()
    discriminator_x.load_state_dict(torch.load('./discriminator2_train_299_xy.pth'))
    discriminator_x = nn.DataParallel(discriminator_x)

    discriminator_y = Discriminator()
    discriminator_y.load_state_dict(torch.load('./discriminator2_train_299_yx.pth'))
    discriminator_y = nn.DataParallel(discriminator_y)

    if torch.cuda.is_available():
        generator_xy.cuda(device=device)
        generator_yx.cuda(device=device)
        discriminator_x.cuda(device=device)
        discriminator_y.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g_xy = optim.Adam(generator_xy.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_g_yx = optim.Adam(generator_yx.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    optimizer_dx = optim.Adam(discriminator_x.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_dy = optim.Adam(discriminator_y.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Training Network
    dataiter = iter(testLoader)
    gt_test, data_test = dataiter.next()
    input_test, dummy = data_test
    testInput = Variable(input_test.type(Tensor_gpu))
    batches_done = 0
    generator_xy_loss = []
    generator_yx_loss = []
    discriminator_x_loss = []
    discriminator_y_loss = []
    learning_rate = LEARNING_RATE
    for epoch in range(NUM_EPOCHS_TRAIN):

        # adaptive adjust learning rate
        for param_group in optimizer_g_xy.param_groups:
            param_group['lr'] = adjustLearningRate(learning_rate, epoch_num=epoch, decay_rate=DECAY_RATE)

        for param_group in optimizer_g_yx.param_groups:
            param_group['lr'] = adjustLearningRate(learning_rate, epoch_num=epoch, decay_rate=DECAY_RATE)

        for param_group in optimizer_dx.param_groups:
            param_group['lr'] = adjustLearningRate(learning_rate, epoch_num=epoch, decay_rate=DECAY_RATE)

        for param_group in optimizer_dy.param_groups:
            param_group['lr'] = adjustLearningRate(learning_rate, epoch_num=epoch, decay_rate=DECAY_RATE)

        for i, (data, gt1) in enumerate(trainLoader_cross, 0):
            input, dummy = data
            groundTruth, dummy = gt1

            input = data_augmentation(data, i)[0]
            groundTruth = data_augmentation(gt1, i)[0]

            x = Variable(input.type(Tensor_gpu))  # X
            y = Variable(groundTruth.type(Tensor_gpu))  # Y

            x = x[:, :3, :, :]
            y = y[:, :3, :, :]

            # if batches_done % 50 == 0:
            #     # TRAIN GENERATOR
            #     generator_xy.zero_grad()
            #     generator_yx.zero_grad()
            #
            #     i_loss = computeIdentityMappingLoss(x, x1, y, y1)
            #     c_loss = computeCycleConsistencyLoss(x, x2, y, y2)
            #     g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)
            #
            #     g_loss.backward(retain_graph=True)
            #
            #     optimizer_g_xy.step()
            #     optimizer_g_yx.step()

            generator_xy.train()
            generator_yx.train()

            # TRAIN DISCRIMINATOR
            optimizer_dx.zero_grad()
            optimizer_dy.zero_grad()

            y1 = generator_xy(x)  # Y'
            x1 = generator_yx(y)  # X'

            # Real Images
            dy = discriminator_y(y)  # D_Y
            # Fake Images
            dy1 = discriminator_y(y1)  # D_Y'

            dx = discriminator_x(x)  # D_X
            dx1 = discriminator_x(x1)  # D_X'

            set_requires_grad([discriminator_x,discriminator_y], True)

            ad, ag = computeAdversarialLosses(dx, dx1, dy, dy1)
            # ad.backward(retain_graph=True)
            gradient_penalty = computeGradientPenaltyFor1WayGAN(discriminator_x, x.data, x1.data) + \
                               computeGradientPenaltyFor1WayGAN(discriminator_y, y.data, y1.data)

            d_loss = computeDiscriminatorLossFor2WayGan(ad, gradient_penalty)
            d_loss.backward(retain_graph=True)

            torch.nn.utils.clip_grad_value_(itertools.chain(discriminator_y.parameters(),discriminator_x.parameters()),clip_value)

            optimizer_dx.step()
            optimizer_dy.step()

            # TRAIN GENERATOR
            if batches_done % 50 == 0:
                optimizer_g_xy.zero_grad()
                optimizer_g_yx.zero_grad()

                y1 = generator_xy(x)  # Y'
                x1 = generator_yx(y)  # X'

                x2 = generator_yx(y1)  # X''
                y2 = generator_xy(x1)  # Y''

                i_loss = computeIdentityMappingLoss(x, x1, y, y1)
                c_loss = computeCycleConsistencyLoss(x, x2, y, y2)
                g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)
                g_loss.backward(retain_graph=True)

                optimizer_g_xy.step()
                optimizer_g_yx.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [I loss: %f] [C loss: %f]" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item(), i_loss.item(),
                c_loss.item()))

            f = open("./models/log/log_Train.txt", "a+")
            f.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [I loss: %f] [C loss: %f] [AD: %f] [AG: %f] [GP: %f]\n" % (
                    epoch + 1,
                    NUM_EPOCHS_TRAIN,
                    i + 1,
                    len(trainLoader_cross),
                    d_loss.item(),
                    g_loss.item(),
                    i_loss.item(),
                    c_loss.item(),
                    ad.item(),
                    ag.item(),
                    gradient_penalty.item()
                ))
            f.close()

            batches_done += 1
            print("Done training discriminator on iteration: %d" % i)

        for k in range(0, y1.data.shape[0]):
            save_image(
                y1.data[k],
                "./models/train_images/2Way/2Way_Train_%d_%d.png" % (epoch + 1, k + 1),
                nrow=1,
                normalize=True
            )
        torch.save(generator_xy.state_dict(),
                   './models/train_checkpoint/2Way/xy/gan2_train_' + str(epoch) + '.pth')
        torch.save(generator_yx.state_dict(),
                   './models/train_checkpoint/2Way/yx/gan2_train_' + str(epoch) + '.pth')
        torch.save(discriminator_x.state_dict(),
                   './models/train_checkpoint/2Way/xy/discriminator2_train_' + str(epoch) + '.pth')
        torch.save(discriminator_y.state_dict(),
                   './models/train_checkpoint/2Way/yx/discriminator2_train_' + str(epoch) + '.pth')

        fake_test_imgs = generator_xy(testInput)
        for k in range(0, fake_test_imgs.data.shape[0]):
            save_image(fake_test_imgs.data[k],
                       "./models/train_test_images/2Way/2Way_Train_Test_%d_%d.png" % (
                           epoch, k),
                       nrow=1, normalize=True)

    # TEST NETWORK
    batches_done = 0
    with torch.no_grad():
        psnrAvg = 0.0
        for j, (gt, data) in enumerate(testLoader, 0):
            input, dummy = data
            groundTruth, dummy = gt
            x = Variable(input.type(Tensor_gpu))
            y = Variable(groundTruth.type(Tensor_gpu))
            output = generator_xy(x)
            loss = criterion(output, y)
            psnr = 10 * torch.log10(1 / loss)
            psnrAvg += psnr

            for k in range(0, output.data.shape[0]):
                save_image(output.data[k],
                           "./models/test_images/2Way/test_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, y.data.shape[0]):
                save_image(y.data[k],
                           "./models/gt_images/2Way/gt_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, x.data.shape[0]):
                save_image(x.data[k],
                           "./models/input_images/2Way/input_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1), nrow=1,
                           normalize=True)

            batches_done += 5
            print("Loss loss: %f" % loss)
            print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
            f = open("./models/log/psnr_Score.txt", "w")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/log/psnr_Score.txt", "w")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))

    end_time = datetime.now()
    print(end_time - start_time)
