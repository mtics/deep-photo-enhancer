import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
# from libs.model import *
from libs.old_model import *


if __name__ == "__main__":

    start_time = datetime.now()

    # Creating generator and discriminator
    generator_xy = Generator()
    generator_xy = nn.DataParallel(generator_xy)
    generator_xy.load_state_dict(torch.load('./gan2_pretrain_50_12_xy.pth'))

    generator_yx = Generator()
    generator_yx = nn.DataParallel(generator_yx)
    generator_yx.load_state_dict(torch.load('./gan2_pretrain_50_12_yx.pth'))

    generator_xy.train()
    generator_yx.train()

    discriminator_xy = Discriminator()
    discriminator_xy = nn.DataParallel(discriminator_xy)

    discriminator_yx = Discriminator()
    discriminator_yx = nn.DataParallel(discriminator_yx)

    if torch.cuda.is_available():
        generator_xy.cuda(device=device)
        generator_yx.cuda(device=device)
        discriminator_xy.cuda(device=device)
        discriminator_yx.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g_xy = optim.Adam(generator_xy.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_g_yx = optim.Adam(generator_yx.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    optimizer_d_xy = optim.Adam(discriminator_xy.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d_yx = optim.Adam(discriminator_yx.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Training Network
    dataiter = iter(testLoader)
    gt_test, data_test = dataiter.next()
    input_test, dummy = data_test
    testInput = Variable(input_test.type(Tensor_gpu))
    batches_done = 0
    generator_xy_loss = []
    generator_yx_loss = []
    discriminator_xy_loss = []
    discriminator_yx_loss = []
    for epoch in range(NUM_EPOCHS_TRAIN):
        for i, (data, gt1) in enumerate(trainLoader_cross, 0):
            input, dummy = data
            groundTruth, dummy = gt1
            x = Variable(input.type(Tensor_gpu))  # stands for X
            y = Variable(groundTruth.type(Tensor_gpu))  # stands for Y

            # TRAIN DISCRIMINATOR
            discriminator_xy.zero_grad()
            discriminator_yx.zero_grad()

            y1 = generator_xy(x)  # Y'
            x1 = generator_yx(y)  # X'

            x2 = generator_yx(y1)  # X''
            y2 = generator_xy(x1)  # Y''

            # Real Images
            dy = discriminator_xy(y)  # D_Y
            # Fake Images
            dy1 = discriminator_xy(y1)  # D_Y'

            dx = discriminator_yx(x)  # D_X
            dx1 = discriminator_yx(x1)  # D_X'

            ad, ag = computeAdversarialLosses(dx, dx1, dy, dy1)
            # ad.backward(retain_graph=True)
            gradient_penalty = computeGradientPenaltyFor1WayGAN(discriminator_xy, y.data, y1.data) + \
                                computeGradientPenaltyFor1WayGAN(discriminator_yx, x.data, x1.data)
            # gradient_penalty.backward(retain_graph=True)
            d_loss = computeDiscriminatorLossFor2WayGan(ad, gradient_penalty)
            d_loss.backward(retain_graph=True)

            optimizer_d_xy.step()
            optimizer_d_yx.step()

            if batches_done % 50 == 0:
                # TRAIN GENERATOR
                generator_xy.zero_grad()
                generator_yx.zero_grad()

                i_loss = computeIdentityMappingLoss(x, x1, y, y1)
                c_loss = computeCycleConsistencyLoss(x, x2, y, y2)
                g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)

                g_loss.backward(retain_graph=True)

                optimizer_g_xy.step()
                optimizer_g_yx.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [I loss: %f] [C loss: %f]" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item(), i_loss.item(), c_loss.item()))

            f = open("./models/log_Train.txt", "w")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [I loss: %f] [C loss: %f] [AD: %f] [AG: %f] [GP: %f]\n" % (
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
                gradient_penalty
            ))
            f.close()

            if batches_done % 50 == 0:
                for k in range(0, y1.data.shape[0]):
                    save_image(y1.data[k], "./models/train_images/2Way/2Way_Train_%d_%d_%d.png" % (
                    epoch + 1, batches_done + 1, k + 1),
                               nrow=1,
                               normalize=True)
                torch.save(generator_xy.state_dict(),
                           './models/train_checkpoint/2Way/xy/gan2_train_' + str(epoch) + '_' + str(i) + '.pth')
                torch.save(generator_yx.state_dict(),
                           './models/train_checkpoint/2Way/yx/gan2_train_' + str(epoch) + '_' + str(i) + '.pth')
                torch.save(discriminator_xy.state_dict(),
                           './models/train_checkpoint/2Way/xy/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')
                torch.save(discriminator_yx.state_dict(),
                           './models/train_checkpoint/2Way/yx/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')
                fake_test_imgs = generator_xy(testInput)
                for k in range(0, fake_test_imgs.data.shape[0]):
                    save_image(fake_test_imgs.data[k],
                               "./models/train_test_images/2Way/2Way_Train_Test_%d_%d_%d.png" % (
                               epoch, batches_done, k),
                               nrow=1, normalize=True)

            batches_done += 1
            print("Done training discriminator on iteration: %d" % i)

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
            f = open("./models/psnr_Score.txt", "w")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "w")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))

    end_time = datetime.now()
    print(end_time - start_time)
