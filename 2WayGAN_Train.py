import torch.optim as optim
from torchvision.utils import save_image
from datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *

if __name__ == "__main__":

    start_time = datetime.now()

    # Creating generator and discriminator
    generator = Generator()
    generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load('./gan1_pretrain_55_87.pth'))
    generator.train()

    discriminator = Discriminator()
    discriminator = nn.DataParallel(discriminator)

    if torch.cuda.is_available():
        generator.cuda(device=device)
        discriminator.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

    # MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_d = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    # Training Network
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
            trainInput = Variable(input.type(Tensor_gpu))   # stands for X
            realImgs = Variable(groundTruth.type(Tensor_gpu))   # stands for Y

            # TRAIN DISCRIMINATOR
            discriminator.zero_grad()
            fake_imgs = generator(trainInput)   # stands for Y'
            x1 = generator(realImgs)    # stands for x'

            x2 = generator(fake_imgs)   # stands for x''
            y2 = generator(x1)          # stands for y''

            # Real Images
            realValid = discriminator(realImgs)     # stands for D_Y
            # Fake Images
            fakeValid = discriminator(fake_imgs)     # stands for D_Y'

            dx = discriminator(trainInput)      # stands for D_X
            dx1 = discriminator(x1)             # stands for D_X'

            ad, ag = computeAdversarialLosses(discriminator, trainInput, x1, realImgs, fake_imgs)
            ad.backward(retain_graph=True)
            gradient_penalty = computeGradientPenaltyFor2Way(discriminator, trainInput, x1, realImgs, fake_imgs)
            gradient_penalty.backward(retain_graph=True)
            d_loss = computeDiscriminatorLossFor2WayGan(ad, gradient_penalty)
            d_loss.backward(retain_graph=True)

            optimizer_d.step()

            if batches_done % 50 == 0:
                # TRAIN GENERATOR
                generator.zero_grad()
                ag.backward(retain_graph=True)
                i_loss = computeIdentityMappingLoss(trainInput, x1, realImgs, fake_imgs)
                i_loss.backward(retain_graph=True)
                c_loss = computeCycleConsistencyLoss(trainInput, x2, realImgs, y2)
                c_loss.backward(retain_graph=True)
                g_loss = computeGeneratorLossFor2WayGan(ag, i_loss, c_loss)
                g_loss.backward(retain_graph=True)
                optimizer_g.step()

            print("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item()))

            f = open("./models/log_Train.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]\n" % (
                epoch + 1, NUM_EPOCHS_TRAIN, i + 1, len(trainLoader_cross), d_loss.item(), g_loss.item()))
            f.close()

            if batches_done % 50 == 0:
                for k in range(0, fake_imgs.data.shape[0]):
                    save_image(fake_imgs.data[k], "./models/train_images/2Way_Train_%d_%d_%d.png" % (epoch, batches_done, k),
                               nrow=1,
                               normalize=True)
                torch.save(generator.state_dict(),
                           './models/train_checkpoint/gan2_train_' + str(epoch) + '_' + str(i) + '.pth')
                torch.save(discriminator.state_dict(),
                           './models/train_checkpoint/discriminator2_train_' + str(epoch) + '_' + str(i) + '.pth')
                fake_test_imgs = generator(testInput)
                for k in range(0, fake_test_imgs.data.shape[0]):
                    save_image(fake_test_imgs.data[k],
                               "./models/train_test_images/2Way_Train_Test_%d_%d_%d.png" % (epoch, batches_done, k),
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
            trainInput = Variable(input.type(Tensor_gpu))
            realImgs = Variable(groundTruth.type(Tensor_gpu))
            output = generator(trainInput)
            loss = criterion(output, realImgs)
            psnr = 10 * torch.log10(1 / loss)
            psnrAvg += psnr

            for k in range(0, output.data.shape[0]):
                save_image(output.data[k],
                           "./models/test_images/test_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, realImgs.data.shape[0]):
                save_image(realImgs.data[k],
                           "./models/gt_images/gt_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1),
                           nrow=1,
                           normalize=True)
            for k in range(0, trainInput.data.shape[0]):
                save_image(trainInput.data[k],
                           "./models/input_images/input_%d_%d_%d.png" % (batches_done + 1, j + 1, k + 1), nrow=1,
                           normalize=True)

            batches_done += 5
            print("Loss loss: %f" % loss)
            print("PSNR Avg: %f" % (psnrAvg / (j + 1)))
            f = open("./models/psnr_Score.txt", "a+")
            f.write("PSNR Avg: %f" % (psnrAvg / (j + 1)))
        f = open("./models/psnr_Score.txt", "a+")
        f.write("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))
        print("Final PSNR Avg: %f" % (psnrAvg / len(testLoader)))

    end_time = datetime.now()
    print(end_time - start_time)
