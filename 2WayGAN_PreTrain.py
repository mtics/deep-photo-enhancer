import torch.optim as optim
from torchvision.utils import save_image
from _datetime import datetime
from libs.compute import *
from libs.constant import *
from libs.model import *

if __name__ == "__main__":

    start_time = datetime.now()

    # Creating generator and discriminator
    generator_xy = Generator()
    generator_yx = Generator()

    generator_xy = nn.DataParallel(generator_xy)
    generator_yx = nn.DataParallel(generator_yx)

    if torch.cuda.is_available():
        generator_xy.cuda(device=device)
        generator_yx.cuda(device=device)

    # Loading Training and Test Set Data
    trainLoader1, trainLoader2, trainLoader_cross, testLoader = data_loader()

    ### MSE Loss and Optimizer
    criterion = nn.MSELoss()

    optimizer_g_xy = optim.Adam(generator_xy.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))
    optimizer_g_yx = optim.Adam(generator_yx.parameters(), lr=LEARNING_RATE, betas=(BETA1, BETA2))

    ### GENERATOR PRE-TRAINING LOOP
    print("Pre-training loop starting")
    batches_done = 0
    running_loss_xy = 0.0
    running_loss_yx = 0.0
    running_losslist_xy = []
    running_losslist_yx = []
    for epoch in range(NUM_EPOCHS_PRETRAIN):
        for i, (target, input) in enumerate(trainLoader1, 0):
            unenhanced_image = input[0]
            enhanced_image = target[0]
            unenhanced = Variable(unenhanced_image.type(Tensor_gpu))    # X
            enhanced = Variable(enhanced_image.type(Tensor_gpu))        # Y

            optimizer_g_xy.zero_grad()

            generated_enhanced_image = generator_xy(unenhanced)   # X->Y'
            loss_xy = criterion(generated_enhanced_image, enhanced)
            loss_xy.backward()
            optimizer_g_xy.step()

            optimizer_g_yx.zero_grad()

            generated_unenhanced_image = generator_yx(enhanced)   # Y->X'
            loss_yx = criterion(unenhanced, generated_unenhanced_image)
            loss_yx.backward()
            optimizer_g_yx.step()

            # Print statistics
            running_loss_xy += loss_xy.item()
            running_loss_yx += loss_yx.item()
            running_losslist_xy.append(loss_xy.item())
            running_losslist_yx.append(loss_yx.item())

            f = open("./models/log_PreTraining.txt", "a+")
            f.write("[Epoch %d/%d] [Batch %d/%d] [G loss(xy): %f]  [G loss(yx): %f]\n" % (
                epoch + 1, NUM_EPOCHS_PRETRAIN + 1, i + 1, len(trainLoader1), loss_xy.item(), loss_yx.item()))
            f.close()

            # if i % 200 == 200:    # print every 200 mini-batches
            if i % 1 == 0:
                print('[%d, %5d] loss_xy: %.5f   loss_yx: %.5f' % (epoch + 1, i + 1, running_loss_xy / 5, running_loss_yx / 5))
                running_loss_xy = 0.0
                running_loss_xy = 0.0

                save_image(generated_enhanced_image.data,
                           "./models/pretrain_images/xy/gan2_pretrain_%d_%d.png" % (epoch + 1, i + 1),
                           nrow=8,
                           normalize=True)
                torch.save(generator_xy.state_dict(),
                           './models/pretrain_checkpoint/xy/gan2_pretrain_' + str(epoch + 1) + '_' + str(i + 1) + '.pth')

                save_image(generated_unenhanced_image.data,
                           "./models/pretrain_images/yx/gan2_pretrain_%d_%d.png" % (epoch + 1, i + 1),
                           nrow=8,
                           normalize=True)
                torch.save(generator_yx.state_dict(),
                           './models/pretrain_checkpoint/yx/gan2_pretrain_' + str(epoch + 1) + '_' + str(i + 1) + '.pth')

    end_time = datetime.now()
    print(end_time-start_time)

    f = open("./models/log_xy_PreTraining_LossList.txt", "a+")
    for item in running_losslist_xy:
        f.write('%f\n' % item)
    f.close()

    f = open("./models/log__yxPreTraining_LossList.txt", "a+")
    for item in running_losslist_yx:
        f.write('%f\n' % item)
    f.close()
