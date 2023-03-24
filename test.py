import os
import torch.backends.cudnn as cudnn
import torch.utils.data
from torchvision import transforms
from data_loader import GetLoader,load_data
from torchvision import datasets


def test(dataloader):


    model_root = 'models'

    cuda = True
    device = 3
    cudnn.benchmark = True
    # batch_size = 32
    # image_size = 28
    alpha = 0

    """load data"""

    # _,dataloader,_ = load_data("","","",batch_size = batch_size)
    # dataloader = torch.utils.data.DataLoader(
    #     dataset=dataset,
    #     batch_size=batch_size,
    #     shuffle=False,
    #     num_workers=8
    # )

    """ test """

    my_net = torch.load(os.path.join(
        model_root, 'cwru_model_epoch_current.pth'
    ))
    my_net = my_net.eval()

    if cuda:
        my_net = my_net.cuda(device)

    len_dataloader = len(dataloader)
    data_target_iter = iter(dataloader)

    i = 0
    n_total = 0
    n_correct = 0

    while i < len_dataloader:

        # test model using target data
        data_target = data_target_iter.next()
        t_img, t_label = data_target

        batch_size = len(t_label)

        if cuda:
            t_img = t_img.cuda(device)
            t_label = t_label.cuda(device)

        class_output,_ = my_net(input_data=t_img, alpha=alpha)
        pred = class_output.data.max(1, keepdim=True)[1]
        n_correct += pred.eq(t_label.data.view_as(pred)).cpu().sum()
        n_total += batch_size

        i += 1

    accu = n_correct.data.numpy() * 1.0 / n_total

    return accu
