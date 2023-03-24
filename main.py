import random
import os
import sys
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import numpy as np
from torch import save
from torch.utils.data import DataLoader,ConcatDataset
from data_loader import GetLoader,load_data
from torchvision import datasets
from torchvision import transforms
from model import CNNModel
from test import test
from torch.nn import NLLLoss,CrossEntropyLoss
from torch.nn.functional import one_hot
from CWRUDataset import CWRUDataset


# load model

def fun():
    model_root = 'models'
    cuda = True
    cudnn.benchmark = True
    lr = 1e-3
    batch_size = 16
    device = 3
    n_epoch = 100

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    dataloader_source, dataloader_target, dataloader_test = load_data("","","",batch_size = batch_size)

    
    my_net = CNNModel()

    # setup optimizer

    optimizer = optim.Adam(my_net.parameters(), lr=lr)

    loss_class = torch.nn.NLLLoss()
    loss_domain = torch.nn.NLLLoss()
    if cuda:
        my_net = my_net.cuda(device)
        loss_class = loss_class.cuda(device)
        
    for p in my_net.parameters():
        p.requires_grad = True

    # training
    best_accu_t = 0.0
    for epoch in range(n_epoch):

        len_dataloader = len(dataloader_source)
        data_source_iter = iter(dataloader_source)
        data_target_iter = iter(dataloader_target)

        for i in range(len_dataloader):

            p = float(i + epoch * len_dataloader) / n_epoch / len_dataloader
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # training model using source data
            data_source = data_source_iter.next()
            data_target = data_target_iter.next()
            t_img,_ = data_target
            s_img, s_label = data_source #有label

            my_net.zero_grad()
            batch_size = len(s_label)
            s_domain_label = torch.zeros(batch_size).long()
            t_domain_label = torch.ones(batch_size).long()
            if cuda:
                s_img = s_img.cuda(device)
                s_label = s_label.cuda(device)
                t_img = t_img.cuda(device)
                loss_domain = loss_domain.cuda(device)
                s_domain_label = s_domain_label.cuda(device)
                t_domain_label = t_domain_label.cuda(device)

            class_output,s_domain_output = my_net(input_data=s_img, alpha=alpha)
            _,t_domain_output = my_net(input_data = t_img , alpha = alpha)
            
            if cuda:
                class_output =class_output.cuda(device)
                s_domain_output = s_domain_output.cuda(device)
                t_domain_output = t_domain_output.cuda(device)
            
            # s_label = one_hot(s_label,9)
            # s_label = s_label.float()
            err_s_label = loss_class(class_output, s_label)
            err_s_domain = loss_domain(s_domain_output , s_domain_label)
            err_t_domain = loss_domain(t_domain_output , t_domain_label)
            err = err_s_label +0.1* err_s_domain +0.1* err_t_domain
            err.backward()
            optimizer.step()

            sys.stdout.write('\r epoch: %d, [iter: %d / all %d], err_s_label: %f, err_s_domain: %f, err_t_domain: %f' \
                % (epoch, i + 1, len_dataloader, err_s_label.data.cpu().numpy(),
                    err_s_domain.data.cpu().numpy(), err_t_domain.data.cpu().item()))
            sys.stdout.flush()
            torch.save(my_net, '{0}/cwru_model_epoch_current.pth'.format(model_root))

        print('\n')
        accu_s = tesct(dataloader_target)
        print('Accuracy of the %s dataset: %f' % ('source', accu_s))
        accu_t = test(dataloader_test)
        print('Accuracy of the %s dataset: %f\n' % ('target', accu_t))
        if accu_t > best_accu_t:
            best_accu_s = accu_s
            best_accu_t = accu_t
            torch.save(my_net, '{0}/cwru_model_epoch_best.pth'.format(model_root))
    
    print('============ Summary ============= \n')
    print('Accuracy of the %s dataset: %f' % ('source', best_accu_s))
    print('Accuracy of the %s dataset: %f' % ('target', best_accu_t))
    print('Corresponding model was save in ' + model_root + '/cwru_model_epoch_best.pth')

    # _,final,_ = load_data("","","",1000)
    # output =  my_net(final,alpha = alpha)
    # save(output , "stft.torch")

if __name__ == '__main__':
    fun()