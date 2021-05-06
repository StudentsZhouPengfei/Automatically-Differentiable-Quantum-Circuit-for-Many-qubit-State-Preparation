from torch import nn
import torch as tc
import numpy as np
import BasicFun as bf
import time
import os
import sys
from termcolor import cprint
import matplotlib
from matplotlib.pyplot import plot, savefig, figure
from TensorNetworkExpasion import TTN_basic, Vectorization, TTN_Pool_2by2to1, \
    num_correct, load_tensor_network, save_tensor_network,\
    test_accuracy_mnist, pre_process_dataset, Attention_FC, Attention_Con2d, \
    TTN_Pool_2xto1, TTN_Pool_2yto1, TTN_ConvTI_2by2to1, TTN_PoolTI_2by2to1

# matplotlib.use('Agg')


def Paras_VL_CNN_BTN_Collected1chg1():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1chg1'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1chg1_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)
    f-MNIST(d=4,chi=24)
    """

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1chg1_BP, self).__init__(num_layers=6)
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'  # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.att = Attention_FC(64*4, 16, para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x, train=True):
        eps_mask = 0.005
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = self.att(x.reshape(num, -1))
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)

                x = mask_x(x, eps_mask, train)
            x = eval('self.layer' + str(n) + '(x)')
            if n in [3, 4, 5]:
                x = mask_x(x, eps_mask, train)
        # print(x.sum(dim=1))
        return x.squeeze()


def Paras_VL_CNN_BTN_Collected1chg2():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1chg2'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1chg2_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)
    f-MNIST(d=4,chi=24)
    """

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1chg2_BP, self).__init__(num_layers=6)
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'  # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x, train=True):
        eps_mask = 0.005
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)

                x = mask_x(x, eps_mask, train)
            x = eval('self.layer' + str(n) + '(x)')
            if n in [3, 4, 5]:
                x = mask_x(x, eps_mask, train)
        # print(x.sum(dim=1))
        return x.squeeze()


def Paras_VL_CNN_BTN_Collected1chg3():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1chg3'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1chg3_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)
    f-MNIST(d=4,chi=24)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1chg3_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x, train=True):
        eps_mask = 0.01
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)

                x = mask_x(x, eps_mask, train)
            x = eval('self.layer' + str(n) + '(x)')
            if n in [3, 4]:
                x = mask_x(x, eps_mask, train)
        # print(x.sum(dim=1))
        return x.squeeze()


def Paras_VL_CNN_BTN_Collected1chg4():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1chg4'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1chg4_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)
    f-MNIST(d=4,chi=24)
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1chg4_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x, train=True):
        eps_mask = 0.005
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)

                x = mask_x(x, eps_mask, train)
            x = eval('self.layer' + str(n) + '(x)')
            if n in [3, 4]:
                x = mask_x(x, eps_mask, train)
        # print(x.sum(dim=1))
        return x.squeeze()


def Paras_VL_CNN_BTN_Collected1chg5():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1chg5'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1chg5_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)
    f-MNIST(d=4,chi=24)
    """

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1chg5_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'  # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x, train=True):
        eps_mask = 0.02
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)

                x = mask_x(x, eps_mask, train)
            x = eval('self.layer' + str(n) + '(x)')
            if n in [3, 4, 5]:
                x = mask_x(x, eps_mask, train)
        # print(x.sum(dim=1))
        return x.squeeze()


# ==========================================================
def tn_multi_classifier_CNNBTN_mnist(para=None):
    logger = bf.logger(para['log_name']+'.log', level='info')
    log = logger.logger.info
    t0 = time.time()
    if para is None:
        para = parameter_default()
    para = make_para_consistent(para)
    log('=' * 15)
    log('Using device: ' + str(para['device']))
    log('=' * 15)
    bf.print_dict(para)

    labels2mat = (para['loss_func'] == 'MSELoss')
    if para['TN'] == 'MPS':
        data_dim = 2
    else:
        data_dim = 5
    train_loader, test_loader = bf.load_mnist_and_batch(
        para['dataset'], para['classes'], para['num_samples'], None, para['batch_size'],
        data_dim=data_dim, labels2mat=labels2mat, channel=len(para['classes']),
        project_name=para['project'], dev=para['device'])

    train_loader, train_num_tot = pre_process_dataset(
        train_loader, para, para['device'])
    test_loader, test_num_tot = pre_process_dataset(
        test_loader, para, para['device'])

    img = train_loader[7][0].reshape(train_loader[3][0].shape[0], -1)
    img = img[3, :].reshape(28, 28)
    matplotlib.pyplot.imshow(img.cpu())
    matplotlib.pyplot.show()
    input()

    num_batch_train = len(train_loader)
    log('Num of training samples:\t' + str(train_num_tot))
    log('Num of testing samples:\t' + str(test_num_tot))
    log('Num of training batches:\t' + str(num_batch_train))
    log('Num of features:\t' + str(para['length']))
    log('Dataset finish processed...')

    loss_func = tc.nn.CrossEntropyLoss()

    tn = eval(para['TN'] + '_BP(para)')
    info = dict()
    info['train_acc'] = list()
    info['train_loss'] = list()
    info['test_acc'] = list()
    info['norm_coeff'] = list()
    if para['normalize_tensors'] is not None:
        tn.normalize_all_tensors(para['normalize_tensors'])

    nc = test_accuracy_mnist(tn, test_loader, para)
    log('Initially, we have test acc = ' + str(nc / test_num_tot))

    parameters_cnn = nn.ParameterList()
    parameters_btn = nn.ParameterList()
    for x in tn.parameters():
        if x.ndimension() in [7, 9]:
            parameters_btn.append(x)
        else:
            parameters_cnn.append(x)

    if parameters_cnn.__len__() > 0:
        optimizer_cnn = tc.optim.Adam(parameters_cnn, lr=para['lr'][0])
    if parameters_btn.__len__() > 0:
        optimizer_btn = tc.optim.Adam(parameters_btn, lr=para['lr'][1])

    log('Start training...')
    log('[Note: data will be save at: ' + para['data_path'] + ']')
    coeff_norm = 0
    if para['if_test']:
        titles = 'Epoch \t train_loss \t train_acc \t test_acc \t norm_coeff'
    else:
        titles = 'Epoch \t train_loss \t train_acc \t norm_coeff'
    log(titles)

    for t in range(para['it_time']):
        t_loop = time.time()
        train_loss = 0
        nc = 0
        if (num_batch_train > 1) and (t > 0):
            train_loader = bf.re_batch_data_loader(train_loader)
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(para['device']), labels.to(para['device'])

            y = tn(imgs)
            loss = loss_func(y, labels)
            with tc.no_grad():
                train_loss += loss.data.item()

            loss.backward()

            for x in tn.parameters():
                if x.ndimension() in [7, 9]:
                    s = x.shape
                    # put grad in tangent space
                    inner = tc.einsum('ac,ac->a', x.data.view(-1, s[-1]),
                                      x.grad.data.view(-1, s[-1]))
                    grad = x.grad.data.view(-1, s[-1]) - tc.einsum(
                        'a,ab->ab', inner, x.data.view(-1, s[-1]))
                    # normalize grad
                    norm = grad.norm(dim=1, p=2) + 1e-12
                    grad = tc.einsum('ab,a->ab', grad, 1 / norm)
                    # print(tc.einsum('ac,ac->a', grad, x.data.view(-1, s[-1])))
                    x.grad.data = grad.view(s)

            if parameters_cnn.__len__() > 0:
                optimizer_cnn.step()
                optimizer_cnn.zero_grad()
            if parameters_btn.__len__() > 0:
                optimizer_btn.step()
                optimizer_btn.zero_grad()

            for x in tn.parameters():
                if x.ndimension() in [7, 9]:
                    s = x.shape
                    x = x.view(-1, s[-1])
                    norm = x.data.norm(
                        dim=1, p=2)
                    x.data[:, :] = tc.einsum(
                        'ab,a->ab', x.data, 1 / norm)
                    x.data = x.data.view(s)

            if ((t + 1) % para['check_time']) == 0:
                nc0, _ = num_correct(labels, y.data)
                nc += nc0
        if ((t + 1) % para['check_time']) == 0:
            info['train_acc'].append(nc / train_num_tot)
            info['train_loss'].append(train_loss)
            info['norm_coeff'].append(coeff_norm)
            message = str(t + 1) + ': '
            message += '\t %.6g' % info['train_loss'][-1]
            message += '\t %.6g' % info['train_acc'][-1]
            if para['if_test']:
                nc = test_accuracy_mnist(
                    tn, test_loader, para)
                info['test_acc'].append(nc / test_num_tot)
                message += '\t %.6g' % info['test_acc'][-1]
            message += '\t %.6g' % info['norm_coeff'][-1]
            log(message)
        if ((t+1) % para['save_time']) == 0:
            if (train_loss == float('nan')) or (train_loss == float('inf')):
                cprint('DO NOT save MPS since NAN/INF appears', color='red')
                sys.exit(1)
            else:
                info['time_1loop'] = time.time() - t_loop
                save_tensor_network(tn, para, info,
                                        para['data_path'], para['data_exp'])
                log('MPS saved: time cost per epoch = ' + str(info['time_1loop']))
                log(titles)
    x = np.arange(para['it_time'])
    fig = figure()
    plot(x, info['test_acc'])
    savefig('../results/' + para['TN'] + '_test_acc.png')

    info['time_tot'] = time.time() - t0
    log('Total time cost = ' + str(info['time_tot']))
    return para['data_path'], para['data_exp']


def parameter_default():
    para = dict()
    para['project'] = 'CNNBTNhybrid'
    para['which_TN_set'] = 'tne'  # 'tne' or 'ctnn'
    para['TN'] = 'MPS'

    para['dataset'] = 'fashion-mnist'
    para['classes'] = list(range(10))
    para['num_samples'] = ['all'] * para['classes'].__len__()
    para['batch_size'] = 3000

    para['binary_imgs'] = False
    para['cut_size'] = [28, 28]
    para['img_size'] = [28, 28]
    # to feature cut-off; not usable yet
    para['update_f_index'] = False
    para['tol_cut_f'] = 1e-12

    para['it_time'] = 200
    para['lr'] = [1e-4, 1e-2]
    para['d'] = 2
    para['chi'] = 2

    para['linear_gauss_noise'] = None
    para['pre_normalize_mps'] = 1
    para['normalize_mps'] = False
    para['optimizer'] = 'Adam'
    para['mps_init'] = 'No.1'
    para['feature_map'] = 'taylor'
    para['feature_theta'] = 1
    para['activate_fun'] = None
    para['activate_fun_final'] = None
    para['Lagrangian'] = None
    para['Lagrangian_way'] = 0
    para['norm_p'] = 1
    para['loss_func'] = 'CrossEntropyLoss'  # MSELoss, CrossEntropyLoss, NLLLoss

    para['check_time'] = 2
    para['save_time'] = 20
    para['if_test'] = True
    para['if_load'] = True
    para['if_load_smaller_chi'] = True
    para['clear_history'] = False
    para['normalize_tensors'] = None
    para['update_way'] = 'bp'
    para['multi_gpu_parallel'] = False

    para['log_name'] = 'record'
    para['device'] = 'cuda'

    para = make_para_consistent(para)
    return para


def make_para_consistent(para):
    if 'TN' not in para:
        para['TN'] = 'MPS'
    if 'norm_p' not in para:
        para['norm_p'] = 1
    if 'binary_imgs' not in para:
        para['binary_imgs'] = False
    if para['TN'] != 'MPS':
        para['normalize_mps'] = False
        para['activate_fun'] = None
        para['activate_fun_final'] = None
    para['data_path'] = './'
    if para['feature_map'] == 'fold_2d_order1':
        para['img_size'] = [round(para['img_size'][0]/2),
                            round(para['img_size'][1]/2)]
    if para['feature_map'].lower() in ['normalized_linear',
                                       'relsig', 'tansig', 'vsigmoid']:
        if para['d'] != 2:
            bf.warning('Warning: Inconsistent para[\'d\']=%g to '
                       'feature map. Please check...' % para['d'])
            para['d'] = 2
    if para['feature_map'].lower() == 'reltansig':
        if para['d'] != 3:
            bf.warning('Warning: Inconsistent para[\'d\']=%g to '
                       'feature map. Please check...' % para['d'])
            para['d'] = 3
    para['length'] = para['img_size'][0] * para['img_size'][1]
    if 'feature_index' not in para:
        para['feature_index'] = None
    elif para['feature_index'] is not None:
        if len(para['feature_index']) > para['length']:
            bf.warning('Error: length > len(feature_index).')
            sys.exit(1)
        elif max(para['feature_index']) > (para['length'] - 1):
            bf.warning('Error: feature_index.max() > len(feature_index).')
            sys.exit(1)
        else:
            para['length'] = len(para['feature_index'])
    para['channel'] = len(para['classes'])
    para['data_exp'] = data_exp_to_save_mps(para)
    if (para['device'] != 'cpu') and (not tc.cuda.is_available()):
        para['device'] = 'cpu'
        bf.warning('Cuda is not available in the device...')
        bf.warning('Changed to \'cpu\' instead...')
    return para


def data_exp_to_save_mps(para):
    exp = para['TN'].upper() + '_L' + str(para['length']) + '_d' + str(para['d']) + '_chi' + \
               str(para['chi']) + '_classes' + str(para['classes']) + '_' + \
               para['feature_map'] + '_' + para['dataset'].upper()
    if para['dataset'].lower() in ['mnist', 'fashion-mnist', 'fashionmnist']:
        if (para['cut_size'][0] != 28) or (para['cut_size'][1] != 28):
            exp += ('_size' + str(para['cut_size']))
        if (para['img_size'][0] != 28) or (para['img_size'][1] != 28):
            exp += str(para['img_size'])
    elif para['dataset'].lower() in ['cifar10', 'cifar-10']:
        if (para['cut_size'][0] != 32) or (para['cut_size'][1] != 32):
            exp += ('_size' + str(para['cut_size']))
        if (para['img_size'][0] != 32) or (para['img_size'][1] != 32):
            exp += str(para['img_size'])
    if 'feature_index' in para:
        if para['feature_index'] is not None:
            exp += '_FindexedNum' + str(len(para['feature_index']))
    if para['binary_imgs']:
        exp += 'binary'
    return exp


def load_saved_tn_smaller_chi_d(para, path1=None):
    if para['if_load']:
        path = './data/' + para['TN'] + '/'
        exp = data_exp_to_save_mps(para)
        mps_file = os.path.join(path, exp)
        if os.path.isfile(mps_file):
            message = 'Load existing ' + para['TN'] + ' data...'
            mps, info, _ = load_tensor_network(mps_file, para)
            return mps, info, message
        elif para['if_load_smaller_chi']:
            if path1 is None:
                path1 = './data/' + para['TN'] + '_saved/'
            chi0 = para['chi']
            d0 = para['d']
            for d in range(d0, 1, -1):
                for chi in range(chi0, 1, -1):
                    para['d'] = d
                    para['chi'] = chi
                    exp = data_exp_to_save_mps(para)
                    mps_file = os.path.join(path1, exp)
                    if os.path.isfile(mps_file):
                        message = 'Load existing ' + para['TN'] + ' with (d, chi) = ' + \
                                  str((para['d'], para['chi']))
                        para['chi'], para['d'] = chi0, d0
                        mps, info, _ = load_tensor_network(
                            mps_file, para)
                        return mps, info, message
            message = 'No existing smaller-chi/d ' + \
                      para['TN'] + ' found...\n ' \
                                   'Create new ' + para['TN'] + ' data ...'
            para['chi'], para['d'] = chi0, d0
            return None, None, message
        else:
            message = 'No existing ' + para['TN'] + ' found...\n Create new ' + \
                      para['TN'] + ' data ...'
            return None, None, message
    else:
        return None, None, 'Create new ' + para['TN'] + ' data ...'


def mask_x(x, eps_mask, train):
    if train:
        mask = (x.data > eps_mask)
        x = x * mask + 1e-12
        s = x.shape
        norm = x.data.permute(0, 1, 3, 4, 2).reshape(-1, s[2]).norm(dim=1)
        norm = norm.reshape(s[0], s[3], s[4])
        x = tc.einsum('ncdxy,nxy->ncdxy', x, 1 / norm)
    return x


# ==========================================================================================
# Collected hybrid models


def Paras_VL_CNN_BTN_Collected1():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected1'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected1_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST(d=4,chi=24)         0.999633        0.9887
    f-MNIST(d=4,chi=24)       0.971017        0.8966
    f-MNIST(d=4,chi=14)       0.971883	      0.8887
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected1_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2by2to1(
            1, 1, 4, 4, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)
            x = eval('self.layer' + str(n) + '(x)')
        x = x.squeeze()
        # print(x.sum(dim=1))
        return x


def Paras_VL_CNN_BTN_Collected2():
    para = parameter_default()
    para['TN'] = 'CNN_BTN_Collected2'
    para['batch_size'] = 600
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected2_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST
    f-MNIST(d=4, chi=24)     0.971217        0.8858
    """

    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected2_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'  # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 4, kernel_size=3),  # 26*26
            nn.LayerNorm([4, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(4, 8, kernel_size=3),  # 24*24
            nn.LayerNorm([8, 24, 24], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 12*12
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=3),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_Pool_2xto1(
            1, 1, 32, 1, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2xto1(
            1, 1, 16, 1, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2xto1(
            1, 1, 8, 1, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer6 = TTN_Pool_2xto1(
            1, 1, 4, 1, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer7 = TTN_Pool_2xto1(
            1, 1, 2, 1, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer8 = TTN_Pool_2xto1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(9):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 64, 1, 4, 1).permute(
                    0, 4, 3, 1, 2)
            x = eval('self.layer' + str(n) + '(x)')
        x = x.squeeze()
        # print(x.sum(dim=1))
        return x


def Paras_VL_CNN_BTN_Collected3():
    para = parameter_default()
    para['TN'] = 'VL_CNN_BTN_Collected3'
    para['batch_size'] = 400
    para['d'] = 4
    para['chi'] = 24
    para['feature_map'] = 'cos_sin'
    para['normalize_tensors'] = 'norm2'
    para['update_way'] = 'rotate'

    para['mps_init'] = 'randn'
    para['Lagrangian_way'] = 0
    para['Lagrangian'] = None
    para['check_time'] = 5
    para['save_time'] = 1000
    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BTN_Collected3_BP(TTN_basic):
    """
                              train_acc       test_acc
    MNIST
    f-MNIST(d=4, chi=24)      0.9768        0.8862
    """
    def __init__(self, para_tn, tensors=None):
        super(VL_CNN_BTN_Collected3_BP, self).__init__(num_layers=6)
        theta = 1
        self.f_map = para_tn['feature_map']
        add_bias = False
        pre_process_tensors = 'square'   # 'normalize', 'softmax', 'square'
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])

        self.layer3 = TTN_PoolTI_2by2to1(
            1, 1, para_tn['d'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer4 = TTN_Pool_2by2to1(
            1, 1, 2, 2, para_tn['chi'], para_tn['chi'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.layer5 = TTN_Pool_2by2to1(
            1, 1, 1, 1, para_tn['chi'], para_tn['channel'],
            para_tn['device'], ini_way=para_tn['mps_init'],
            if_pre_proc_T=pre_process_tensors, add_bias=add_bias)
        self.input_tensors(tensors)

    def forward(self, x):
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(6):
            if n == 3:
                x = x.reshape(-1, 4)
                x = nn.Softmax(dim=1)(x)
                x = x.reshape(num, 8, 8, 4, 1).permute(
                    0, 4, 3, 1, 2)
            x = eval('self.layer' + str(n) + '(x)')
        x = x.squeeze()
        # print(x.sum(dim=1))
        return x


def Paras_VL_CNN():
    para = parameter_default()
    para['TN'] = 'VL_CNN'
    para['batch_size'] = 600
    para['normalize_tensors'] = 'norm2'

    para['mps_init'] = 'randn'
    para['check_time'] = 5
    para['save_time'] = 1000

    para['it_time'] = 1000
    para['lr'] = [1e-4, 2e-2]
    return para


class VL_CNN_BP(TTN_basic):
    """
                     train_acc   test_acc
    MNIST
    f-MNIST         0.962283        0.8917
    """
    def __init__(self, para_tn):
        super(VL_CNN_BP, self).__init__(num_layers=6)
        self.layer0 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3),  # 26*26
            nn.LayerNorm([8, 26, 26], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 13*13
        ).to(device=para_tn['device'])
        self.layer1 = nn.Sequential(
            nn.Conv2d(8, 32, kernel_size=4),  # 10*10
            nn.LayerNorm([32, 10, 10], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 5*5
        ).to(device=para_tn['device'])
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=2),  # 4*4
            nn.LayerNorm([64, 4, 4], eps=1e-05, elementwise_affine=True),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 2*2
        ).to(device=para_tn['device'])
        self.layer3 = nn.Sequential(
            nn.Linear(64*4, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, para_tn['channel']),
            nn.Sigmoid()
        ).to(device=para_tn['device'])

    def forward(self, x):
        num = x.shape[0]
        x = x.reshape(-1, 1, 28, 28)
        for n in range(4):
            if n == 3:
                x = x.reshape(num, -1)
            x = eval('self.layer' + str(n) + '(x)')
        x = x.squeeze()
        # print(x.sum(dim=1))
        return x




