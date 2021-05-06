import torch as tc
# import numpy as np
import copy
import os,sys
from torch.utils.checkpoint import checkpoint
import BasicFunSJR as bfs
from CNNBTN import Paras_VL_CNN_BTN_Collected1chg1
import BasicFun as bf

tmp = sys.argv[0][sys.argv[0].rfind(os.sep) + 1:]       # 返回文件名
mark = tmp[-5]
which_gpu = tmp[-4]               # 调用固定

para = Paras_VL_CNN_BTN_Collected1chg1()
para['dataset'] = 'fashion-mnist'
para['device'] = bf.choose_device(which_gpu)
para['log_name'] = './record' + mark + which_gpu

# import MPS_optimize1_project.test_svd as ec
# import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICE'] = '0'
dtype = tc.float32
eye = tc.eye(2, dtype=dtype).to(para['device'])
out_list = list([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
mps_num = 48
ini_state = list()
t_list = list([[], [], [], [], [], [], [], [], [], []])

for nn in range(mps_num):                           # Initial vacuum zero state
    ini_state.append(tc.tensor([1, 0], dtype=dtype).reshape(1, 2, 1).to(para['device']))


def mps_norm(tar_tensor_):           # 对目标量子态进行归一化 log归一化
    tv = tc.einsum('asb,asd->bd', tar_tensor_[0].data, tar_tensor_[0].data)
    t_norm = tc.norm(tv)
    tv = tv / t_norm
    tar_tensor_[0] = tar_tensor_[0].data / tc.sqrt(t_norm)
    for gt in range(1, mps_num):
        if gt < mps_num - 1:
            tv = tc.einsum('ac,asb,csd->bd', tv, tar_tensor_[gt].data, tar_tensor_[gt].data)
        else:
            tv = tc.einsum('ac,asb,csd->bd', tv, tar_tensor_[gt].data, tar_tensor_[gt].data)
        norm_t = tc.norm(tv)
        tv = tv / norm_t
        tar_tensor_[gt] = tar_tensor_[gt] / tc.sqrt(norm_t)


mps_norm(ini_state)


class Evolve:
    def __init__(self, length, chi, d, gatenum, layer):
        self.length = length
        self.chi = chi
        self.d = d
        self.tensor_list = list()
        self.identity = tc.einsum('ab,cg->abcg', eye, eye).permute(0, 1, 3, 2).reshape(2, 4, 2)
        self.gatenum = gatenum
        self.layer = layer
        self.list1 = list()
        self.device = None
        self.tar_list = list()                       # 目标MPS态

    def init_tensor_list(self, tensor_lists=None):  # 初始化MPS态的单元tensor
        # print('调用初始化输入量子态')
        if tensor_lists is None:
            for lt in range(self.length):
                unittensor = tc.rand((self.chi, self.d, self.chi)).to(para['device'])
                self.tensor_list.append(unittensor)
        else:
            assert len(tensor_lists) >= self.length
            self.tensor_list = tensor_lists

    @staticmethod
    def contract(tensor, gate, which):
        if which == 0:
            tensor1 = tc.einsum('iba,gfb->igaf', tensor, gate)
            s_1 = tensor1.shape
            ten2 = tensor1.reshape(s_1[0], s_1[1], s_1[2] * s_1[3])
            return ten2
        else:
            tensor1 = tc.einsum('ajc,ghj->ahgc', tensor, gate)
            s_2 = tensor1.shape
            ten2 = tensor1.reshape(s_2[0] * s_2[1], s_2[2], s_2[3])
            return ten2

    def full_tensor(self, tar_mps):
        self.tar_list = tar_mps
        tensor = self.tar_list[0].data
        for nlt in range(1, self.length):
            tensor_ = self.tar_list[nlt].data
            tensor = tc.einsum('asb,bdf->asdf', tensor, tensor_)
            size = tensor.shape
            tensor = tensor.reshape(size[0], size[1]*size[2], size[3])
        return tc.squeeze(tensor)                          # 从数组的形状中删除单维度条目，即把shape中为1的维度去掉

    def evolve_mps(self, gatelist):                  # 将不同层的量子门演化到目标量子态
        for lt in range(0, self.layer):
            # print('层数：', str(lt))
            for et in range(self.gatenum // self.layer):  # 单层门的个数
                # print('演化的量子门次序：', str(et + (lt * self.gatenum // self.layer)))
                # print('演化前的MPS态：   ', self.tensor_list)
                us, _, vs = tc.svd(gatelist[et + (lt * self.gatenum // self.layer)], compute_uv=True)
                uni_gate1 = us.mm(vs.t())
                # print('检查第', str(et + (lt * self.gatenum // self.layer)), '个量子门是否为幺正门')
                # print(uni_gate1.data.mm(uni_gate1.data.t()))
                uni_gate1 = uni_gate1.reshape(2, 2, 2, 2).permute(0, 1, 3, 2)
                uni_gate1 = uni_gate1.reshape(2, 4, 2)
                con_tensor1 = self.contract(self.tensor_list[et], uni_gate1, 0)
                # con_tensor1 = checkpoint(self.contract(self.tensor_list[et], uni_gate1, 0), )
                self.tensor_list[et] = con_tensor1  # 将门作用产生的量子态存入新的list
                # print('将第', str(et + (lt * self.gatenum // self.layer)), '个量子门演化到MPS上的结果', self.tensor_list)
                con_tensor2 = self.contract(self.tensor_list[et + 1], self.identity.data, 1)
                # con_tensor2 = checkpoint(self.contract(self.tensor_list[et + 1], self.identity.data, 1), )
                self.tensor_list[et + 1] = con_tensor2
        # print('检查演化后量子态是否归一：  ', self.inner_dot())

    def mps_log_norm(self):                       # 对MPS进行归一化， 借助MPS优势，跨越指数墙
        vecs = [None]
        v = tc.einsum('asb,asd->bd', self.tensor_list[0].data, self.tensor_list[0].data)
        vecs.append(v)
        norm = tc.norm(v)
        # print(norm)
        v = v / norm
        self.tensor_list[0] = self.tensor_list[0].data / tc.sqrt(norm)
        # print(tc.norm(self.tensor_list[0]))
        # norm = tc.log(norm)
        for nlt in range(1, self.length):
            if nlt < self.length - 1:
                v = tc.einsum('ac,asb,csd->bd', v, self.tensor_list[nlt].data, self.tensor_list[nlt].data)
            else:
                v = tc.einsum('ac,asb,csd->bd', v, self.tensor_list[nlt].data, self.tensor_list[nlt].data)
            _norm = tc.norm(v)
            # print(_norm)
            v = v / _norm
            vecs.append(v)
            # norm = norm + tc.log(_norm)
            self.tensor_list[nlt] = self.tensor_list[nlt] / tc.sqrt(_norm)
        # print(self.tensor_list[-1].shape)

    def inner_dot(self):  # 通过对整个MPS做内积，验证量子态是否归一
        s = list()
        for st in range(self.length):
            s.append(self.tensor_list[st].data)
        for mt in range(self.length - 1):
            in_tensor = tc.einsum('asd,dzx->aszx', s[mt].data, s[mt + 1].data)
            in_size = in_tensor.shape
            in_tensor1 = in_tensor.reshape(in_size[0], in_size[1] * in_size[2], in_size[3])
            s[mt + 1] = in_tensor1
        inner_tensor = tc.einsum('acd,acd-> ', s[self.length - 1].data, s[self.length - 1].data)
        print('验证MPS是否归一化：', inner_tensor)  # 验证正交
        return inner_tensor

    def mps_norm(self):                                  # QR分解来归一化MPS，存在近似误差。 存在指数墙问题
        r_list = list()
        tensor_size = self.tensor_list[0].shape
        rep_site_tensor = self.tensor_list[0].reshape(tensor_size[0] * tensor_size[1], tensor_size[2])
        # 对局域mps，reshape第一个物理指标和虚拟指标
        q1, r1 = tc.qr(rep_site_tensor)
        r2 = r1 / tc.norm(r1)
        r_list.append(r2)
        self.tensor_list[0] = q1.reshape(tensor_size[0], tensor_size[1], tensor_size[2])
        for zt in range(1, self.length - 1):
            tensor = tc.einsum('ab, bcd->acd', r_list[zt - 1], self.tensor_list[zt])
            tensor_size_n = tensor.shape
            rep_site_tensor_n = tensor.reshape(tensor_size_n[0] * tensor_size_n[1], tensor_size_n[2])
            qn, rn = tc.qr(rep_site_tensor_n)
            rnn = rn / tc.norm(rn)
            r_list.append(rnn)
            self.tensor_list[zt] = qn.reshape(tensor_size_n[0], tensor_size_n[1], tensor_size_n[2])
        tensor_1 = tc.einsum('ab,bcd->acd', r_list[-1], self.tensor_list[-1])
        norm = tc.norm(tensor_1)
        tensor_2 = tensor_1 / norm
        self.tensor_list[-1] = tensor_2

    def log_fidelity(self, tensors):
        v = tc.einsum('asb,asd->bd', self.tensor_list[0], tensors[0])
        norm = tc.norm(v)
        v = v / norm
        norm = tc.log(norm)
        for nllt in range(1, self.length):
            if nllt < self.length - 1:
                v = tc.einsum('ac,asb,csd->bd', v,
                              self.tensor_list[nllt],
                              tensors[nllt])
            else:
                v = tc.einsum('ac,asb,csd->bd', v,
                              self.tensor_list[nllt],
                              tensors[nllt])
            _norm = tc.norm(v)
            # print('log——norm:  ', norm)
            # print('-norm:  ', _norm)
            # print('log_norm_:  ', tc.log(_norm))
            v = v / _norm
            norm = norm + tc.log(_norm)
            # print('总的log_norm:  ', norm)
            # print('=============')
        # print('log——norm:  ', norm)
        # print('-norm:  ', _norm)
        # print('log_norm_:  ', tc.log(_norm))
        # print('总的log_norm:  ', tc.abs(norm))
        # print('=============')
        return - norm / self.length

    def fidelity(self, tensors):                  # 直接对量子态做归一化，维数指数增大，不具备MPS的优点
        s = list()
        for st in range(self.length):
            s.append(self.tensor_list[st])
        for mt in range(self.length - 1):
            in_tensor = tc.einsum('asd,dzx->aszx', s[mt], s[mt + 1])
            in_size = in_tensor.shape
            in_tensor1 = in_tensor.reshape(in_size[0], in_size[1] * in_size[2], in_size[3])
            s[mt + 1] = in_tensor1
        k = list()
        for st in range(self.length):
            k.append(tensors[st])
        for zt in range(self.length - 1):
            # print(k[zt].shape)
            # print(k[zt + 1].shape)
            in_tensor2 = tc.einsum('asd,dzx->aszx', k[zt], k[zt + 1])
            in_size2 = in_tensor2.shape
            in_tensor3 = in_tensor2.reshape(in_size2[0], in_size2[1] * in_size2[2], in_size2[3])
            # print(in_tensor3.shape)
            # print('========================')
            k[zt + 1] = in_tensor3
        inner_tensor = tc.einsum('acd,acd-> ', s[-1], k[-1])
        inner_tensor = - tc.log(tc.abs(inner_tensor)) / self.length
        # print('计算梯度', self.tensor_list)
        return inner_tensor

    def layered_optimization(self):                     # 将每层量子门的演化结果 存进一个新的list() 然后进行交错优化
        layer_list = list()
        for lay in range(self.length):
            layer_list.append(0)
        for nlist in range(0, self.length):
            layer_list[nlist] = self.tensor_list[nlist].data
        return layer_list

    def out_optimization(self):                     # 将分层演化的结果存进硬盘
        con_list = list()
        for con in range(self.length):
            con_list.append(0)
        for nlist in range(0, self.length):
            con_list[nlist] = self.tensor_list[nlist].cpu().data
        return con_list

    def layered_evolve_mps(self, gatelist, lt):          # 仅控制单层量子门演化 lt 门的层数
        for et in range(self.gatenum // self.layer):  # 单层门的个数
            # print('演化的量子门次序：', str(et + (lt * self.gatenum // self.layer)))
            # print('演化前的MPS态：   ', self.tensor_list)
            us, _, vs = tc.svd(gatelist[et + (lt * self.gatenum // self.layer)], compute_uv=True)
            uni_gate1 = us.mm(vs.t())
            # print('检查第', str(et + (lt * self.gatenum // self.layer)), '个量子门是否为幺正门')
            # print(uni_gate1.data.mm(uni_gate1.data.t()))
            uni_gate1 = uni_gate1.reshape(2, 2, 2, 2).permute(0, 1, 3, 2)
            uni_gate1 = uni_gate1.reshape(2, 4, 2)
            con_tensor1 = self.contract(self.tensor_list[et], uni_gate1, 0)
            self.tensor_list[et] = con_tensor1  # 将门作用产生的量子态存入新的list
            # print('将第', str(et + (lt * self.gatenum // self.layer)), '个量子门演化到MPS上的结果', self.tensor_list)
            con_tensor2 = self.contract(self.tensor_list[et + 1], self.identity.data, 1)
            self.tensor_list[et + 1] = con_tensor2

    def qr_left_and_right_location(self, MPS_list, location, vol, feature_num=2):  # 对目标MPS进行正交，并求解其纠缠熵
        # print('location', location)
        for k in range(location):             # 正交归一化的MPS_list, 正交中心位置：location, 小量：vol, 物理指标feature_num=2
            # print('k', k)
            q, r = tc.qr(MPS_list[k].reshape(-1, MPS_list[k].shape[2]))
            r = r
            MPS_list[k] = q.reshape(-1, feature_num, q.shape[1])
            MPS_list[k + 1] = tc.einsum('nl, lmk-> nmk', [r, MPS_list[k + 1]])
        for i in range(len(MPS_list) - 1, location, -1):
            # print('i', i)
            q, r = tc.qr(MPS_list[i].reshape(MPS_list[i].shape[0], -1).t())
            q_shape = q.t().shape
            MPS_list[i] = q.t().reshape(q_shape[0], feature_num, -1)
            r = r
            MPS_list[i - 1] = tc.einsum('ldk, nk-> ldn', [MPS_list[i - 1], r])
        MPS_list[location] = MPS_list[location] / tc.norm(MPS_list[location])
        # u, s, v = tc.svd(MPS_list[location].reshape(-1, MPS_list[location].shape[2]))
        u, s, v = tc.svd(MPS_list[location].reshape(MPS_list[location].shape[0], -1))
        s = s[s > vol]
        y = (-1) * tc.sum(tc.pow(s, 2) * tc.log(tc.pow(s, 2)), dim=0).item()
        return y, MPS_list

    def read_layer_out_optimization(self, lt, it_time):          # 初始化每层的起始值
        if it_time == 0:
            if lt == self.layer - 1:
                self.init_tensor_list(copy.deepcopy(ini_state))
            else:
                self.init_tensor_list(copy.deepcopy(out_list[lt]))
        else:
            if lt == 0:
                self.init_tensor_list(copy.deepcopy(ini_state))
            else:
                self.init_tensor_list(copy.deepcopy(out_list[lt - 1]))

    def storage_layer_out_optimization(self, lt, it_time):          # 将每层结果存进一个新的list()
        if it_time == 0:
            t_list[lt] = []
            # t_list[lt] = copy.deepcopy(self.tensor_list)
            for n in range(self.length):
                t_list[lt].append(self.tensor_list[n].data * 1)
            out_list[lt] = t_list[lt]