import torch as tc
import numpy as np
import copy
import sys, os
import Circle_Function_Class as ev
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
tmp = sys.argv[0][sys.argv[0].rfind(os.sep) + 1:]       # 返回文件名
mark = tmp[-5]
which_gpu = tmp[-4]               # 调用固定
start = tc.cuda.Event(enable_timing=True)
end = tc.cuda.Event(enable_timing=True)
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

dtype = tc.float32               # float
mps_num = 48                     # num qubits
lr = 1e-3                        # Adam's initial learning rate for automatic differentiation
it_time = 3                      # epoch
dt_print = 1                     # Number of epochs between printing
x1_axis = list()                                      # Drawing horizontal axis optimization times
identity_4 = tc.eye(4, dtype=dtype).cuda()            # Identity matrix
vol = tc.tensor(1e-3, dtype=dtype).cuda()             # Except for the first layer, the other layer gates are initialized # to unit array quantum gates plus a small amount of perturbation
con_vol = tc.tensor(1e-5, dtype=dtype).cuda()
entropy_list = list()
average = tc.tensor(0, dtype=dtype).cuda()            # The initial value used to calculate the entanglement entropy
k_bood = 48                                           # dim virtual bond
file_name = r'.\tar_data.npz'                         # file_name target MPS
out_file_name = r'.\layer_out_data.npz'               #
memory_gate_name = r'.\layer_out_data.npz'
center_position = 24                                  # Orthogonal center position
layer_num = 1                                         # num layers
gatenum = (mps_num - 1)*layer_num                     # num gates
tar_mpslist = list()
ini_state = list()
y_loss_layer = list()
y_loss_conda = list()
read_gatenum = (mps_num - 1)*(layer_num -1)
zero_gatetensor = tc.zeros(gatenum, 4, 4)

conba_gatalist = list()
layer_gatelist = list()

layer_gatelist_0 = list()     # 将门分层储存
layer_gatelist_1 = list()     # 将门分层储存
layer_gatelist_2 = list()     # 将门分层储存
layer_optimize = list()                   # 分层存储优化器


print('The quantum circuit is' + str(layer_num))
print('lr=:' + str(lr) + ', k_bood=: ' + str(k_bood) + ', A small amount of vol per unit door is: ' + str(vol))

start.record()              # Start to calculate the calculation time of the model

for targ in range(mps_num):  # Randomly initialize the target quantum state as a list without normalization
    if targ < 1:
        tar_mpslist.append(tc.rand((1, 2, k_bood), dtype=dtype).reshape(1, 2, k_bood).cuda())
    else:
        if targ >= (mps_num - 1):
            tar_mpslist.append(tc.rand((k_bood, 2, 1), dtype=dtype).reshape(k_bood, 2, 1).cuda())
        else:
            tar_mpslist.append(tc.randn((k_bood, 2, k_bood), dtype=dtype).cuda())


def mps_norm(tar_tensor_):           # Normalize the target quantum state log normalization
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


mps_norm(tar_mpslist)  # Instantiate the target MPS for orthogonal normalization


def qr_left_and_right_location(MPS_list, location, vol, feature_num=2):      # Orthogonalize the target MPS and solve its entanglement entropy
    # print('location', location)
    for k in range(location):
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
    MPS_list[location] = MPS_list[location]/tc.norm(MPS_list[location])
    # u, s, v = tc.svd(MPS_list[location].reshape(-1, MPS_list[location].shape[2]))
    u, s, v = tc.svd(MPS_list[location].reshape(MPS_list[location].shape[0], -1))
    s = s[s > vol]
    y = (-1) * tc.sum(tc.pow(s, 2) * tc.log(tc.pow(s, 2)), dim=0).item()
    return y, MPS_list                                        # y returns the entanglement entropy, mps_list returns the list() of the orthogonalized target mps


entro_tar = copy.deepcopy(tar_mpslist)

for et in range(1, mps_num):
    entropy = qr_left_and_right_location(entro_tar, et, 1e-16)[0]
    entropy_list.append(entropy)

for m in range(mps_num - 2):
    average_ = entropy_list[m]
    average = average + average_

average = average / (mps_num - 1)          # Solve for the average entanglement entropy

center_entropy = qr_left_and_right_location(entro_tar, center_position, 1e-16)[0]
print('平均纠缠熵是：{}'.format(average))
print('正交中心为第' + str(center_position) + '个tensor的MPS纠缠熵是：{}'.format(center_entropy))
for nn in range(mps_num):                           # Initial vacuum zero state
    ini_state.append(tc.tensor([1, 0], dtype=dtype).reshape(1, 2, 1).cuda())

for ut in range(gatenum):        # In order to optimize the result of the next layer better than the single layer, randomly initialize the unit array with a small amount of perturbation
    if ut < (gatenum//layer_num):
        first_layer_gate = tc.rand((4, 4), dtype=dtype).cuda()
        first_layer_gate.requires_grad = True
        layer_gatelist.append(first_layer_gate)
    else:
        vol_gate = tc.mul(tc.rand((4, 4), dtype=dtype).cuda(), vol)
        unitary_gate = tc.add(vol_gate, identity_4)
        unitary_gate.requires_grad = True
        layer_gatelist.append(unitary_gate)

mps_norm(ini_state)                                                # Normalize the initial quantum state

print('分层储存优化器进入list')

for nt in range(mps_num):                      # Export the target MPS into a numpy array
    tar_mpslist[nt] = tar_mpslist[nt].cpu().numpy()

np.savez(file_name,
         tar_mpslist0=tar_mpslist[0], tar_mpslist1=tar_mpslist[1], tar_mpslist2=tar_mpslist[2],
         tar_mpslist3=tar_mpslist[3], tar_mpslist4=tar_mpslist[4], tar_mpslist5=tar_mpslist[5],
         tar_mpslist6=tar_mpslist[6], tar_mpslist7=tar_mpslist[7], tar_mpslist8=tar_mpslist[8],
         tar_mpslist9=tar_mpslist[9], tar_mpslist10=tar_mpslist[10], tar_mpslist11=tar_mpslist[11],
         tar_mpslist12=tar_mpslist[12], tar_mpslist13=tar_mpslist[13], tar_mpslist14=tar_mpslist[14],
         tar_mpslist15=tar_mpslist[15], tar_mpslist16=tar_mpslist[16], tar_mpslist17=tar_mpslist[17],
         tar_mpslist18=tar_mpslist[18], tar_mpslist19=tar_mpslist[19], tar_mpslist20=tar_mpslist[20],
         tar_mpslist21=tar_mpslist[21], tar_mpslist22=tar_mpslist[22], tar_mpslist23=tar_mpslist[23],
         tar_mpslist24=tar_mpslist[24], tar_mpslist25=tar_mpslist[25], tar_mpslist26=tar_mpslist[26],
         tar_mpslist27=tar_mpslist[27], tar_mpslist28=tar_mpslist[28], tar_mpslist29=tar_mpslist[29],
         tar_mpslist30=tar_mpslist[30], tar_mpslist31=tar_mpslist[31], tar_mpslist32=tar_mpslist[32],
         tar_mpslist33=tar_mpslist[33], tar_mpslist34=tar_mpslist[34], tar_mpslist35=tar_mpslist[35],
         tar_mpslist36=tar_mpslist[36], tar_mpslist37=tar_mpslist[37], tar_mpslist38=tar_mpslist[38],
         tar_mpslist39=tar_mpslist[39],
         tar_mpslist40=tar_mpslist[40], tar_mpslist41=tar_mpslist[41], tar_mpslist42=tar_mpslist[42],
         tar_mpslist43=tar_mpslist[43], tar_mpslist44=tar_mpslist[44], tar_mpslist45=tar_mpslist[45],
         tar_mpslist46=tar_mpslist[46], tar_mpslist47=tar_mpslist[47])

end.record()                      # Time spent calculating up to record model

# Waits for everything to finish running
tc.cuda.synchronize()    # Wait for all cores in all streams on the current device to complete.

print('Runtime: ', start.elapsed_time(end))

