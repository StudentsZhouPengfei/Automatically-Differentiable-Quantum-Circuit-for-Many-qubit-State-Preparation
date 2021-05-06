import torch as tc
import numpy as np
import copy
import os,sys
import Circle_Function_Class_A0 as ev
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from torch.optim.lr_scheduler import StepLR
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

start = tc.cuda.Event(enable_timing=True)
end = tc.cuda.Event(enable_timing=True)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['CUDA_VISIBLE_DEVICE'] = '0'

# tc.manual_seed(7)              # 固定随机数，使产生的随机数可以复现
dtype = tc.float32               # float 监控norm
mps_num = 48                     #  num qubits
lr = 1e-2                        # Adam's initial learning rate for automatic differentiation
it_time = 10                     # Epoch scanned once
pt_time = 250                    # num scanned
dt_print = 10                    # Number of epochs between printing
step_size = it_time * pt_time // 5       # lr learning rate decreasing interval epoch
x1_axis = list()                                      # Drawing horizontal axis optimization times
identity_4 = tc.eye(4, dtype=dtype).to(para['device'])          # Identity matrix
vol = tc.tensor(1e-3, dtype=dtype).to(para['device'])           # Except for the first layer, the other layer gates are initialized # to unit array quantum gates plus a small amount of perturbation
con_vol = tc.tensor(1e-5, dtype=dtype).to(para['device'])
entropy_list = list()                                           #
average = tc.tensor(0, dtype=dtype).to(para['device'])          # The initial value used to calculate the entanglement entropy
k_bood = 64                                                     # dim virtual bond
file_name = r'./tar_data.npz'                                   # file_name target MPS
out_file_name = r'./layer_out_data.npz'
center_position = 24                                # Orthogonal center position

layer_num = 1                                       # num layers
gatenum = (mps_num - 1)*layer_num                   # num gates
tar_mpslist = list()
ini_state = list()
y_loss_layer = list()
y_loss_conda = list()

read_gatenum = (mps_num - 1)*(layer_num -1)
zero_gatetensor = tc.zeros(gatenum, 4, 4)

conba_gatalist = list()
layer_gatelist = list()       # In the subsequent reshape into a third-order tensor of (2, 4, 2)

layer_gatelist_0 = list()    # Hierarchical storage of quantum gates
layer_gatelist_1 = list()    # Hierarchical storage of quantum gates
layer_gatelist_2 = list()    # Hierarchical storage of quantum gates
layer_gatelist_3 = list()    # Hierarchical storage of quantum gates
layer_gatelist_4 = list()    # Hierarchical storage of quantum gates
layer_gatelist_5 = list()    # Hierarchical storage of quantum gates

layer_optimize = list()                   # Tiered storage optimizer
loss_ = list([list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]), list([]),
              list([]), list([]), list([])])

half_entropy_list = list([])               # Make a heat map
half_entropy_list.append(tc.zeros([pt_time+1, mps_num-1]))  # Entangled Entropy for the Target Last Time
number_list = list([0])   

print('The quantum circuit is' + str(layer_num))
print('lr=:' + str(lr) + ', k_bood=: ' + str(k_bood) + ', A small amount of vol per unit door is: ' + str(vol))

data = np.load(file_name)
tar_mpslist.append(tc.from_numpy(data['tar_mpslist0']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist1']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist2']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist3']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist4']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist5']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist6']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist7']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist8']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist9']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist10']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist11']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist12']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist13']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist14']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist15']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist16']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist17']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist18']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist19']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist20']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist21']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist22']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist23']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist24']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist25']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist26']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist27']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist28']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist29']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist30']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist31']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist32']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist33']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist34']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist35']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist36']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist37']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist38']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist39']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist40']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist41']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist42']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist43']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist44']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist45']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist46']).to(para['device']))
tar_mpslist.append(tc.from_numpy(data['tar_mpslist47']).to(para['device']))


def fprint(content, file=None, print_screen=True, append=True):
    if file is None:
        file = './record.log'
    if append:
        way = 'ab'
    else:
        way = 'wb'
    with open(file, way, buffering=0) as log:
        log.write((content + '\n').encode(encoding='utf-8'))
    if print_screen:
        print(content)


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


def half_entropy(out_mps):
    for ht in range(1, mps_num):
        h_entropy = qr_left_and_right_location(out_mps, ht, 1e-16)[0]
        half_entropy_list[0][number_list[0], ht-1] = h_entropy
    number_list[0] = number_list[0] + 1


entro_tar = copy.deepcopy(tar_mpslist)

for et in range(1, mps_num):
    entropy = qr_left_and_right_location(entro_tar, et, 1e-16)[0]
    entropy_list.append(entropy)

for m in range(mps_num - 2):
    average_ = entropy_list[m]
    average = average + average_

average = average / (mps_num - 1)          # Solve for the average entanglement entropy

center_entropy = qr_left_and_right_location(entro_tar, center_position, 1e-16)[0]
print('The average entanglement entropy is：{}'.format(average))
print('Orthogonal center is the first' + str(center_position) + 'The MPS entanglement entropy of a tensor is：{}'.format(center_entropy))

for nn in range(mps_num):                           # Initial vacuum zero state
    ini_state.append(tc.tensor([1, 0], dtype=dtype).reshape(1, 2, 1).to(para['device']))


for ut in range(gatenum):        # In order to optimize the result of the next layer better than the single layer, randomly initialize the unit array with a small amount of perturbation
    if ut < (gatenum//layer_num):
        first_layer_gate = tc.rand((4, 4), dtype=dtype).to(para['device'])
        first_layer_gate.requires_grad = True
        layer_gatelist.append(first_layer_gate)
    else:
        vol_gate = tc.mul(tc.rand((4, 4), dtype=dtype).to(para['device']), vol)
        unitary_gate = tc.add(vol_gate, identity_4)
        unitary_gate.requires_grad = True
        layer_gatelist.append(unitary_gate)

mps_norm(ini_state)                                                 # Normalize the initial quantum state

print('分层储存优化器进入list')


for it in range(gatenum):                   # Separate the list of layered optimized loss according to the number of layers
    if it < (gatenum//layer_num)*1:
        layer_gatelist_0.append(layer_gatelist[it])
    else:
        if it < (gatenum//layer_num)*2:
            layer_gatelist_1.append(layer_gatelist[it])
        else:
            if it < (gatenum//layer_num)*3:
                layer_gatelist_2.append(layer_gatelist[it])
            else:
                if it < (gatenum//layer_num)*4:
                    layer_gatelist_3.append(layer_gatelist[it])
                else:
                    if it < (gatenum//layer_num)*5:
                        layer_gatelist_4.append(layer_gatelist[it])
                    else:
                        layer_gatelist_5.append(layer_gatelist[it])

lay_optimize_0 = tc.optim.Adam(layer_gatelist_0, lr=lr)             # The quantum gate parameters of hierarchical optimization are coordinated after the hierarchical optimization

layer_optimize.append(lay_optimize_0)                               # Three-tier optimizer

scheduler_0 = StepLR(lay_optimize_0, step_size=step_size, gamma=0.1)


scheduler = list()
scheduler.append(scheduler_0)


evo = ev.Evolve(mps_num, k_bood, 2, gatenum, layer_num)
evo.init_tensor_list(copy.deepcopy(ini_state))

for bt in range(layer_num):
    print('初始化第' + str(bt) + '的学习率：', layer_optimize[bt].defaults['lr'])

start.record()              # Start to calculate the calculation time of the model

for pt in range(pt_time):   # The number of times that the staggered optimization is located
    fprint('Circle优化位于第' + str(pt) + '次', file=para['log_name'])
    for lay_num in range(layer_num):
        fprint('Circle优化位于第' + str(lay_num) + '层', file=para['log_name'])
        for vt in range(it_time):
            for llt in range(lay_num, lay_num + 1):  # 先将优化层进行演化，演化完成后将其存进新的list，作为下一层初始
                evo.layered_evolve_mps(layer_gatelist, llt)
                if vt == it_time - 1:
                    evo.storage_layer_out_optimization(llt, 0)
            for at in range(lay_num + 1, layer_num):  # 将不变分的量子门演化进入线路
                evo.layered_evolve_mps(layer_gatelist, at)
            lay_loss = evo.log_fidelity(tar_mpslist)  # 借助了mps跨越指数复杂度的优势
            if (vt % dt_print) == 0:
                fprint('At t = ' + str(vt) + ', loss = ' + str(lay_loss.item()), file=para['log_name'])
                loss_[lay_num].append(lay_loss.item())
            lay_loss.backward()
            layer_optimize[lay_num].step()
            layer_optimize[lay_num].zero_grad()
            if (vt % dt_print) == 0:
                fprint("第%d个epoch的学习率：%f" % (vt, layer_optimize[lay_num].param_groups[0]['lr']),
                       file=para['log_name'])
            scheduler[lay_num].step()
            tc.cuda.empty_cache()  # Delete unnecessary variables
            if lay_num == layer_num-1:
                if vt == it_time - 1:
                    half_entropy(evo.out_optimization())
            if vt == it_time - 1:
                evo.read_layer_out_optimization(lay_num, 0)
            else:
                evo.read_layer_out_optimization(lay_num, 1)

half_entropy(tar_mpslist)   # The last line of the heat map is the information about the entanglement of the target state

bfs.save('.', 'out_memory_half_entropy_data', [half_entropy_list], ['half_entropy'])

for dt in range(gatenum):
    zero_gatetensor[dt, :, :] = layer_gatelist[dt].data

bfs.save('.', 'out_memory_gate_data', [zero_gatetensor], ['gate'])

out_layer = evo.out_optimization()
out_layer_numpy = list()

for nt in range(mps_num):                      # Export the target MPS into a numpy array
    out_layer_numpy.append(out_layer[nt].numpy())

np.savez(out_file_name,
         tar_mpslist0=out_layer_numpy[0], tar_mpslist1=out_layer_numpy[1], tar_mpslist2=out_layer_numpy[2],
         tar_mpslist3=out_layer_numpy[3], tar_mpslist4=out_layer_numpy[4], tar_mpslist5=out_layer_numpy[5],
         tar_mpslist6=out_layer_numpy[6], tar_mpslist7=out_layer_numpy[7], tar_mpslist8=out_layer_numpy[8],
         tar_mpslist9=out_layer_numpy[9],
         tar_mpslist10=out_layer_numpy[10], tar_mpslist11=out_layer_numpy[11], tar_mpslist12=out_layer_numpy[12],
         tar_mpslist13=out_layer_numpy[13], tar_mpslist14=out_layer_numpy[14], tar_mpslist15=out_layer_numpy[15],
         tar_mpslist16=out_layer_numpy[16], tar_mpslist17=out_layer_numpy[17], tar_mpslist18=out_layer_numpy[18],
         tar_mpslist19=out_layer_numpy[19],
         tar_mpslist20=out_layer_numpy[20], tar_mpslist21=out_layer_numpy[21], tar_mpslist22=out_layer_numpy[22],
         tar_mpslist23=out_layer_numpy[23], tar_mpslist24=out_layer_numpy[24], tar_mpslist25=out_layer_numpy[25],
         tar_mpslist26=out_layer_numpy[26], tar_mpslist27=out_layer_numpy[27], tar_mpslist28=out_layer_numpy[28],
         tar_mpslist29=out_layer_numpy[29],
         tar_mpslist30=out_layer_numpy[30], tar_mpslist31=out_layer_numpy[31], tar_mpslist32=out_layer_numpy[32],
         tar_mpslist33=out_layer_numpy[33], tar_mpslist34=out_layer_numpy[34], tar_mpslist35=out_layer_numpy[35],
         tar_mpslist36=out_layer_numpy[36], tar_mpslist37=out_layer_numpy[37], tar_mpslist38=out_layer_numpy[38],
         tar_mpslist39=out_layer_numpy[39],
         tar_mpslist40=out_layer_numpy[40], tar_mpslist41=out_layer_numpy[41], tar_mpslist42=out_layer_numpy[42],
         tar_mpslist43=out_layer_numpy[43], tar_mpslist44=out_layer_numpy[44], tar_mpslist45=out_layer_numpy[45],
         tar_mpslist46=out_layer_numpy[46], tar_mpslist47=out_layer_numpy[47])

for nt in range(mps_num):                      # Export the target MPS into a numpy array
    tar_mpslist[nt] = tar_mpslist[nt].cpu().numpy()

end.record()                      # Time spent calculating up to record model

# Waits for everything to finish running
tc.cuda.synchronize()    # Wait for all cores in all streams on the current device to complete.

print('Runtime: ', start.elapsed_time(end))

for i in range(250):
    x1_axis.append(i*10)

color_list = list(['deeppink', 'red', 'gold', 'black', 'lime', 'peru', 'purple', 'blue'])
plt.figure(num=1, figsize=(16, 12), dpi=100)
plt.tick_params(labelsize=16)
plt.xlabel("num of optimize", fontsize=20)           # x轴上的名字
plt.ylabel("negative-logarithmic fidelities (NLFs) per site", fontsize=20)
plt.grid(axis='x', c='g', linestyle='--', alpha=0.5)
for kt in range(layer_num):
    plt.plot(x1_axis, loss_[kt], color=color_list[kt], linewidth=3, label=' Circle layered Optimize' + str(kt))
plt.legend(prop={'family': 'Times New Roman', 'size': 16}, loc='upper right')
plt.savefig('./MPS_Step_1layer_Circle.jpg')


