import BasicFunctions as bf
import numpy as np
import torch as tc
import BasicFunSJR as bfs

file_name = r'.\tar_data.npz'
mps_num = 48
chi = 64
tar_mpslist = list([])

for m in range(mps_num):
    tar_mpslist.append(0)

a = bf.load_pr('../data/XY_Model/chainN48_j(1,0)_h(0,0)_chi64open.pr', 'a')
# a = bf.load_pr('../data/XY_Model/chainN48_j(1,0)_h(0,0)_chi32open.pr', 'a')
print(a)
# print(type(a.mps))
# print(a.center)
# print(a.mps[0].shape)

for nt in range(mps_num):                      # 将目标MPS转存成numpy数组
    tar_mpslist[nt] = tc.from_numpy(a.mps[nt]).to(dtype=tc.float32).numpy()

# np.savez(file_name,
#          tar_mpslist0=tar_mpslist[0], tar_mpslist1=tar_mpslist[1], tar_mpslist2=tar_mpslist[2],
#          tar_mpslist3=tar_mpslist[3], tar_mpslist4=tar_mpslist[4], tar_mpslist5=tar_mpslist[5],
#          tar_mpslist6=tar_mpslist[6], tar_mpslist7=tar_mpslist[7], tar_mpslist8=tar_mpslist[8],
#          tar_mpslist9=tar_mpslist[9], tar_mpslist10=tar_mpslist[10], tar_mpslist11=tar_mpslist[11],
#          tar_mpslist12=tar_mpslist[12], tar_mpslist13=tar_mpslist[13], tar_mpslist14=tar_mpslist[14],
#          tar_mpslist15=tar_mpslist[15], tar_mpslist16=tar_mpslist[16], tar_mpslist17=tar_mpslist[17],
#          tar_mpslist18=tar_mpslist[18], tar_mpslist19=tar_mpslist[19], tar_mpslist20=tar_mpslist[20],
#          tar_mpslist21=tar_mpslist[21], tar_mpslist22=tar_mpslist[22], tar_mpslist23=tar_mpslist[23],
#          tar_mpslist24=tar_mpslist[24], tar_mpslist25=tar_mpslist[25], tar_mpslist26=tar_mpslist[26],
#          tar_mpslist27=tar_mpslist[27], tar_mpslist28=tar_mpslist[28], tar_mpslist29=tar_mpslist[29],
#          tar_mpslist30=tar_mpslist[30], tar_mpslist31=tar_mpslist[31], tar_mpslist32=tar_mpslist[32],
#          tar_mpslist33=tar_mpslist[33], tar_mpslist34=tar_mpslist[34], tar_mpslist35=tar_mpslist[35],
#          tar_mpslist36=tar_mpslist[36], tar_mpslist37=tar_mpslist[37], tar_mpslist38=tar_mpslist[38],
#          tar_mpslist39=tar_mpslist[39],
#          tar_mpslist40=tar_mpslist[40], tar_mpslist41=tar_mpslist[41], tar_mpslist42=tar_mpslist[42],
#          tar_mpslist43=tar_mpslist[43], tar_mpslist44=tar_mpslist[44], tar_mpslist45=tar_mpslist[45],
#          tar_mpslist46=tar_mpslist[46], tar_mpslist47=tar_mpslist[47],
#          tar_mpslist48=tar_mpslist[48], tar_mpslist49=tar_mpslist[49],
#          tar_mpslist50=tar_mpslist[50], tar_mpslist51=tar_mpslist[51], tar_mpslist52=tar_mpslist[52],
#          tar_mpslist53=tar_mpslist[53], tar_mpslist54=tar_mpslist[54], tar_mpslist55=tar_mpslist[55],
#          tar_mpslist56=tar_mpslist[56], tar_mpslist57=tar_mpslist[57], tar_mpslist58=tar_mpslist[58],
#          tar_mpslist59=tar_mpslist[59], tar_mpslist60=tar_mpslist[60], tar_mpslist61=tar_mpslist[61],
#          tar_mpslist62=tar_mpslist[62], tar_mpslist63=tar_mpslist[63], tar_mpslist64=tar_mpslist[64],
#          tar_mpslist65=tar_mpslist[65], tar_mpslist66=tar_mpslist[66], tar_mpslist67=tar_mpslist[67],
#          tar_mpslist68=tar_mpslist[68], tar_mpslist69=tar_mpslist[69], tar_mpslist70=tar_mpslist[70],
#          tar_mpslist71=tar_mpslist[71], tar_mpslist72=tar_mpslist[72], tar_mpslist73=tar_mpslist[73],
#          tar_mpslist74=tar_mpslist[74], tar_mpslist75=tar_mpslist[75], tar_mpslist76=tar_mpslist[76],
#          tar_mpslist77=tar_mpslist[77], tar_mpslist78=tar_mpslist[78], tar_mpslist79=tar_mpslist[79],
#          tar_mpslist80=tar_mpslist[80],
#          tar_mpslist81=tar_mpslist[81], tar_mpslist82=tar_mpslist[82], tar_mpslist83=tar_mpslist[83],
#          tar_mpslist84=tar_mpslist[84], tar_mpslist85=tar_mpslist[85], tar_mpslist86=tar_mpslist[86],
#          tar_mpslist87=tar_mpslist[87], tar_mpslist88=tar_mpslist[88], tar_mpslist89=tar_mpslist[89],
#          tar_mpslist90=tar_mpslist[90], tar_mpslist91=tar_mpslist[91], tar_mpslist92=tar_mpslist[92],
#          tar_mpslist93=tar_mpslist[93], tar_mpslist94=tar_mpslist[94], tar_mpslist95=tar_mpslist[95],
#          tar_mpslist96=tar_mpslist[96], tar_mpslist97=tar_mpslist[97], tar_mpslist98=tar_mpslist[98],
#          tar_mpslist99=tar_mpslist[99])


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

# list1 = list([0, 0, 0])
# a = bfs.save('.', 'jjjjj', [list1], ['tar_gate'])
# b = bfs.load('jjjjj', 'tar_gate')
# print(b)