from DMRGalgorithms import dmrg_finite_size
import Parameters as Pm
import numpy as np
from os import path
from BasicFunctions import load_pr, save_pr, plot, output_txt
from TensorBasicModule import open_mps_product_state_spin_up


length = list(range(3, 30))

lattice = 'chain'
para_dmrg = Pm.generate_parameters_dmrg(lattice)
para_dmrg['spin'] = 'half'
para_dmrg['bound_cond'] = 'periodic'
para_dmrg['chi'] = 128
para_dmrg['l'] = 12
para_dmrg['jxy'] = 1
para_dmrg['jz'] = 1
para_dmrg['hx'] = 0
para_dmrg['hz'] = 0
para_dmrg['project_path'] = '.'
para_dmrg['data_path'] = 'data/HeisenbergChain'

for n in range(len(length)):
    para_dmrg['l'] = length[n]
    para_dmrg = Pm.make_consistent_parameter_dmrg(para_dmrg)
    ob, a, info, para = dmrg_finite_size(para_dmrg)
    print('Energy per site = ' + str(ob['e_per_site']))
    save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
            ('ob', 'a', 'info', 'para'))

# if path.isfile(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr')):
#     print('Load existing data ...')
#     a, ob = load_pr(path.join(para_dmrg['data_path'], para_dmrg['data_exp'] + '.pr'), ['a', 'ob'])
# else:
#     ob, a, info, para = dmrg_finite_size(para_dmrg)
#     save_pr(para['data_path'], para['data_exp'] + '.pr', (ob, a, info, para),
#             ('ob', 'a', 'info', 'para'))
