from BasicFunctions import save_pr, load_pr, print_dict
from DMRGalgorithms import dmrg_infinite_size
from QESclass import QES_1D
from EDspinClass import EDbasic
from Parameters import parameter_qes_gs_by_ed, parameter_qes_ft_by_ed
from HamiltonianModule import hamiltonian_heisenberg
from TensorBasicModule import entanglement_entropy
from scipy.sparse.linalg import LinearOperator as LinearOp
from scipy.sparse.linalg import eigsh as eigs
import os.path as opath
import numpy as np


def prepare_bath_hamilts(para, inputs=None):
    # inputs = (bath, ob0, hamilt)
    print('Starting iDMRG for the entanglement bath')
    bath_data = opath.join(para['bath_path'], para['bath_exp'])
    if inputs is None:
        if para['if_load_bath'] and opath.isfile(bath_data):
            print('Bath data found. Load the bath.')
            bath, ob0, hamilt = load_pr(bath_data, ['A', 'ob0', 'hamilt'])
        else:
            print('Bath data not found. Calculate bath by iDMRG.')
            hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                            para['hx'] / 2, para['hz'] / 2)
            bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt)[:2]
            save_pr(para['bath_path'], para['bath_exp'], [bath, ob0, hamilt], ['A', 'ob0', 'hamilt'])
    else:
        bath, ob0, hamilt = inputs
    if (bath.is_symme_env is True) and (bath.dmrg_type is 'mpo'):
        bath.env[1] = bath.env[0]

    print('Preparing the physical-bath Hamiltonians')
    qes = QES_1D(para['d'], para['chi'], para['d'] * para['d'],
                 para['l_phys'], para['tau'], spin=para['spin'])
    if bath.dmrg_type is 'mpo':
        qes.obtain_physical_gate_tensors(hamilt)
        qes.obtain_bath_h(bath.env, 'both')
    else:
        qes.obtain_bath_h_by_effective_ops_1d(
            bath.bath_op_onsite, bath.effective_ops, bath.hamilt_index)
    hamilts = [hamilt] + qes.hamilt_bath
    return hamilts, bath, ob0


def find_degenerate_ground_state(para, it_time, tol=1e-2):
    # if not para['is_symme_env']:
    #     para['is_symme_env'] = True
    #     print('In \'find_degenerate_bath\', set para[\'is_symme_env\'] = True')

    dege_states = list()
    for t in range(it_time):
        # Randomly initialize env
        env = list()
        env.append(np.random.randn(para['chi'], para['d']**para['n_site'], para['chi']))
        env[0] = env[0] + env[0].transpose(2, 1, 0)
        env[0] /= np.linalg.norm(env[0])
        env.append(env[0].copy())
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx'] / 2, para['hz'] / 2)
        bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt, env=env)[:2]
        gs = bath.mps[1]
        if len(dege_states) > 0:
            delta = list()
            add_new = True
            for n in range(len(dege_states)):
                delta.append(np.sqrt(np.abs(2-2*np.abs(
                    dege_states[n].reshape(1, -1).dot(gs.reshape(-1, 1))[0, 0]))))
                add_new = add_new and (delta[-1] > tol)
            print('Differences = ' + str(delta))
            if add_new:
                dege_states.append(gs)
                print(str(len(dege_states)) + ' envs have been found.')
        else:
            dege_states.append(gs)
    print('After ' + str(it_time) + ' iterations, ' + str(len(dege_states)) + ' have been found.')


def find_degenerate_rho(para, it_time, tol=1e-2):
    dege_rho = list()
    for t in range(it_time):
        # Randomly initialize env
        env = list()
        env.append(np.random.randn(para['chi'], para['d']**para['n_site'], para['chi']))
        env[0] = env[0] + env[0].transpose(2, 1, 0)
        env[0] /= np.linalg.norm(env[0])
        env.append(env[0].copy())
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx'] / 2, para['hz'] / 2)
        bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt, env=env)[:2]
        bath.rho_from_central_tensor()
        rho = bath.rho
        if len(dege_rho) > 0:
            delta = list()
            for n in range(len(dege_rho)):
                delta.append(np.sqrt(np.abs(2-2*np.abs(
                    dege_rho[n].reshape(1, -1).dot(rho.reshape(-1, 1))[0, 0]))))
                # delta.append(np.abs(np.trace(dege_rho[n].dot(rho))))
            print('Differences = ' + str(delta))
            if np.min(delta) > tol:
                dege_rho.append(rho)
                print(str(len(dege_rho)) + ' have been found.')
        else:
            dege_rho.append(rho)
    print('After ' + str(it_time) + ' iterations, ' + str(len(dege_rho)) + ' have been found.')


def find_degenerate_hbaths(para, it_time, tol=1e-2):
    hbaths = list()
    for t in range(it_time):
        # Randomly initialize env
        env = list()
        env.append(np.random.randn(para['chi'], para['d'] ** para['n_site'], para['chi']))
        env[0] = env[0] + env[0].transpose(2, 1, 0)
        env[0] /= np.linalg.norm(env[0])
        env.append(env[0].copy())
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx'] / 2, para['hz'] / 2)
        bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt, env=env)[:2]

        para_qes = parameter_qes_gs_by_ed(para)
        qes_hamilt = prepare_bath_hamilts(para_qes, (bath, ob0, hamilt))[0]
        qes_hamilt = qes_hamilt[1]
        # print(np.trace(qes_hamilt), qes_hamilt.shape)

        # find degenerate hbaths
        if len(hbaths) > 0:
            delta = list()
            add_new = True
            for n in range(len(hbaths)):
                delta1 = (hbaths[n]/np.linalg.norm(hbaths[n])).reshape(1, -1).dot(
                    (qes_hamilt/np.linalg.norm(qes_hamilt)).reshape(-1, 1))[0, 0]
                delta.append(np.sqrt(np.abs(2 - 2*delta1)))
                add_new = add_new and (delta[-1] > tol)
            print('Differences = ' + str(delta))
            if add_new:
                hbaths.append(qes_hamilt)
                print(str(len(hbaths)) + ' envs have been found.')
        else:
            hbaths.append(qes_hamilt)
    print('After ' + str(it_time) + ' iterations, ' + str(len(hbaths)) + ' have been found.')


def find_degenerate_rings(para, it_time, tol=1e-2):
    # if not para['is_symme_env']:
    #     para['is_symme_env'] = True
    #     print('In \'find_degenerate_bath\', set para[\'is_symme_env\'] = True')

    rings = list()
    for t in range(it_time):
        # Randomly initialize env
        env = list()
        env.append(np.random.randn(para['chi'], para['d']**para['n_site'], para['chi']))
        env[0] = env[0] + env[0].transpose(2, 1, 0)
        env[0] /= np.linalg.norm(env[0])
        env.append(env[0].copy())
        hamilt = hamiltonian_heisenberg(para['spin'], para['jxy'], para['jxy'], para['jz'],
                                        para['hx'] / 2, para['hz'] / 2)
        bath, ob0 = dmrg_infinite_size(para, hamilt=hamilt, env=env)[:2]
        bath.env[1] = bath.env[0]

        # find degenerate ring tensors
        rt = bath.obtain_ring_tensor()
        rt = np.real(rt)
        rt /= np.linalg.norm(rt)
        if len(rings) > 0:
            delta = list()
            add_ring = True
            for n in range(len(rings)):
                delta.append(np.sqrt(np.abs(2-2*np.abs(
                    rings[n].reshape(1, -1).dot(rt.reshape(-1, 1))[0, 0]))))
                add_ring = add_ring and (delta[-1] > tol)
            print('Differences = ' + str(delta))
            if add_ring:
                rings.append(rt)
                print(str(len(rings)) + ' envs have been found.')
        else:
            rings.append(rt)
    print('After ' + str(it_time) + ' iterations, ' + str(len(rings)) + ' have been found.')


def qes_gs_1d_ed(para=None):
    if para is None:
        para = parameter_qes_ft_by_ed()
    hamilts, bath, ob0 = prepare_bath_hamilts(para)
    print('Starting ED for the entanglement bath')
    dims = [para['d'] for _ in range(para['l_phys'])]
    dims = [para['chi']] + dims + [para['chi']]
    ob = dict()
    solver = EDbasic(dims, spin=para['spin'])
    heff = LinearOp((solver.dim_tot, solver.dim_tot),
                    lambda x: solver.project_all_hamilt(
                        x, hamilts, para['tau'], para['couplings']))
    ob['e_eig'], solver.v = eigs(heff, k=1, which='LM', v0=solver.v.reshape(-1, ).copy())
    solver.is_vec = True
    ob['e_eig'] = (1 - ob['e_eig']) / para['tau']
    ob['mx'], ob['mz'] = solver.observe_magnetizations(para['phys_sites'])
    ob['eb'] = solver.observe_bond_energies(hamilts[0], para['positions_h2'][1:para['num_h2']-1, :])
    ob['lm'] = solver.calculate_entanglement()
    ob['ent'] = entanglement_entropy(ob['lm'])
    ob['e_site'] = sum(ob['eb']) / (para['l_phys'] - 1)
    ob['corr_xx'] = solver.observe_correlations(para['pos4corr'], para['op'][1])
    ob['corr_zz'] = solver.observe_correlations(para['pos4corr'], para['op'][3])
    for n in range(para['pos4corr'].shape[0]):
        p1 = para['pos4corr'][n, 0] - 1
        p2 = para['pos4corr'][n, 1] - 1
        ob['corr_xx'][n] -= ob['mx'][p1] * ob['mx'][p2]
        ob['corr_zz'][n] -= ob['mz'][p1] * ob['mz'][p2]
    return bath, solver, ob0, ob


def qes_ft_1d_ltrg(para=None):
    if para is None:
        para = parameter_qes_gs_by_ed()
    hamilts, bath, ob0 = prepare_bath_hamilts(para)
    print('Starting ED for the entanglement bath')
    dims = [para['d'] for _ in range(para['l_phys'])]
    dims = [para['chi']] + dims + [para['chi']]
    ob = dict()
    solver = EDbasic(dims)
    heff = LinearOp((solver.dim_tot, solver.dim_tot),
                    lambda x: solver.project_all_hamilt(
                        x, hamilts, para['tau'], para['couplings']))
    ob['e_eig'], solver.v = eigs(heff, k=1, which='LM', v0=solver.v.reshape(-1, ).copy())
    solver.is_vec = True
    ob['e_eig'] = (1 - ob['e_eig']) / para['tau']
    ob['mx'], ob['mz'] = solver.observe_magnetizations(para['phys_sites'])
    ob['eb'] = solver.observe_bond_energies(hamilts[0], para['positions_h2'][1:para['num_h2']-1, :])
    ob['lm'] = solver.calculate_entanglement()
    ob['ent'] = entanglement_entropy(ob['lm'])
    ob['e_site'] = sum(ob['eb']) / (para['l_phys'] - 1)
    ob['corr_xx'] = solver.observe_correlations(para['pos4corr'], para['op'][1])
    ob['corr_zz'] = solver.observe_correlations(para['pos4corr'], para['op'][3])
    for n in range(para['pos4corr'].shape[0]):
        p1 = para['pos4corr'][n, 0] - 1
        p2 = para['pos4corr'][n, 1] - 1
        ob['corr_xx'][n] -= ob['mx'][p1] * ob['mx'][p2]
        ob['corr_zz'][n] -= ob['mz'][p1] * ob['mz'][p2]
    return bath, solver, ob0, ob