import TensorBasicModule as T_module
import numpy as np
from ipdb import set_trace
from scipy.sparse.linalg import eigsh as eigs
from scipy.sparse.linalg import LinearOperator as LinearOp
from scipy.linalg import expm, logm
from HamiltonianModule import spin_operators
from BasicFunctions import empty_list, trace_stack, sort_list, print_error, print_sep, \
    print_options, print_dict, info_contact
from termcolor import colored, cprint
import copy


class MpsBasic:

    def __init__(self):
        self.operators = list()
        self.tmp = None  # This is to exchange data within the class

    def append_operators(self, op_new):
        if type(op_new) is np.ndarray:
            self.operators.append(op_new)
        else:
            for n in range(0, len(op_new)):
                self.operators.append(op_new[n])


class MpsOpenBoundaryClass(MpsBasic):

    def __init__(self, length, d, chi, spin='half', way='qr', ini_way='r', operators=None,
                 debug=False, is_parallel=False, is_save_op=False, eig_way=0, par_pool=None,
                 is_env_parallel_lmr=True, is_eco_dims=True):
        MpsBasic.__init__(self)
        self.spin = spin
        self.phys_dim = d
        self.decomp_way = way  # 'svd' or 'qr'
        self.length = length
        # self.orthogonality:  -1: left2right; 0: not orthogonal or center; 1: right2left
        self.orthogonality = np.zeros((length, 1))
        self.center = -1  # orthogonal center; -1 means no center
        self.lm = [np.zeros(0) for _ in range(0, length-1)]
        self.ent = np.zeros((self.length-1, 1))
        if ini_way == 'r':  # randomly initialize MPS
            self.mps, self.virtual_dim = \
                T_module.random_open_mps(length, d, chi, is_eco=is_eco_dims)
        elif ini_way == '1':  # initialize MPS as eyes
            self.mps, self.virtual_dim = T_module.ones_open_mps(
                length, d, chi, is_eco=is_eco_dims)
        if operators is None:
            op = spin_operators(spin)
            self.operators = [op['id'], op['sx'], op['sy'], op['sz'], op['su'],
                              op['sd']]
        else:
            self.operators = operators

        self._is_save_op = is_save_op  # whether saving all effective operators to accelerate the code
        self.effect_s = {'none': np.zeros(0)}
        self.pos_effect_s = np.zeros((0, 3)).astype(int)
        self.effect_ss = {'none': np.zeros(0)}
        self.pos_effect_ss = np.zeros((0, 5)).astype(int)
        self.effective_id = {'none': np.zeros(0)}
        self.opt_env = dict()

        self._is_parallel = is_parallel
        self.pool = par_pool
        self._debug = debug  # if in debug mode
        self.eig_way = eig_way
        self._is_env_parallel_lmr = is_env_parallel_lmr

        if self._is_parallel and self._is_save_op:
            cprint('Note: this version forbids to use parallel computing while in the is_save_op mode', 'cyan')
            cprint('The is_save_op mode has been automatically switched off', 'magenta')
            cprint('This issue will be fixed in the next version', 'cyan')
            self._is_save_op = False
        if debug:
            cprint('Note: you are in the debug mode', 'cyan')
        if self._is_save_op:
            cprint('Note: you are in the is_save_op mode. The code will save intermediate results to accelerate '
                   'the computation', 'cyan')
        if self._is_parallel:
            cprint('Note: you are using parallel computing. The parallel computing will be used when '
                   'computing the environment for different different coupling terms', 'cyan')

    def input_mps(self, mps, attributes=None, if_deepcopy=True):
        if type(mps) is list:
            data = {'mps': mps}
        else:
            data = mps.wrap_data(attributes, if_deepcopy=if_deepcopy)
        self.refresh_mps_properties(data, if_deepcopy=if_deepcopy)

    def wrap_data(self, attributes=None, if_deepcopy=True):
        if attributes is None:
            attributes = ['mps', 'orthogonality', 'center', 'virtual_dim']
        data = dict()
        for a in attributes:
            exec('data[\'' + a + '\'] = self.' + a)
        if if_deepcopy:
            return copy.deepcopy(data)
        else:
            return data

    def refresh_mps_properties(self, data=None, if_deepcopy=True):
        if data is None:
            data = dict()
        for a in data:
            if if_deepcopy:
                exec('self.' + a + '=copy.deepcopy(data[\'' + a + '\'] )')
            else:
                exec('self.' + a + '=data[\'' + a + '\']')
        self.length = self.mps.__len__()
        if 'orthogonality' not in data:
            self.orthogonality = np.zeros((self.length, 1))
        if 'center' not in data:
            self.center = -1  # orthogonal center; -1 means no center
        if 'lm' not in data:
            self.lm = [np.zeros(0) for _ in range(0, self.length - 1)]
        if 'ent' not in data:
            self.ent = np.zeros((self.length - 1, 1))
        if 'virtual_dim' not in data:
            self.virtual_dim = empty_list(self.length, content=1)
            for n in range(self.length):
                self.virtual_dim[n] = self.mps[n].shape[0]
        self.clean_to_save()

    def orthogonalize_mps(self, l0, l1, normalize=False, is_trun=False, chi=-1):
        """
        Orthogonalization of the MPS from l0-th to l1-th tensors. Note that from l0-th to (l1-1)-th, the tensor will
        be left-to-right orthogonal (l0<l1) or right-to-left orthogonal (l0>l1)
        :param l0: starting pointing of the orthogonalization
        :param l1: ending pointing of the orthogonalization
        :param normalize: whether to normalize the matrix
        :param is_trun: whether to truncate the dimension
        :param chi: dimension cut-off
        """
        if is_trun:
            decomp_way = 'svd'  # if truncation is needed, it is mandatory to use 'svd'
        else:
            decomp_way = self.decomp_way
        if l0 < l1:  # Orthogonalize MPS from left to right
            for n in range(l0, l1):
                self.mps[n], mat, self.virtual_dim[n+1], lm = \
                    T_module.left2right_decompose_tensor(self.mps[n], decomp_way)
                # if normalize:
                #     if decomp_way is 'svd':
                #         mat /= np.linalg.norm(lm)
                #     else:
                #         mat /= max(abs(mat.reshape(-1, )))
                if is_trun and (mat.shape[1] > chi):
                    self.mps[n + 1] = T_module.absorb_matrix2tensor(
                        self.mps[n + 1], mat[:, :chi], 0)
                    self.mps[n] = self.mps[n][:, :, :chi]
                    if lm.size > 0 and self.center > -1:
                        self.lm[n] = lm[:chi]
                    self.virtual_dim[n + 1] = chi
                else:
                    self.mps[n+1] = T_module.absorb_matrix2tensor(self.mps[n+1], mat, 0)
                    if lm.size > 0 and self.center > -1:
                        self.lm[n] = lm.copy()
                if normalize:
                    self.mps[n + 1] /= np.linalg.norm(self.mps[n+1])
            self.orthogonality[l0:l1] = -1
            self.orthogonality[l1] = 0
        elif l0 > l1:  # Orthogonalize MPS from right to left
            for n in range(l0, l1, -1):
                self.mps[n], mat, self.virtual_dim[n], lm =\
                    T_module.right2left_decompose_tensor(self.mps[n], decomp_way)
                # if normalize:
                #     if decomp_way is 'svd':
                #         mat /= np.linalg.norm(lm)
                #     else:
                #         mat /= max(abs(mat.reshape(-1, )))
                if is_trun and (mat.shape[1] > chi):
                    self.mps[n - 1] = T_module.absorb_matrix2tensor(
                        self.mps[n - 1], mat[:, :chi], 2)
                    self.mps[n] = self.mps[n][:chi, :, :]
                    if lm.size > 0 and self.center > -1:
                        self.lm[n - 1] = lm[:chi]
                    self.virtual_dim[n] = chi
                else:
                    self.mps[n-1] = T_module.absorb_matrix2tensor(self.mps[n - 1], mat, 2)
                    if lm.size > 0 and self.center > -1:
                        self.lm[n - 1] = lm.copy()
                if normalize:
                    self.mps[n - 1] /= np.linalg.norm(self.mps[n - 1])
            self.orthogonality[l0:l1:-1] = 1
            self.orthogonality[l1] = 0

    # transfer the MPS into the central orthogonal form with the center lc
    def central_orthogonalization(self, lc, l0=0, l1=-1, normalize=False):
        """
        Transform the MPS into central orthogonal form, with lc the center. Note you can specify the starting
        and ending point of the orthogonalization process
        :param lc: new center
        :param l0: starting point (0 as default)
        :param l1: ending point (self.length-1 as default)
        :param normalize: whether to normalize the matrix
        Warning: * if the MPS is not central orthogonal, this function may not transform it into a orthogonal form
                 * To central-orthogonalize the MPS or change the center, recommend to use correct_orthogonal_center
        Example:
        """
        if l1 == -1:
            l1 = self.length-1
        self.orthogonalize_mps(l0, lc, normalize=normalize)
        self.orthogonalize_mps(l1, lc, normalize=normalize)
        self.center = lc

    # move the orthogonal center at p
    def correct_orthogonal_center(self, p=-1, normalize=False):
        # if p<0 (default) and there is no center, automatically find a new center
        if p < -0.5 and self.center < -0.5:
            p = self.check_orthogonal_center(if_print=False)
        elif p < -0.5:
            p = self.center
        if self.center < -0.5:
            self.central_orthogonalization(p, normalize=normalize)
        elif self.center != p:
            self.orthogonalize_mps(self.center, p, normalize=normalize)
        self.center = p

# ===========================================================
# For handling effective operators in the fast mode
    @ staticmethod
    def key_effective_operators(info):
        # generate the key of one-body or two-body effective operator
        # info = (sn, ssn, p0, q0, p1)
        x = ''
        for n in range(0, info.__len__() - 1):
            x += str(info[n]) + '_'
        x += str(info[-1])
        return x

    @ staticmethod
    def key_restore_info(key):
        return key.split('_')

    def add_key_and_pos(self, which_op, key_info, op):
        key = self.key_effective_operators(key_info)
        if which_op == 'one':
            if key not in self.effect_s:
                self.pos_effect_s = np.vstack((self.pos_effect_s, np.array(key_info)))
            self.effect_s[key] = op
        elif which_op == 'two':
            if key not in self.effect_ss:
                self.pos_effect_ss = np.vstack((self.pos_effect_ss, np.array(key_info)))
            self.effect_ss[key] = op

    def find_nearest_key_one_body(self, sn, p0, p1):
        pos = self.pos_effect_s[self.pos_effect_s[:, 0] == sn, :]
        pos = pos[pos[:, 1] == p0, :]
        if p0 < p1:  # RG flow: left to right
            pos = pos[pos[:, 2] < p1, :]
            pos = pos[pos[:, 2] > p0, :]
            if pos.size == 0:
                p_before = None
                key_info = (sn, p0, p0+1)
            else:
                n = np.argmax(pos[:, 2])
                p_before = pos[n, 2]
                key_info = tuple(pos[n, :])
        else:
            pos = pos[pos[:, 2] > p1, :]
            pos = pos[pos[:, 2] <= p0, :]
            if pos.size == 0:
                p_before = None
                key_info = (sn, p0, p0)
            else:
                n = np.argmin(pos[:, 2])
                p_before = pos[n, 2]
                key_info = tuple(pos[n, :])
        return key_info, p_before

    def get_effective_operators_one_body(self, sn, p0, p1, is_update_op=True):
        # sn: which operator
        # p0: original position of the operator (site)
        # p1: position of the target effective operator (bond)
        key = self.key_effective_operators((sn, p0, p1))
        if key in self.effect_s:
            return self.effect_s[key]
        else:
            key_info, p_before = self.find_nearest_key_one_body(sn, p0, p1)
            if p_before is None:
                if p0 < p1:
                    if self.center < p0:
                        v = self.effective_id[str(self.center) + '_' + str(p0)]
                        v = T_module.bound_vec_operator_left2right(self.mps[p0], self.operators[sn], v=v)
                    else:
                        v = T_module.bound_vec_operator_left2right(self.mps[p0], self.operators[sn])
                    if is_update_op:
                        self.add_key_and_pos('one', key_info, v)
                    v = self.update_effect_op_l0_to_l1(p0+1, p1, v, sn, p0, is_update_op=is_update_op)
                else:
                    if self.center > p0:
                        v = self.effective_id[str(self.center) + '_' + str(p0+1)]
                        v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn], v=v)
                    else:
                        v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn])
                    if is_update_op:
                        self.add_key_and_pos('one', key_info, v)
                    v = self.update_effect_op_l0_to_l1(p0-1, p1-1, v, sn, p0, is_update_op=is_update_op)
            else:
                key_before = self.key_effective_operators(key_info)
                if p0 < p1:
                    v = self.update_effect_op_l0_to_l1(p_before, p1, self.effect_s[key_before],
                                                       sn, p0, is_update_op=is_update_op)
                else:
                    v = self.update_effect_op_l0_to_l1(p_before-1, p1-1, self.effect_s[key_before],
                                                       sn, p0, is_update_op=is_update_op)
            return v

    def get_effective_operator_two_body(self, sn, snn, p0, q0, p1, is_update_op=True):
        # the self.operators[sn]  is originally at p0-th site
        # the self.operators[ssn] is originally at q0-th site
        # the effective two-body operator is at the p1-th bond
        # here, we have p0 < q0 <= p1, or p1 >= q0 > p0 (on the same side of the RG endpoint)
        if p0 > q0:  # make sure p0 < q0
            p0, q0 = q0, p0
            sn, snn = snn, sn
        key2 = self.key_effective_operators((sn, snn, p0, q0, p1))
        if key2 in self.effect_ss:
            return self.effect_ss[key2]
        elif q0 == p1:
            print_error('LogicBug detected: please check')
        else:
            return self.update_effect_from_op1_to_op2(sn, snn, p0, q0, p1, is_update_op=is_update_op)

    def del_bad_effective_operators(self, p):
        # delete the badly defined effective operators due to the change of the p-th tensor
        if self.pos_effect_s.shape[0] > 0:
            ind = (self.pos_effect_s[:, 1] < p) * (self.pos_effect_s[:, 2] <= p)
            ind += (self.pos_effect_s[:, 1] > p) * (self.pos_effect_s[:, 2] > p)
            ind_del = (~ ind)
            pos_del = self.pos_effect_s[ind_del, :]
            for n in range(0, pos_del.shape[0]):
                key = self.key_effective_operators(tuple(pos_del[n, :]))
                self.effect_s.__delitem__(key)
            self.pos_effect_s = self.pos_effect_s[ind, :]
        if self.pos_effect_ss.shape[0] > 0:
            ind = (self.pos_effect_ss[:, 3] < p) * (self.pos_effect_ss[:, 4] <= p)
            ind += (self.pos_effect_ss[:, 2] > p) * (self.pos_effect_ss[:, 4] > p)
            ind_del = (~ ind)
            pos_del = self.pos_effect_ss[ind_del, :]
            for n in range(0, pos_del.shape[0]):
                key = self.key_effective_operators(tuple(pos_del[n, :]))
                self.effect_ss.__delitem__(key)
            self.pos_effect_ss = self.pos_effect_ss[ind, :]

# ===========================================================================
# DMRG related functions
    @ staticmethod
    def calculate_environment_for_parallel(results, dim):
        x = np.zeros((dim, dim))
        for n in range(0, len(results)):
            x += np.kron(np.kron(results[n][0], results[n][1]), results[n][2])
        return x

    def environment_s1_s2(self, inputs):
        # p is the center and the position of the tensor to be updated
        # the two operators are at positions[0] and positions[1]
        p, sn, positions = inputs
        if self._debug:
            self.check_orthogonal_center(p)
        operators = [self.operators[sn[0]], self.operators[sn[1]]]
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if positions[0] > positions[1]:
            positions = sort_list(positions, [1, 0])
            operators = sort_list(operators, [1, 0])
        if p < positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = np.eye(self.mps[p].shape[1])
            if self._is_save_op:
                v_right = self.get_effective_operator_two_body(sn[0], sn[1], positions[0],
                                                               positions[1], p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1]-1, positions[0], v_right)
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[0]], operators[0], v_right)
                v_right = self.contract_v_l0_to_l1(positions[0] - 1, p, v_right)
        elif p > positions[1]:
            if self._is_save_op:
                v_left = self.get_effective_operator_two_body(sn[0], sn[1], positions[0],
                                                              positions[1], p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0]+1, positions[1], v_left)
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v_left)
                v_left = self.contract_v_l0_to_l1(positions[1] + 1, p, v_left)
            v_middle = np.eye(self.mps[p].shape[1])
            v_right = np.eye(self.virtual_dim[p + 1])
        elif p == positions[0]:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = operators[0]
            if self._is_save_op:
                v_right = self.get_effective_operators_one_body(sn[1], positions[1], p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
        elif p == positions[1]:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn[0], positions[0], p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operators[1]
        else:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn[0], positions[0], p)
                v_right = self.get_effective_operators_one_body(sn[1], positions[1], p+1)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v_left)
                v_left = self.contract_v_l0_to_l1(positions[0] + 1, p, v_left)
                v_right = T_module.bound_vec_operator_right2left(self.mps[positions[1]], operators[1], v_right)
                v_right = self.contract_v_l0_to_l1(positions[1] - 1, p, v_right)
            v_middle = np.eye(self.mps[p].shape[1])
        # if self.virtual_dim[p] != v_left.shape[0] or self.virtual_dim[p+1] != v_right.shape[0]:
        #     print('Wrong dimension: ' + str((sn[0], sn[1], positions[0], positions[1], p+1)))
        return v_left, v_middle, v_right

    def environment_s1_parallel(self, inputs):
        env2 = 0
        for n in range(0, inputs.__len__()):
            v_left, v_middle, v_right = self.environment_s1(inputs[n][:3])
            env2 += inputs[n][3] * np.kron(np.kron(v_left, v_middle), v_right)
        return env2

    # calculate the environment (one-body terms)
    def environment_s1(self, inputs):
        # p is the position of the tensor to be updated
        # the operator[sn] is at position
        p, sn, position = inputs
        if self._debug:
            self.check_orthogonal_center(p)
            self.check_virtual_bond_dimensions()

        operator = self.operators[sn]
        v_left = np.zeros(0)
        v_right = np.zeros(0)
        if p < position:
            v_left = np.eye(self.virtual_dim[p])
            v_middle = np.eye(self.mps[p].shape[1])
            if self._is_save_op:
                v_right = self.get_effective_operators_one_body(sn, position, p+1)
            else:
                v_right = T_module.bound_vec_operator_right2left(self.mps[position],
                                                                 operator, v_right)
                v_right = self.contract_v_l0_to_l1(position - 1, p, v_right)
        elif p > position:
            if self._is_save_op:
                v_left = self.get_effective_operators_one_body(sn, position, p)
            else:
                v_left = T_module.bound_vec_operator_left2right(self.mps[position],
                                                                operator, v_left)
                v_left = self.contract_v_l0_to_l1(position + 1, p, v_left)
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = np.eye(self.mps[p].shape[1])
        else:  # p == position
            v_left = np.eye(self.virtual_dim[p])
            v_right = np.eye(self.virtual_dim[p + 1])
            v_middle = operator
        return v_left, v_middle, v_right

    def environment_s1_s2_parallel(self, inputs):
        env2 = 0
        for n in range(0, inputs.__len__()):
            v_left, v_middle, v_right = self.environment_s1_s2(inputs[n])
            env2 += np.kron(np.kron(v_left, v_middle), v_right)
        return env2

    # update the boundary vector v by contracting from l0 to l1 without operators
    def contract_v_l0_to_l1(self, l0, l1, v=np.zeros(0)):
        if l0 < l1:
            for n in range(l0, l1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
        elif l0 > l1:
            for n in range(l0, l1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
        return v

    def contract_v_with_phys_l0_to_l1(self, l0, lp, l1, v=np.zeros(0)):
        # Note: lp is the position of remained physical bond, and l0<lp<l1, or l0>lp>l1
        if l0 < l1:  # left to right
            for n in range(l0, lp):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
            for n in range(lp, l1):
                v = T_module.bound_vec_with_phys_left2right(self.mps[n], v)
        elif l0 > l1:
            for n in range(l0, lp, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
            for n in range(lp, l1, -1):
                v = T_module.bound_vec_with_phys_right2left(self.mps[n], v)
        return v

    def update_effect_op_l0_to_l1(self, l0, l1, v, sn=-1, pos0=-1, is_update_op=True):
        # l0: starting site
        # l1: before the ending site
        # sn is the number of the operator
        # pos0 is the original position of v (effective operator)
        if l0 < l1:
            for n in range(l0, l1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
                if is_update_op:
                    self.add_key_and_pos('one', (sn, pos0, n+1), v)
        elif l0 > l1:
            for n in range(l0, l1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
                if is_update_op:
                    self.add_key_and_pos('one', (sn, pos0, n), v)
        return v

    def update_all_effective_id(self):
        self.effective_id = dict()
        v = np.eye(self.virtual_dim[self.center])
        for n in range(self.center, self.length):
            key = str(self.center) + '_' + str(n+1)
            v = T_module.bound_vec_operator_left2right(self.mps[n], v=v)
            self.effective_id[key] = v
        v = np.eye(self.virtual_dim[self.center+1])
        for n in range(self.center, 0, -1):
            key = str(self.center) + '_' + str(n)
            v = T_module.bound_vec_operator_right2left(self.mps[n], v=v)
            self.effective_id[key] = v

    def update_effect_from_op1_to_op2(self, sn, snn, p0, q0, p1, is_update_op=True):
        # here, we have p0 < q0 < p1, or p1 >= q0 > p0 (on the same side of the RG endpoint)
        if q0 < p1:  # left to right
            v = self.get_effective_operators_one_body(sn, p0, q0)
            v = T_module.bound_vec_operator_left2right(self.mps[q0], self.operators[snn], v)
            if is_update_op:
                self.add_key_and_pos('two', (sn, snn, p0, q0, q0+1), v)
            for n in range(q0+1, p1):
                v = T_module.bound_vec_operator_left2right(tensor=self.mps[n], v=v)
                # v = self.update_effect_op_l0_to_l1(n, n+1, v, is_update_op=False)
                if is_update_op:
                    self.add_key_and_pos('two', (sn, snn, p0, q0, n+1), v)
        elif p1 <= p0:
            v = self.get_effective_operators_one_body(snn, q0, p0+1)
            v = T_module.bound_vec_operator_right2left(self.mps[p0], self.operators[sn], v)
            if is_update_op:
                self.add_key_and_pos('two', (sn, snn, p0, q0, p0), v)
            for n in range(p0-1, p1-1, -1):
                v = T_module.bound_vec_operator_right2left(tensor=self.mps[n], v=v)
                # v = self.update_effect_op_l0_to_l1(n, n - 1, v, is_update_op=False)
                if is_update_op:
                    self.add_key_and_pos('two', (sn, snn, p0, q0, n), v)
        else:
            # if this happen, there must be a logic bug
            v = None
            print_error('LogicBug detected. Please check')
        return v

    def effective_hamiltonian_dmrg(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]  # number of one-body Hamiltonians
        nh2 = index2.shape[0]  # number of two-body Hamiltonians
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        dim = np.prod(s)
        if not self._is_parallel:
            h_effect = np.zeros((dim, dim))
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if (abs(coeff1[n]) > tol) and \
                        (np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol):
                    v_left, v_middle, v_right = \
                        self.environment_s1((p, index1[n, 1], index1[n, 0]))
                    if self._debug:
                        self.check_environments(v_left, v_middle, v_right, p)
                    h_effect += coeff1[n] * np.kron(np.kron(v_left, v_middle), v_right)
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    v_left, v_middle, v_right = \
                        self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2]))
                    if self._debug:
                        self.check_environments(v_left, v_middle, v_right, p)
                    h_effect += coeff2[n] * np.kron(np.kron(v_left, v_middle), v_right)
        else:  # parallel computations
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0], coeff1[n]))
                    n_now += 1
            tmp = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            h_effect = 0
            for n in range(0, tmp.__len__()):
                h_effect += tmp[n]
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            tmp = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            for n in range(0, tmp.__len__()):
                h_effect += tmp[n]
        # h_effect = (h_effect + h_effect.conj().T) / 2
        return h_effect, s

    def all_environments(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        # for 'update_tensor_eigs' while mapping the effective H to a linear operator
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]
        nh2 = index2.shape[0]
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        if not self._is_parallel and not self._is_env_parallel_lmr:
            # No parallel computing
            env1 = list()
            env2 = list()
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    env1.append(self.environment_s1((p, index1[n, 1], index1[n, 0])))
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    env2.append(self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2])))
        elif (not self._is_parallel) and self._is_env_parallel_lmr:
                inputs = list()
                for n in range(0, nh1):
                    if abs(coeff1[n]) > tol and np.linalg.norm(
                            self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                        inputs.append((p, index1[n, 1], index1[n, 0]))
                env1 = self.pool['pool'].map(self.environment_s1, inputs)
                inputs = list()
                for n in range(0, nh2):
                    if abs(coeff2[n]) > tol:
                        inputs.append((p, index2[n, 2:4], index2[n, :2]))
                env2 = self.pool['pool'].map(self.environment_s1_s2, inputs)
        else:
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0]))
                    n_now += 1
            env1 = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            env2 = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            self.pool['pool'].join()
        return env1, env2, s

    def all_environments_optimized(self, p, index1, index2, coeff1, coeff2, tol=1e-12):
        # for 'update_tensor_eigs' while mapping the effective H to a linear operator
        if self._debug and p != self.center:
            print_error('CenterError: the tensor must be at the orthogonal center before '
                        'defining the function handle', 'magenta')
        nh1 = index1.shape[0]
        nh2 = index2.shape[0]
        s = [self.virtual_dim[p], self.phys_dim, self.virtual_dim[p+1]]
        if not self._is_parallel and not self._is_env_parallel_lmr:
            # No parallel computing
            for n in range(0, nh1):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    env1 = self.environment_s1((p, index1[n, 1], index1[n, 0]))
                    self.classify_and_update_env(env1, coeff1[n], 'one', index1[n, 0], index1[n, 1], p)
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    env2 = self.environment_s1_s2((p, index2[n, 2:4], index2[n, :2]))
                    self.classify_and_update_env(env2, coeff2[n], 'two', index2[n, :2], index2[n, 2:4], p)
        elif (not self._is_parallel) and self._is_env_parallel_lmr:
                inputs = list()
                for n in range(0, nh1):
                    if abs(coeff1[n]) > tol and np.linalg.norm(
                            self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                        inputs.append((p, index1[n, 1], index1[n, 0]))
                env1 = self.pool['pool'].map(self.environment_s1, inputs)
                inputs = list()
                for n in range(0, nh2):
                    if abs(coeff2[n]) > tol:
                        inputs.append((p, index2[n, 2:4], index2[n, :2]))
                env2 = self.pool['pool'].map(self.environment_s1_s2, inputs)
        else:
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh1):
                if abs(coeff1[n]) > tol and np.linalg.norm(self.operators[index1[n, 1]].reshape(1, -1)) > tol:
                    inputs[n_now % self.pool['n']].append((p, index1[n, 1], index1[n, 0]))
                    n_now += 1
            env1 = self.pool['pool'].map(self.environment_s1_parallel, inputs)
            inputs = empty_list(self.pool['n'], list())
            n_now = 0
            for n in range(0, nh2):
                # if the coefficient is too small, ignore its contribution
                if abs(coeff2[n]) > tol:
                    inputs[n_now % self.pool['n']].append((p, index2[n, 2:4], index2[n, :2]))
                    n_now += 1
            env2 = self.pool['pool'].map(self.environment_s1_s2_parallel, inputs)
            self.pool['pool'].join()
        return s

    def classify_and_update_env(self, envs, coeff, which_env, pos, sn, p):
        if which_env == 'one':
            if pos < p:
                key = '1_0_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif pos == p:
                key = '0_' + str(sn) + '_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[1] * coeff
                else:
                    self.opt_env[key] = envs[1] * coeff
            elif pos > p:
                key = '0_0_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
        elif which_env == 'two':
            if max(pos) < p:
                key = '1_0_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif min(pos) > p:
                key = '0_0_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
            elif max(pos) == p:
                key = '1_' + str(sn[1]) + '_0'
                if key in self.opt_env:
                    self.opt_env[key] += envs[0] * coeff
                else:
                    self.opt_env[key] = envs[0] * coeff
            elif min(pos) == p:
                key = '0_' + str(sn[0]) + '_1'
                if key in self.opt_env:
                    self.opt_env[key] += envs[2] * coeff
                else:
                    self.opt_env[key] = envs[2] * coeff
            else:
                key = '1_0_1'
                if key not in self.opt_env:
                    self.opt_env[key] = list()
                self.opt_env[key].append([coeff, envs[0], envs[2]])

    @staticmethod
    def update_tensor_eigs_f_handle(tensor, env1, env2, coeff1, coeff2, s, tau):
        nh1 = len(env1)
        nh2 = len(env2)
        tensor = tensor.reshape(s)
        tensor1 = tensor.copy()
        if type(env1) is list:
            for n in range(0, nh1):
                tensor1 -= tau * coeff1[n] * T_module.absorb_matrices2tensor_full_fast(
                    tensor, [x.T for x in env1[n]])
            for n in range(0, nh2):
                tensor1 -= tau * coeff2[n] * T_module.absorb_matrices2tensor_full_fast(
                    tensor, [x.T for x in env2[n]])
        else:
            for n in range(0, nh1):
                tensor1 -= tau * env1[n].dot(tensor)
            for n in range(0, nh2):
                tensor1 -= tau * env2[n].dot(tensor)
        return tensor1.reshape(-1, 1)

    def update_tensor_eigs_f_handle_optimized(self, tensor, s, tau):
        tensor = tensor.reshape(s)
        tensor1 = tensor.copy()
        for key in self.opt_env:
            x = key.split('_')
            if x[0] == '1' and x[1] == '0' and x[2] == '0':  # left
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 0)
            elif x[0] == '0' and x[1] == '0' and x[2] == '1':  # right
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 2)
            elif x[0] == '0' and x[1] != '0' and x[2] == '0':  # middle
                tensor1 -= tau * T_module.absorb_matrix2tensor(tensor, self.opt_env[key].T, 1)
            elif x[0] == '1' and x[1] != '0' and x[2] == '0':
                tensor1 -= tau * T_module.absorb_matrices2tensor(
                    tensor, [self.opt_env[key].T, self.operators[int(x[1])].T], [0, 1])
            elif x[0] == '0' and x[1] != '0' and x[2] == '1':
                tensor1 -= tau * T_module.absorb_matrices2tensor(
                    tensor, [self.opt_env[key].T, self.operators[int(x[1])].T], [2, 1])
            elif key == '1_0_1':
                for env in self.opt_env['1_0_1']:
                    tensor1 -= tau * env[0] * T_module.absorb_matrices2tensor(
                        tensor, [x.T for x in env[1:]], [0, 2])
        return tensor1.reshape(-1, )

    def update_tensor_eigs(self, p, index1, index2, coeff1, coeff2, tau, is_real, tol=1e-16):
        _center = self.center
        self.correct_orthogonal_center(p)  # move the orthogonal tensor to p
        if self._is_save_op:
            if -0.5 < _center < p:
                for n in range(_center, p+1):
                    self.del_bad_effective_operators(n)
            elif _center >= p:
                for n in range(p, _center+1):
                    self.del_bad_effective_operators(n)
            else:
                cprint('CenterError: central-orthogonalize MPS before updating the tensor', 'magenta')
                set_trace()
            self.update_all_effective_id()
        if self.eig_way == 0:
            h_effect, s = self.effective_hamiltonian_dmrg(p, index1, index2, coeff1, coeff2)
            h_effect = np.eye(h_effect.shape[0]) - tau * h_effect
        else:
            # env1, env2, s = self.all_environments(p, index1, index2, coeff1, coeff2, tol=tol)
            # dim = np.prod(s)
            # h_effect = LinearOp((dim, dim), lambda a: self.update_tensor_eigs_f_handle(
            #     a, env1, env2, coeff1, coeff2, s, tau))
            s = self.all_environments_optimized(p, index1, index2, coeff1, coeff2, tol=tol)
            dim = np.prod(s)
            h_effect = LinearOp((dim, dim), lambda a: self.update_tensor_eigs_f_handle_optimized(
                a, s, tau))
        self.mps[p] = eigs(h_effect, k=1, which='LM', v0=self.mps[p].reshape(-1, 1),
                           tol=tol)[1].reshape(s)
        if is_real:
            self.mps[p] = self.mps[p].real
        if self.eig_way == 1:
            self.opt_env = dict()

# ==========================================================
    def evolve_tensor(self, p, gate, enlarge_which_bond):
        # gate: a third-order tensor
        # enlarge_which_bond=0: enlarge left virtual bond; 2: enlarge right bond
        d, dd = gate.shape[:2]
        chi1 = self.mps[p].shape[0]
        chi2 = self.mps[p].shape[2]
        self.mps[p] = gate.reshape(d * dd, d).dot(self.mps[p].transpose(1, 0, 2).reshape(d, chi1 * chi2))
        if enlarge_which_bond == 2:
            self.mps[p] = self.mps[p].reshape(d, dd, chi1, chi2).transpose(2, 0, 1, 3).reshape(chi1, d, dd * chi2)
            self.virtual_dim[p + 1] = dd * chi2
        else:
            self.mps[p] = self.mps[p].reshape(d, dd, chi1, chi2).transpose(1, 2, 0, 3).reshape(dd * chi1, d, chi2)
            self.virtual_dim[p] = dd * chi1
        self.orthogonality[p] = 0

    def truncate_virtual_bonds(self, chi1, center, way='simple'):
        if way == 'simple':
            for n in range(self.length):
                dim1 = min(self.virtual_dim[n], chi1)
                dim2 = min(self.virtual_dim[n+1], chi1)
                self.mps[n] = self.mps[n][:dim1, :, :dim2]
                self.virtual_dim[n] = dim1
            self.central_orthogonalization(center, normalize=True)
        else:
            self.correct_orthogonal_center(0)
            self.orthogonalize_mps(0, self.length-1, normalize=True, is_trun=True, chi=chi1)
            self.center = self.length-1
            if center != self.length-1:
                self.correct_orthogonal_center(center)

# ========================================================
    def fidelity_per_site(self, mps, way='log_per_site'):
        length = self.mps.__len__()
        v = np.eye(self.mps[0].shape[0])
        if way == 'log_per_site':
            f = 0
        else:
            f = 1
        for n in range(0, length):
            v = T_module.cont([self.mps[n], mps[n], v], [[2, 3, -2], [1, 3, -1], [1, 2]])
            norm = np.linalg.norm(v.reshape(-1, ))
            v /= norm
            if way == 'log_per_site':
                f -= np.log(norm) / length
            else:
                f *= norm
        return f

    def fidelity_log_by_spins_up(self):
        v = copy.deepcopy(self.mps[0][:, 0, :].reshape(-1, ))
        norm = np.linalg.norm(v)
        v /= norm
        norm = -np.log(norm) / self.length
        for n in range(1, self.length):
            v = np.tensordot(self.mps[n], v, [[0], [0]])[0, :].reshape(-1, )
            norm1 = np.linalg.norm(v)
            v /= norm1
            norm -= np.log(norm1) / self.length
        return norm

    def fidelity_log_to_product_state(self):
        data = self.wrap_data(if_deepcopy=True)
        # self.correct_orthogonal_center(0)
        self.orthogonalize_mps(self.length - 1, 0, normalize=False, is_trun=False)
        self.orthogonalize_mps(0, self.length-1, normalize=False, is_trun=True, chi=1)
        self.center = self.length - 1
        self.mps[self.center] /= np.linalg.norm(self.mps[self.center])
        f = self.fidelity_per_site(data['mps'])
        self.refresh_mps_properties(data)
        return f

    def calculate_entanglement_spectrum(self, if_fast=True):
        # NOTE: this function will central orthogonalize the MPS
        _way = self.decomp_way
        _center = self.center
        self.decomp_way = 'svd'
        if if_fast and _center > -0.5:
            p0 = self.length - 1
            p1 = 0
            for n in range(0, self.length - 1):
                if self.lm[n].size == 0:
                    p0 = min(p0, n)
                    p1 = max(p1, n)
            self.correct_orthogonal_center(p0)
            self.correct_orthogonal_center(p1+1)
            self.correct_orthogonal_center(_center)
        else:
            self.correct_orthogonal_center(0)
            self.correct_orthogonal_center(self.length-1)
            if _center > 0:
                self.correct_orthogonal_center(_center)
        self.decomp_way = _way

    def calculate_entanglement_entropy(self):
        for i in range(0, self.length - 1):
            if self.lm[i].size == 0:
                self.ent[i] = -1
            else:
                self.ent[i] = T_module.entanglement_entropy(self.lm[i])

    def reduced_density_matrix_one_body(self, p):
        v = self.contract_v_with_phys_l0_to_l1(self.center, p, p + (p > self.center) - (p < self.center))
        identity = np.eye(v.shape[2])
        rho = np.tensordot(v, identity, [[2, 3], [0, 1]])
        rho = rho + rho.conj().T
        return rho / np.trace(rho)

    def reduced_density_matrix_two_body(self, p1, p2):
        # p1 < p2
        if self.center < p1:
            v1 = self.contract_v_with_phys_l0_to_l1(self.center, p1, p2)
            v2 = self.contract_v_with_phys_l0_to_l1(p2, p2, p2 - 1)
        elif self.center > p2:
            v1 = self.contract_v_with_phys_l0_to_l1(p1, p1, p1 + 1)
            v2 = self.contract_v_with_phys_l0_to_l1(self.center, p2, p1)
        else:
            v1 = self.contract_v_with_phys_l0_to_l1(p1, p1, p2)
            v2 = self.contract_v_with_phys_l0_to_l1(p2, p2, p2 - 1)
        rho = np.tensordot(v1, v2, ([2, 3], [2, 3])).transpose(0, 2, 1, 3)
        d = rho.shape[0]
        rho = rho.reshape([d*d, d*d])
        return rho/np.trace(rho)

    def reduced_density_matrix_n_body_center(self, n_site, center=None, purify=True):
        if n_site > self.length:
            print('Error: n_site in RDM cannot be larger than the length of MPS')
        if center is None:
            center = int(self.length / 2)
            nt0 = center - int(n_site / 2)
        else:
            if center-n_site/2-1 < 0:
                nt0, = 0
            elif center+n_site/2+1 > self.length:
                nt0 = self.length-n_site
            else:
                nt0 = center-int(n_site/2)
        self.correct_orthogonal_center(center)
        tmp = copy.deepcopy(self.mps[nt0])
        s = tmp.shape
        tmp = tmp.reshape(s[0]*s[1], s[2])
        v_dim0, v_dim1 = s[0], s[2]
        for n in range(1, n_site):
            tmp = np.tensordot(tmp, self.mps[nt0+n], [[1], [0]])
            s = tmp.shape
            tmp = tmp.reshape(s[0]*s[1], s[2])
            if n == (n_site-1):
                v_dim1 = s[-1]
        tmp = tmp.reshape(v_dim0, -1, v_dim1)
        rho = np.tensordot(tmp, tmp.conj(), [[0, 2], [0, 2]])
        rho /= np.trace(rho)
        if purify:
            rho = rho.reshape([self.phys_dim] * n_site * 2)
            ind = [0] * n_site * 2
            for n in range(n_site):
                ind[2*n] = n
                ind[2*n+1] = n + n_site
            rho = rho.transpose(ind).reshape([self.phys_dim*2] * n_site)
        return rho

    def calculate_onsite_entanglement_entropy(self):
        # This function does NOT require the MPS to be central orthogonal
        # If the MPS is central orthogonal, this function will NOT change the center
        vl = empty_list(self.length)
        vl[0] = np.eye(self.virtual_dim[0])
        for n in range(1, self.length):
            if n <= self.center and self.center > -0.5:
                vl[n] = np.eye(self.virtual_dim[n])
            else:
                vl[n] = T_module.cont([self.mps[n-1].conj(), self.mps[n-1], vl[n-1]],
                                      [[1, 2, -1], [3, 2, -2], [1, 3]])
            # vl[n] /= np.linalg.norm(vl[n])
        vr = empty_list(self.length)
        vr[self.length-1] = np.eye(self.virtual_dim[self.length])
        for n in range(self.length-2, -1, -1):
            # if n >= self.center and self.center > -0.5:
            if n >= self.center > -0.5:
                vr[n] = np.eye(self.virtual_dim[n+1])
            else:
                # print(n)
                # print(self.mps[n+1].shape)
                # print(vr[n+1].shape)
                vr[n] = T_module.cont([self.mps[n+1].conj(), self.mps[n+1], vr[n+1]],
                                      [[-1, 2, 1], [-2, 2, 3], [1, 3]])
            # vr[n] /= np.linalg.norm(vr[n])
        ent = np.zeros((self.length, ))
        rho = empty_list(self.length)
        for n in range(self.length):
            # print(self.mps[n].shape)
            rho[n] = T_module.cont([self.mps[n].conj(), self.mps[n], vl[n], vr[n]],
                                [[1, -1, 2], [3, -2, 4], [1, 3], [2, 4]])
            rho[n] = rho[n] + rho[n].conj().T
            rho[n] /= np.trace(rho[n])
            lm = np.linalg.eigh(rho[n])[0]
            lm = lm[lm > 1e-25]
            ent[n] = -np.inner(lm, np.log(lm))
            # ent[n] = -np.trace(rho.dot(logm(rho)))
        return ent, rho

    def markov_measurement(self, if_restore=True):
        order = list()
        v = list()
        if if_restore:
            mps0 = self.wrap_data(['mps', 'orthogonality', 'center', 'ent', 'lm', 'virtual_dim'])
        while self.length > 0.5:
            ent, rho = self.calculate_onsite_entanglement_entropy()
            order.append(int(np.argmax(ent)))
            v.append(eigs(rho[order[-1]], which='LM', k=1)[1].reshape(-1, ))
            # This changes the orthogonal center
            self.correct_orthogonal_center(order[-1])
            self.measure_mps_at_center_by_v(v[-1])
        order = np.array(order)
        order_ref = np.arange(order.size)
        order_true = np.arange(order.size)
        for n in range(0, order.__len__()):
            order_true[n] = order_ref[order[n]]
            order_ref = np.delete(order_ref, order[n])
        if if_restore:
            self.refresh_mps_properties(mps0)
        return order_true, v

    def measure_mps_at_center_by_v(self, v, normalize=True):
        # Observe at the orthogonal center
        # NOTE: MPS will be changed by this function
        norm = 1
        self.mps[self.center] = np.tensordot(self.mps[self.center], v, [[1], [0]])
        if self.length > 1:
            if self.center == (self.length-1):
                self.mps[self.center-1] = np.tensordot(self.mps[self.center-1],
                                                       self.mps[self.center], [[2], [0]])
                self.virtual_dim = np.delete(self.virtual_dim, self.center)
                self.orthogonality = np.delete(self.orthogonality, self.center)
                self.orthogonality[self.center-1] = 0
                self.mps.__delitem__(self.center)
                self.center -= 1
            else:
                self.mps[self.center+1] = np.tensordot(self.mps[self.center],
                                                       self.mps[self.center+1], [[1], [0]])
                self.virtual_dim = np.delete(self.virtual_dim, self.center + 1)
                self.orthogonality = np.delete(self.orthogonality, self.center)
                self.orthogonality[self.center] = 0
                self.mps.__delitem__(self.center)
        if normalize:
            norm = np.linalg.norm(self.mps[self.center])
            self.mps[self.center] /= norm
        self.length -= 1
        return norm

    def observation_s1(self, inputs):
        sn, position = inputs
        operator = self.operators[sn]
        if self._is_save_op:
            if position >= self.center:
                v = self.get_effective_operators_one_body(sn, position, self.center, is_update_op=False)
            else:
                v = self.get_effective_operators_one_body(sn, position, self.center+1, is_update_op=False)
        else:
            if position > self.center:
                v = T_module.bound_vec_operator_right2left(self.mps[position], operator)
                v = self.contract_v_l0_to_l1(position - 1, self.center - 1, v)
            else:
                v = T_module.bound_vec_operator_left2right(self.mps[position], operator)
                v = self.contract_v_l0_to_l1(position + 1, self.center + 1, v)
        return np.trace(v)

    def observation_s1_s2(self, inputs):
        ssn, positions = inputs
        if self._debug:
            self.check_mps_norm1()
        if positions[0] > positions[1]:
            ssn = sort_list(ssn, [1, 0])
            positions = sort_list(positions, [1, 0])
        operators = [self.operators[ssn[0]], self.operators[ssn[1]]]
        if self._is_save_op:
            if self.center <= positions[0]:
                v = self.get_effective_operator_two_body(ssn[0], ssn[1], positions[0],
                                                         positions[1], self.center,
                                                         is_update_op=False)
                return np.trace(v)
            elif self.center > positions[1]:
                v = self.get_effective_operator_two_body(ssn[0], ssn[1], positions[0],
                                                         positions[1], self.center+1,
                                                         is_update_op=False)
                return np.trace(v)
            else:
                vl = self.get_effective_operators_one_body(ssn[0], positions[0],
                                                           self.center, is_update_op=False)
                vr = self.get_effective_operators_one_body(ssn[1], positions[1],
                                                           self.center, is_update_op=False)
                return np.trace(vl.dot(vr.T))
        else:
            if self.center < positions[0]:
                v = self.contract_v_l0_to_l1(self.center, positions[0])
            else:
                v = np.zeros(0)
            v = T_module.bound_vec_operator_left2right(self.mps[positions[0]], operators[0], v)
            v = self.contract_v_l0_to_l1(positions[0] + 1, positions[1], v)
            v = T_module.bound_vec_operator_left2right(self.mps[positions[1]], operators[1], v)
            if positions[1] < self.center:
                v = self.contract_v_l0_to_l1(positions[1] + 1, self.center + 1, v)
            return np.trace(v)

    def observe_magnetization(self, sn):
        mag = np.zeros((self.length, 1))
        inputs = list()
        for i in range(0, self.length):
            if self._is_parallel:
                inputs.append((sn, i))
            else:
                mag[i] = self.observation_s1((sn, i))
        if self._is_parallel:
            mag = np.array(self.pool['pool'].map(self.observation_s1, inputs))
        return mag

    def observe_bond_energy(self, index2, coeff2):
        nh = index2.shape[0]
        eb = np.zeros((nh, 1))
        inputs = list()
        for n in range(0, nh):
            if self._is_parallel:
                inputs.append((index2[n, 2:], index2[n, :2]))
            else:
                eb[n] = coeff2[n] * self.observation_s1_s2((index2[n, 2:], index2[n, :2]))
        if self._is_parallel:
            eb = np.array(self.pool['pool'].map(self.observation_s1_s2, inputs))
        return eb

    def observe_bond_energy_from_jxyz(self, pos2, jx, jy, jz, tol=1e-20):
        nh = pos2.shape[0]
        eb = np.zeros((nh, 1))
        for n in range(0, nh):
            if abs(jx) > tol:
                eb[n] += jx * self.observation_s1_s2(([1, 1], pos2[n, :2]))
            if abs(jy) > tol:
                eb[n] += jy * np.real(self.observation_s1_s2(([2, 2], pos2[n, :2])))
            if abs(jz) > tol:
                eb[n] += jz * self.observation_s1_s2(([3, 3], pos2[n, :2]))
        return eb

    def observe_correlators_from_middle(self, op1, op2, ob_len=None):
        if ob_len is None:
            ob_len = self.length
        corr = list()
        pos_mid = int(round(self.length/2))
        pos1 = pos_mid
        pos2 = pos_mid + 1
        n_control = 0
        while pos1 > -0.1 and pos2 < self.length and ob_len > -0.1:
            corr.append(self.observation_s1_s2(([op1, op2], [pos1, pos2])))
            if n_control % 2 == 0:
                pos1 -= 1
            else:
                pos2 += 1
            n_control += 1
            ob_len -= 1
        return np.array(corr)

    @ staticmethod
    def string_operator(num_op, sz=None, theta=np.pi):
        if sz is None:
            op = [np.array([[1, 0, 0], [0, 0, 0], [0, 0, -1]])]
            exp_sz = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:
            op = [sz]
            exp_sz = None
        for n in range(1, num_op):
            if sz is None:
                op.append(exp_sz)
            else:
                op.append(expm(1j * theta * sz))
        op[-1] = op[-1].dot(op[0])
        return op

    def observe_string_order(self, i, j, sz=None, theta=np.pi):
        tmp = self.mps.copy()
        op = self.string_operator(j - i + 1, sz, theta)
        for n in range(i, j + 1):
            tmp[n] = np.tensordot(op[n - i], tmp[n], ([1], [1])).transpose(1, 0, 2)
        return self.fidelity_per_site(tmp, way='normal')

    def observe_string_order_nn(self, i, j, sz=None, theta=np.pi):
        # Next-nearest-neighbor string order
        tmp = self.mps.copy()
        op = self.string_operator(round((j - i) / 2 + 1), sz, theta)
        for n in range(i, j + 2, 2):
            tmp[n] = np.tensordot(op[round((n - i) / 2)], tmp[n], ([1], [1])).transpose(1, 0, 2)
        return self.fidelity_per_site(tmp, way='normal')

    def norm_mps(self, if_normalize=False):
        # calculate the norm of an MPS
        if self._debug:
            lc = self.check_orthogonal_center()
            if lc != self.center:
                cprint('CenterError: center should be at %d but at %d' % self.center, lc, 'magenta')
                trace_stack()
        if self.center < -0.5:
            v = self.contract_v_l0_to_l1(0, self.length)
            norm = v[0, 0]
        else:
            norm = np.linalg.norm(self.mps[self.center].reshape(1, -1))
        if if_normalize and (self.center > -0.1):
            self.mps[self.center] /= norm
        return norm

    def full_coefficients_mps(self, tol_memory=20):
        cprint('Warning: full_coefficients_mps is used to calculate the full coefficients of the MPS', 'magenta')
        tot_size_log2 = self.length * np.log2(self.phys_dim) - 5
        if tot_size_log2 > tol_memory:
            cprint('The memory cost of the total coefficients is too large (a lot more than %d Mb). '
                   'Stop calculation' % tot_size_log2, 'magenta')
            cprint('If you want to calculate anyway, please input a larger \'tol_memory\'', 'cyan')
        else:
            s = self.mps[0].shape
            x = self.mps[0].reshape(s[0]*s[1], s[2])
            d0 = s[0] * s[1]
            for n in range(1, self.length):
                s = self.mps[n].shape
                x = x.dot(self.mps[n].reshape(s[0], s[1]*s[2]))
                x = x.reshape(d0*s[1], s[2])
                d0 *= s[1]
            return x.reshape(-1, 1)

    def to_unitary_mpo_qubits(self, theta, center=None, if_trun=True):
        if center is None:
            center = self.length-1
        if if_trun:
            self.truncate_virtual_bonds(chi1=2, center=center, way='full')
        mpo = empty_list(self.length)
        mpo[0] = copy.deepcopy(self.mps[0])
        for n in range(1, self.length-1):
            mpo[n] = T_module.tensor3_to_unitary4(self.mps[n], theta=theta)
        mpo[-1] = T_module.get_orthogonal_vecs(self.mps[-1])
        mpo[-1] = mpo[-1].reshape(self.mps[-1].shape[0], self.mps[-1].shape[1],
                                  self.mps[-1].shape[0], self.mps[-1].shape[1])
        return mpo

# ===========================================================
# Checking functions
    def check_orthogonal_center(self, expected_center=-2, if_print=True):
        # Check if MPS has the correct center, or at the expected center
        # if not, find the correct center, or recommend a new center while it is not central orthogonal
        # NOTE: no central-orthogonalization in this function, only recommendation
        if self.center > -0.5:
            left = self.orthogonality[:self.center]
            right = self.orthogonality[self.center+1:]
            if not(np.prod(left == -1) and np.prod(right == 1)):
                if if_print:
                    cprint(colored('self.center is incorrect. Change it to -1', 'magenta'))
                    trace_stack()
                self.center = -1
        if self.center < -0.5:
            left = np.nonzero(self.orthogonality == -1)
            left = left[0][-1]
            right = np.nonzero(self.orthogonality == 1)
            right = right[0][0]
            if np.prod(self.orthogonality[:left+1]) and np.prod(self.orthogonality[right:]) and left + 2 == right:
                self.center = left+1
                if if_print:
                    cprint(colored('self.center is corrected to %g' % self.center, 'cyan'))
            else:
                if if_print:
                    cprint(colored('MPS is not central orthogonal. self.center remains -1', 'cyan'))
        else:
            left = self.center - 1
        if self.center > -0.5:
            if expected_center > -0.5 and expected_center != self.center:
                cprint('The center is at %d, not the expected position at %d' % (self.center, expected_center))
            recommend_center = self.center
        else:
            # if not central-orthogonal, recommend the tensor after the last left-orthogonal one as the new center
            recommend_center = left + 1
        return recommend_center

    def check_orthogonality_by_tensors(self, tol=1e-20, is_print=True):
        incorrect_ort = list()
        for n in range(0, self.length):
            if self.orthogonality[n] == -1:
                is_ort = T_module.check_orthogonality(self.mps[n], [2], tol=tol)
            elif self.orthogonality[n] == 1:
                is_ort = T_module.check_orthogonality(self.mps[n], [0], tol=tol)
            else:
                is_ort = True
            if not is_ort:
                incorrect_ort.append(n)
        if is_print:
            if incorrect_ort.__len__() == 0:
                print('The orthogonality of all tensors are marked correctly by self.orthogonality')
            else:
                cprint('In self.orthogonality, the orthogonality of the following tensors is incorrect:', 'magenta')
                cprint(str(incorrect_ort), 'cyan')
        return incorrect_ort

    def check_environments(self, vl, vm, vr, n):
        # check if the environments of the n-th tensor have consistent dimensions
        is_bug0 = False
        bond = str()
        if vl.shape[0] != vl.shape[1]:
            is_bug0 = True
            bond = 'LEFT'
        if vm.shape[0] != vm.shape[1]:
            is_bug0 = True
            bond = 'MIDDLE'
        if vr.shape[0] != vr.shape[1]:
            is_bug0 = True
            bond = 'RIGHT'
        if is_bug0:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v is not square', 'magenta')
        is_bug = False
        if vl.shape[0] != self.virtual_dim[n]:
            is_bug = True
            bond = 'LEFT'
        if vr.shape[0] != self.virtual_dim[n + 1]:
            is_bug = True
            bond = 'RIGHT'
        if vm.shape[0] != self.mps[n].shape[1]:
            bond = 'MIDDLE'
            is_bug = True
        if is_bug:
            cprint('EnvError: for the %d-th tensor, the ' % n + bond + ' v has inconsistent dimension', 'magenta')
        if is_bug0 or is_bug:
            trace_stack()

    def check_virtual_bond_dimensions(self):
        is_error = False
        for n in range(1, self.length):
            if self.virtual_dim[n] != self.mps[n].shape[0] or self.virtual_dim[n] != self.mps[n-1].shape[2]:
                cprint('BondDimError: inconsistent dimension detected for the %d-th virtual bond' % n, 'magenta')
                is_error = True
        if is_error:
            trace_stack(2)

    def check_mps_norm1(self, if_print=False):
        # check if the MPS is norm-1
        norm = self.norm_mps()
        if abs(norm - 1) > 1e-14:
            print_error('The norm is MPS is %g away from 1' % abs(norm - 1))
        if if_print:
            cprint('The norm of MPS is %g' % norm, 'cyan')

    def clean_to_save(self):
        self.effect_s = dict()
        self.effect_ss = dict()
        self.effect_s = {'none': np.zeros(0)}
        self.effect_ss = {'none': np.zeros(0)}
        self.effective_id = {'none': np.zeros(0)}
        self.pos_effect_s = np.zeros((0, 3)).astype(int)
        self.pos_effect_ss = np.zeros((0, 5)).astype(int)
        self.pool = None
        self.tmp = None


class MpsStandardTEBD(MpsOpenBoundaryClass):

    def __init__(self, length, d, chi, spin='half', ini_way='r', evolve_way='gate'):
        MpsOpenBoundaryClass.__init__(self, length, d, chi, spin=spin, way='qr', ini_way=ini_way, operators=None,
                                      debug=False, is_parallel=False, is_save_op=False, eig_way=0, par_pool=None,
                                      is_env_parallel_lmr=False, is_eco_dims=False)
        self.chi = chi

    def evolve_gate_tebd(self, p1, p2, gates):
        # gate: two third-order tensors
        if p1 > p2:
            (p1, p2) = (p2, p1)
        dd = gates[0].shape[1]
        identity = np.eye(dd).reshape(1, -1)
        self.evolve_tensor(p1, gates[0], 2)
        for n in range(p1+1, p2):
            s = self.mps[n].shape
            self.mps[n] = self.mps[n].reshape(-1, 1).dot(identity).reshape(s + (dd, dd)).transpose(
                3, 0, 1, 4, 2).reshape(dd*s[0], s[1], dd*s[2])
            self.virtual_dim[n + 1] = dd * s[2]
            self.orthogonality[n] = 0
        self.evolve_tensor(p2, gates[1], 0)

    def truncate_mps_tebd(self, p1, p2):
        if p1 > p2:
            (p1, p2) = (p2, p1)
        c0 = self.center
        if c0 < p1:
            self.orthogonalize_mps(p2, p1, normalize=True, is_trun=False)
            self.orthogonalize_mps(c0, p1, normalize=True, is_trun=False)
            self.orthogonalize_mps(p1, p2, normalize=True, is_trun=True, chi=self.chi)
            self.center = p2
        elif c0 > p2:
            self.orthogonalize_mps(self.center, p1, normalize=True, is_trun=False)
            self.orthogonalize_mps(p1, p2, normalize=True, is_trun=True, chi=self.chi)
            self.center = p2
        else:
            self.orthogonalize_mps(p2, p1, normalize=True, is_trun=False)
            self.orthogonalize_mps(p1, p2, normalize=True, is_trun=True, chi=self.chi)
            self.center = p2


class MpsTebdAnyH(MpsOpenBoundaryClass):

    def __init__(self, phys_dims, d, chi, mpo_pos, spin='half', ini_way='r'):
        self.phys_dims = phys_dims
        length = len(phys_dims)
        MpsOpenBoundaryClass.__init__(self, length, d, chi, spin=spin, way='qr', ini_way=ini_way, operators=None,
                                      debug=False, is_parallel=False, is_save_op=False, eig_way=0, par_pool=None,
                                      is_env_parallel_lmr=False, is_eco_dims=False)
        self.chi = chi
        self.virtual_dim = list(np.ones((self.length+1, )) * self.chi)
        self.virtual_dim[0] = 1
        self.virtual_dim[-1] = 1
        self.initial_mps()
        self.mpo = list()
        self.mpo_pos = mpo_pos

    def initial_mps(self):
        self.mps = [] * self.length
        for n in range(self.length):
            self.mps[n] = np.random.randn(self.virtual_dim[n], self.phys_dims[n],
                                          self.virtual_dim[n+1])

    def get_mpo_strings(self, hamilt, coup, tau):
        nh = hamilt.__len__()
        n_mpo = coup.__len__()
        gates_u = [] * nh
        gates_v = [] * nh
        for n in range(nh):
            s = hamilt.shape
            tmp = expm(-tau * hamilt.reshape(s[0]*s[1], s[2]*s[3])).reshape(s).transpose(
                0, 2, 1, 3).reshape(s[0]*s[2], s[1]*s[3])
            gates_u[n], lm, gates_v[n] = np.linalg.svd(tmp)
            d = min(s[0] * s[2], s[1] * s[3])
            gates_u[n] = gates_u[n][:, :d].dot(np.diag(lm ** 0.5)).reshape(
                s[0], s[2], d).transpose(0, 2, 1)
            gates_v[n] = np.diag(lm ** 0.5).dot(gates_v[n][:d, :]).reshape(
                d, s[1], s[3]).transpose(1, 0, 2)
        self.mpo = [] * n_mpo
        for n in range(n_mpo):
            self.mpo[n] = list()
            self.mpo[n].append(gates_u[coup[n][0]])
            self.mpo[n][-1] = self.mpo[n][-1].reshape((1,) + self.mpo[n][-1].shape).transpose(0, 1, 3, 2)
            for nn in range(1, coup[n].__len__()):
                p1 = coup[n][nn - 1]
                p2 = coup[n][nn]
                if nn % 2 == 1:
                    self.mpo[n].append(np.tensordot(gates_v[p1], gates_u[p2], ([2], [0])))
                    self.mpo[n][-1] = self.mpo[n][-1].transpose(1, 0, 3, 2)
                else:
                    self.mpo[n].append(np.tensordot(gates_v[p1], gates_u[p2], ([0], [2])))
                    self.mpo[n][-1] = self.mpo[n][-1].transpose(0, 2, 1, 3)
            self.mpo[n].append(gates_v[coup[n][-1]])
            self.mpo[n][-1] = self.mpo[n][-1].reshape((1,) + self.mpo[n][-1].shape).transpose(2, 1, 3, 0)

    def evolve_and_truncate_mpo_string(self, ns):
        # Evolve by the ns-th MPO string
        self.evolve_one_tensor(ns, self.mpo_pos[ns][0])
        for n in range(1, self.mpo_pos[ns].__len__()):
            for nn in range(self.mpo_pos[ns][n-1]+1, self.mpo_pos[ns][n]):
                d = self.mpo[ns][n-1].shape[3]
                id = np.eye(d)
                self.evolve_identity_one_tensor(nn, id)
            self.evolve_one_tensor(ns, self.mpo_pos[ns][n])
        p_max = max(self.center, self.mpo_pos[ns][-1])
        p_min = min(self.center, self.mpo_pos[ns][0])
        for n in range(p_min, p_max + 1):
            self.orthogonality[n] = 0
        # Orthogonalize and truncate
        if self.center < self.mpo_pos[ns][0]:
            self.orthogonalize_mps(self.center, self.mpo_pos[ns][-1])
            self.orthogonalize_mps(self.mpo_pos[ns][-1], self.mpo_pos[ns][0],
                                   is_trun=True, chi=self.chi)
        elif self.center > self.mpo_pos[ns][-1]:
            self.orthogonalize_mps(self.center, self.mpo_pos[ns][0])
            self.orthogonalize_mps(self.mpo_pos[ns][0], self.mpo_pos[ns][-1],
                                   is_trun=True, chi=self.chi)
        else:
            self.orthogonalize_mps(self.mpo_pos[ns][0], self.mpo_pos[ns][-1])
            self.orthogonalize_mps(self.mpo_pos[ns][-1], self.mpo_pos[ns][0],
                                   is_trun=True, chi=self.chi)

    def evolve_one_tensor(self, ns, p):
        # For the ns-th MPO string, evolve the p-th tensor
        self.mps[p] = np.tensordot(self.mpo[ns][0], self.mps[p], ([2], [1])).transpose(0, 3, 1, 2, 4)
        s = self.mps[p].shape
        self.mps[p] = self.mps[p].reshape(s[0] * s[1], s[2], s[3] * s[4])
        self.virtual_dim[p] = self.mps[p].shape[0]

    def evolve_identity_one_tensor(self, p, id):
        # Evolve the p-th tensor by identity
        d = id.shape[0]
        s = self.mps[p].shape
        self.mps[p] = np.outer(self.mps[p].reshape(-1, ), id.reshape(-1, ))
        self.mps[p] = self.mps[p].reshape(s + (d, d)).transpose(3, 1, 0, 4, 2).reshape(
            d * s[0], s[1], d * s[2])


# ========================================================================
# class for infinite-size MPS
# relevant algorithms: iDMRG, iTEBD, AOP (1D)
class MpsInfinite(MpsBasic):
    """The tensors and bonds for central orthogonal MPS are arranged as:
            |        |        |
        - mps[0] - mps[1] - mps[2] -

             1                    1                    1
             |                    |                    |
        0 - mps[0] - 2       0 - mps[1] - 2       2 - mps[2] - 0
      (left-to-right ort)     (normalized)      (right-to-left ort)

    The tensors and bonds for MPO are arranged as:
            1
            |
        0 - T - 3
            |
            2
    """
    def __init__(self, form, d, chi, D, n_tensor=3, n_site=1, spin='half', dmrg_type='mpo',
                 way='qr', operators=None, hamilt_index=None, is_symme_env=True, env=None,
                 is_real=True, debug=False):
        # form = 'center_ort': central orthogonal form; self.n_tensor fixed to 3 and self.lm[n] = zeros(0)
        # form = 'translation_invariant': central orthogonal form; self.n_tensor is flexible, self.lm[n]
        # must be initialized
        # dmrg_type: mpo - mpo way; white: conventional way by effective operators
        MpsBasic.__init__(self)
        self.spin = spin
        self.n_tensor = n_tensor  # number of tensors to give the iMPS
        self.n_site = n_site  # n-site DMRG algorithm
        self.mps = empty_list(self.n_tensor)
        self.lm = [np.zeros(0) for _ in range(0, self.n_tensor)]
        self.env = [np.zeros(0) for _ in range(0, 2)]
        if self.n_site == 1:
            self.rho = np.ndarray([])
        else:
            self.rho = empty_list(self.n_site*2-1)

        self.orthogonality = np.zeros((self.n_tensor, 1)).astype(int)
        self.is_center_ort = False
        self.is_canonical = False
        self.d = d
        self.chi = chi
        self.D = D  # dimension of the "physical" bond of the time MPS
        self.form = form

        self.decomp_way = way
        self.is_symme_env = is_symme_env
        self.initialize_imps()
        if env is not None:
            self.env = env
        if is_symme_env:
            self.env[1] = self.env[0]
        self.is_real = is_real
        if operators is None:
            op_half = spin_operators(spin)
            self.operators = [op_half['id'], op_half['sx'], op_half['sy'], op_half['sz'],
                              op_half['su'], op_half['sd']]
        else:
            self.operators = operators
        self._debug = debug
        self.dmrg_type = dmrg_type
        # On-site effective operators (total)
        self.bath_op_onsite = np.eye(chi)
        self.effective_ops = [np.eye(chi) for _ in range(self.operators.__len__())]
        self.hamilt_index = hamilt_index  # Indexes for the interactions in hamilt
        # the set of indexes to be updated as effective ops
        # self.op_index = set(range(self.operators.__len__()))
        self.op_index = set(self.hamilt_index.reshape(-1, ).astype(int))

    def initialize_imps(self):
        if self.form == 'center_ort':
            # randomly initialize central orthogonal tensor
            self.n_tensor = 3
            if self.n_site == 1:
                self.mps[1] = np.random.randn(self.chi, self.d, self.chi)
            else:  # n_site == 2
                self.mps[1] = np.random.randn(self.chi, self.d, self.d, self.chi)
            self.mps[1] /= np.linalg.norm(self.mps[1].reshape(-1, 1))
            self.mps[0] = np.ndarray([])
            self.mps[2] = np.ndarray([])
            self.lm = [np.zeros(0)] * self.n_tensor
            self.update_ort_tensor_mps('both')
            self.env[0] = np.ones((self.chi, self.D, self.chi))
            self.env[1] = np.ones((self.chi, self.D, self.chi))
            self.orthogonality = np.array([-1, 0, 1])
            self.is_center_ort = True
        elif self.form == 'translation_invariant':
            # randomly initialize translational invariant tensor
            # the ordering is: - lm[0] - mps[0] - lm[1] - mps[1] - ... - lm[n] - mps[n]
            self.mps[0] = np.random.randn(self.chi, self.d, self.chi)
            self.lm[0] = np.ones((self.chi, 1)) / (self.chi ** (-0.5))
            for n in range(1, self.n_tensor):
                self.mps[n] = self.mps[0].copy()
                self.lm[n] = self.lm[0].copy()

    def simplified_op_index(self):
        self.op_index = set()
        self.op_index.add(1)
        self.op_index.add(3)
        for n in range(self.hamilt_index.shape[0]):
            self.op_index.add(int(self.hamilt_index[n, 0]))
            self.op_index.add(int(self.hamilt_index[n, 1]))

    def update_left_env(self, tensor):
        # compatible to one-site and two-site iDMRG
        self.env[0] = T_module.cont([self.mps[0].conj(), tensor, self.mps[0], self.env[0]],
                                    [[4, 5, -1], [1, 5, 3, -2], [2, 3, -3], [4, 1, 2]])
        self.env[0] /= np.linalg.norm(self.env[0].reshape(1, -1))
        self.env[0] = (self.env[0] + self.env[0].transpose(2, 1, 0)) / 2

    def update_right_env(self, tensor):
        # compatible to one-site and two-site iDMRG
        self.env[1] = T_module.cont([self.mps[2].conj(), tensor, self.mps[2], self.env[1]],
                                    [[4, 5, -1], [-2, 5, 3, 1], [2, 3, -3], [4, 1, 2]])
        self.env[1] /= np.linalg.norm(self.env[1].reshape(1, -1))
        self.env[1] = (self.env[1] + self.env[1].transpose(2, 1, 0)) / 2

    def update_ort_tensor_mps(self, which, dc=None):
        if self.n_site == 1:
            if which in ['left', 'both']:
                tmp = T_module.left2right_decompose_tensor(self.mps[1], self.decomp_way)
                self.mps[0] = tmp[0]
                if self.decomp_way == 'svd':
                    self.lm[0] = tmp[3]
            if which in ['right', 'both']:
                tmp = T_module.left2right_decompose_tensor(self.mps[1].transpose(2, 1, 0),
                                                           self.decomp_way)
                self.mps[2] = tmp[0]
                if self.decomp_way == 'svd':
                    self.lm[1] = tmp[3]
        elif self.n_site == 2:
            s = self.mps[1].shape
            self.mps[0], self.lm[0], self.mps[2] = np.linalg.svd(
                self.mps[1].reshape(s[0] * s[1], s[2] * s[3]))
            if dc is None:
                dc = min(self.chi, s[0] * s[1])
            else:
                dc = min(dc, s[0] * s[1])
            self.mps[0] = self.mps[0][:, :dc].reshape(s[0], s[1], dc)
            self.mps[2] = self.mps[2][:dc, :].reshape(dc, s[2], s[3]).transpose(2, 1, 0)
            self.lm[0] = self.lm[0][:dc]

    def update_central_tensor_effective_ops_fh(self, psi, tau):
        psi = psi.reshape(self.chi, self.d, self.d, self.chi)
        psi1 = psi.copy()
        # Projecting on-site bath parts
        psi1 -= tau * self.update_by_given_effective_ops(psi, [self.bath_op_onsite], [0])
        psi1 -= tau * self.update_by_given_effective_ops(psi, [self.bath_op_onsite], [3])
        # Projecting physical parts
        for n in range(0, self.hamilt_index.shape[0]):
            op1 = self.operators[int(self.hamilt_index[n, 0])]
            op2 = self.operators[int(self.hamilt_index[n, 1])]
            j = self.hamilt_index[n, 2]
            psi1 -= tau * j * self.update_by_given_effective_ops(psi, [op1, op2], [1, 2])
        # Projecting physical-bath parts [left and right parts are symmetrical]
        for n in range(0, self.hamilt_index.shape[0]):
            op1 = self.effective_ops[int(self.hamilt_index[n, 0])]
            op2 = self.operators[int(self.hamilt_index[n, 1])]
            j = self.hamilt_index[n, 2]
            psi1 -= tau * j * self.update_by_given_effective_ops(psi, [op1, op2], [0, 1])
            psi1 -= tau * j * self.update_by_given_effective_ops(psi, [op1, op2], [3, 2])
        return psi1

    def effective_hamilt_from_op(self):
        h = np.zeros(((self.chi ** 2) * (self.d ** 2), (self.chi ** 2) * (self.d ** 2)))
        id0 = np.eye(self.d)
        id1 = np.eye(self.chi)
        for n in range(0, self.hamilt_index.shape[0]):
            op1 = self.operators[int(self.hamilt_index[n, 0])]
            op2 = self.operators[int(self.hamilt_index[n, 1])]
            j = self.hamilt_index[n, 2]
            h += j * np.kron(id1, np.kron(np.kron(op1, op2), id1))
        # Projecting on-site bath parts
        h += np.kron(np.kron(np.kron(self.bath_op_onsite, id0), id0), id1)
        h += np.kron(np.kron(np.kron(id1, id0), id0), self.bath_op_onsite)
        # Projecting physical-bath parts [left and right parts are symmetrical]
        for n in range(0, self.hamilt_index.shape[0]):
            op1 = self.effective_ops[int(self.hamilt_index[n, 0])]
            op2 = self.operators[int(self.hamilt_index[n, 1])]
            j = self.hamilt_index[n, 2]
            h += j * np.kron(np.kron(np.kron(op1, op2), id0), id1)
            h += j * np.kron(np.kron(np.kron(id1, id0), op2), op1)
        return h

    def effective_hamilt_white_dmrg(self, tau, way='fh'):
        dim = (self.chi ** 2) * (self.d ** 2)
        if way == 'fh':
            return LinearOp((dim, dim), lambda v: self.update_central_tensor_effective_ops_fh(
                v, tau))
        else:
            return np.eye(dim) - tau * self.effective_hamilt_from_op()

    def update_effective_ops(self, which='op_index'):
        # Only use mps[0] assuming left and right parts are symmetrical
        if which == 'op_index':
            for n in self.op_index:
                self.effective_ops[n] = T_module.bound_vec_operator_left2right(
                    self.mps[0], self.operators[n])
        elif type(which) is int:
            self.effective_ops[which] = T_module.bound_vec_operator_left2right(
                self.mps[0], self.operators[which])
        elif which == 'all':
            for n in range(self.effective_ops.__len__()):
                self.effective_ops[n] = T_module.bound_vec_operator_left2right(
                    self.mps[0], self.operators[n])
        elif (type(which) is list) or (type(which) is tuple):
            for n in which:
                self.effective_ops[n] = T_module.bound_vec_operator_left2right(
                    self.mps[0], self.operators[n])

    def update_bath_onsite(self):
        op = T_module.bound_vec_operator_left2right(
            self.mps[0], np.eye(self.d), self.bath_op_onsite)
        # op = self.bath_op_onsite.copy()
        for n in range(0, self.hamilt_index.shape[0]):
            op1 = self.effective_ops[int(self.hamilt_index[n, 0])]
            op2 = self.operators[int(self.hamilt_index[n, 1])]
            j = self.hamilt_index[n, 2]
            op += j * T_module.bound_vec_operator_left2right(self.mps[0], op2, op1)
        self.bath_op_onsite = (op + op.conj().T) / 2

    @ staticmethod
    def update_by_given_effective_ops(psi, ops, bonds):
        indexes = empty_list(1 + bonds.__len__())
        indexes[0] = list(range(psi.ndim))
        x = 1
        for n in range(psi.ndim):
            if n in bonds:
                indexes[0][n] = x
                indexes[bonds.index(n) + 1] = [-n - 1, x]
                x += 1
            else:
                indexes[0][n] = -n - 1
        return T_module.cont([psi] + ops, indexes)

    def central_orthogonalize_imps_without_h(self, mps='mps', iter_time=200, tol=1e-12):
        cprint('Warning: this function has not been testified!', 'red')
        """ Only for the MPS in the following form:
                    1                1
                    |                |
                - mps[0] - lm[0] - mps[2] -
               0         2       2         0
            (NOTE: this function will change self.mps; the original mps can deviate
            from central orthogonal form)
        """
        if mps == 'mps':
            for t in range(0, iter_time):
                vl = T_module.bound_vec_operator_left2right(
                    self.mps[0], normalize=True, symme=True)
                vr = T_module.bound_vec_operator_left2right(
                    self.mps[2], normalize=True, symme=True)
                ul, ur, lm = T_module.transformation_from_env_mats(
                    vl, vr, self.lm[0])[:3]
                self.mps[0] = np.tensordot(np.tensordot(self.mps[0], vl, ([2], [0])),
                                           np.linalg.pinv(vl), ([0], [0]))
                self.mps[2] = np.tensordot(np.tensordot(self.mps[2], vr, ([2], [0])),
                                           np.linalg.pinv(vr), ([0], [0]))
                if np.linalg.norm(lm - self.lm[0]) < tol:
                    self.lm[0] = lm
                    break
                else:
                    self.lm[0] = lm

    def effective_hamiltonian(self, tensor):
        # compatible to one-site and two-site iDMRG
        if self.n_site == 1:
            if self.is_symme_env:
                h = T_module.cont([self.env[0], tensor, self.env[0]],
                                  [[-1, 1, -4], [1, -2, -5, 2], [-3, 2, -6]])
            else:
                h = T_module.cont([self.env[0], tensor, self.env[1]],
                                  [[-1, 1, -4], [1, -2, -5, 2], [-3, 2, -6]])
            s = h.shape
            h = h.reshape(s[0] * s[1] * s[2], s[3] * s[4] * s[5])
            h = (h + h.transpose(1, 0)) / 2
            return h
        elif self.n_site == 2:
            if self.is_symme_env:
                h = T_module.cont([self.env[0], tensor, tensor, self.env[0]],
                                  [[-1, 1, -5], [1, -2, -6, 3], [3, -3, -7, 2], [-4, 2, -8]])
            else:
                h = T_module.cont([self.env[0], tensor, tensor, self.env[1]],
                                  [[-1, 1, -5], [1, -2, -6, 3], [3, -3, -7, 2], [-4, 2, -8]])

            s = h.shape
            h = h.reshape(s[0] * s[1] * s[2] * s[3], s[4] * s[5] * s[6] * s[7])
            h = (h + h.transpose(1, 0)) / 2
            return h

    def update_central_tensor(self, inputs):
        # compatible to one-site and two-site iDMRG
        s = self.mps[1].shape
        if self.dmrg_type == 'white':
            tau, way = inputs
            h = self.effective_hamilt_white_dmrg(tau, way)
        else:
            h = self.effective_hamiltonian(inputs)
        self.mps[1] = eigs(h, 1, v0=self.mps[1].reshape(-1, 1))[1].reshape(s)
        if self.is_real:
            self.mps[1] = self.mps[1].real

    def obtain_ring_tensor(self):
        if self.dmrg_type == 'mpo':
            # ring tensor indexes: D, d, d, D, d, d
            if self.is_symme_env:
                tmp1 = np.tensordot(self.env[0], self.mps[1], [[2], [0]])
                tmp2 = np.tensordot(self.env[0], self.mps[1].conj(), [[0], [3]])
                return np.tensordot(tmp1, tmp2, [[0, 4], [2, 1]])
            else:
                tmp1 = np.tensordot(self.env[0], self.mps[1], [[2], [0]])
                tmp2 = np.tensordot(self.env[1], self.mps[1].conj(), [[0], [3]])
                return np.tensordot(tmp1, tmp2, [[0, 4], [2, 1]])

    def rho_from_central_tensor(self):
        if self.n_site == 1:
            self.rho = np.tensordot(self.mps[1].conj(), self.mps[1], ([0, 2], [0, 2]))
        elif (self.n_site == 2) and self.dmrg_type == 'mpo':
            # In this case, the dimension of one physical bond is d*d = 4
            d0 = self.operators[0].shape[0]
            tmp = self.mps[1].reshape(self.chi, d0, d0, d0, d0, self.chi)
            nb = 6
            self.rho = empty_list(nb - 3)
            for n in range(1, nb - 2):
                bonds_con = list(range(0, n)) + list(range(n + 2, nb))
                self.rho[n - 1] = np.tensordot(
                    tmp, tmp, (bonds_con, bonds_con)).reshape(self.d, self.d)
        elif (self.n_site == 2) and self.dmrg_type == 'white':
            self.rho = np.tensordot(self.mps[1].conj(), self.mps[1], ([0, 3], [0, 3])).reshape(
                self.d ** 2, self.d ** 2)
        return self.rho

    def rho_bulk_iDMRG(self, length_bulk):
        """
        mps[0]: left-ort tensor
        mps[1]: central tensor
        mps[2]: right-ort tensor

        v:
        0  1  2  3  4  5
        |  |  |  |  |  |
        ----------------
        |  |  |  |  |  |
        6  7  8  9  10 11
        """
        #

        def contract_one_tensor(_v, tensor):
            _v = T_module.cont([tensor, tensor.conj(), _v],
                               [[1, -2, -5], [2, -4, -6], [-1, -3, 1, 2]])
            _s = _v.shape
            _v = _v.reshape(_s[0] * _s[1], _s[2] * _s[3], _s[4], _s[5])
            return _v

        v = np.eye(self.chi).reshape(1, 1, self.chi, self.chi)
        lh = int(int((length_bulk - 1) / 2))
        for n in range(lh):
            v = contract_one_tensor(v, self.mps[0])
        dims = [self.mps[0].shape[1]] * lh

        if self.mps[1].ndim > 3:
            s = self.mps[1].shape
            mps1 = self.mps[1].reshape(s[0], -1, s[-1])
        else:
            mps1 = self.mps[1]
        v = contract_one_tensor(v, mps1)
        dims.append(mps1.shape[1])

        for n in range(length_bulk-2-lh):
            v = contract_one_tensor(v, self.mps[2].transpose(2, 1, 0))
        v = T_module.cont([self.mps[2], self.mps[2].conj(), v],
                          [[1, -2, 2], [1, -4, 3], [-1, -3, 2, 3]])
        dims = dims + [self.mps[2].shape[1]] * (length_bulk-1-lh)
        v = v.reshape(dims * 2)
        return v

    def observe_energy(self, h):
        # compatible to one-site iDMRG and one-site deep iDMRG
        if self.form == 'center_ort':
            if type(self.rho) is np.ndarray:
                energy = np.trace(self.rho.dot(h))
                return energy
            elif type(self.rho) is list:
                nr = self.rho.__len__()
                energy = np.ndarray(nr, )
                for n in range(0, nr):
                    energy[n] = np.trace(self.rho[n].dot(h))
                return energy

    def check_orthogonality_mps(self):
        no_bug = True
        tol = 1e-12
        tmp = np.eye(self.chi).reshape(-1, 1)
        tmp1 = np.tensordot(self.mps[0].conj(), self.mps[0], ([0, 1], [0, 1])).reshape(-1, 1)
        if np.linalg.norm(tmp - tmp1) > tol:
            no_bug = False
            print('The left part of the MPS is not orthogonal')
        tmp1 = np.tensordot(self.mps[2].conj(), self.mps[2], ([0, 1], [0, 1])).reshape(-1, 1)
        if np.linalg.norm(tmp - tmp1) > tol:
            no_bug = False
            print('The right part of the MPS is not orthogonal')
        if no_bug:
            print('The MPS satisfies the orthogonality')


def fidelity_per_site(mps1, mps2, way='log_per_site'):
    """
    :param mps1: one MPS
    :param mps2: the other MPS
    :return: ln fidelity per site
    Example:
    >>> f = fidelity_per_site(mps1, mps2)
    """
    length = mps1.__len__()
    v = np.eye(mps1[0].shape[0])
    if way == 'log_per_site':
        f = 0
    else:
        f = 1
    for n in range(0, length):
        v = T_module.cont([mps1[n], mps2[n], v], [[2, 3, -2], [1, 3, -1], [1, 2]])
        norm = np.linalg.norm(v.reshape(-1, ))
        v /= norm
        if way == 'log_per_site':
            f -= np.log(norm) / length
        else:
            f *= norm
    return f



