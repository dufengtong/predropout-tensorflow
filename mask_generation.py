import numpy as np
import math
import pickle

def nCr(n, r):
    '''number of combinations when choosing r from n'''
    f = math.factorial
    return f(n) // f(r) // f(n-r)

def unique_rows(a):
    '''remove same rows with same order'''
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def generate_random_fake_fvs(v_num, v_len, code_num, max_overlap=None):
    """
    :param v_num: number of fvs need to generated.(number of concept classes.)
    :param v_len: number of neurons encoding all the concept.
    :param code_num: number of neurons that encode one concept.
    :return:
    """
    if max_overlap==None: max_overlap = int(code_num / 2)
    # check vector number
    if v_num > nCr(np.float32(v_len), np.float32(code_num)):
        print('Vector number %d is out of boundary, \nthe maximum vector number is %d,' \
              '\nplease check and try again!' % (v_num, nCr(v_len, code_num)))
        exit(0)

    fvs = np.zeros([v_num, v_len])
    fv_num = 0
    while fv_num < v_num:
        fire_idx = np.sort(np.random.choice(v_len, code_num, replace=False)).astype(int)
        fv = np.zeros(v_len)
        fv[fire_idx] = 1
        overlaps = np.sum(fvs * fv, axis=1)
        if np.max(overlaps) <= max_overlap:
            fvs[fv_num] = fv
            fv_num += 1

    print('Generate %d features with %d neuron per concept done!' % (len(fvs), code_num))
    return fvs, max_overlap

def generate_fake_fvs(v_num, v_len, code_num):
    fvs = np.zeros([v_num, v_len])

    overlap = int((v_num*code_num - v_len) / (v_num - 1))
    for i in range(v_num):
        fv = np.zeros(v_len)
        one_start_idx = (code_num - overlap) * i
        fv[one_start_idx:one_start_idx+code_num] = 1
        fvs[i] = fv
    return fvs, overlap

def generate_share_neuron_fvs(c, o, m):
    n = c * (m - o) + o # vector length when c vectors share o neurons
    fvs = np.ones([c, n])
    print(c, n-o, m-o)
    indepent_fvs,_ = generate_fake_fvs(c, n-o, m-o)
    fvs[:, o:] = indepent_fvs
    return fvs, n


if __name__=='__main__':
    class_num = 10
    concept_coding_num = 20
    vector_len = 40
    overlap = None
    fvs, overlap = generate_random_fake_fvs(v_num=class_num, v_len=vector_len, code_num=concept_coding_num, max_overlap=overlap)
    # fvs, overlap = generate_fake_fvs(class_num, vector_len, concept_coding_num)
    # fvs, vector_len = generate_share_neuron_fvs(class_num, overlap, concept_coding_num)
    print(fvs)
    pickle_name = 'feature_vectors/fv_%d_%d_%d_%d.pickle' % (class_num, concept_coding_num, vector_len, overlap)
    with open(pickle_name, 'wb') as f:
        pickle.dump(fvs, f)
