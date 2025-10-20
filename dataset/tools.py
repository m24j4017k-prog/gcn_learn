import numpy as np
import random
import torch
import math

def downsample(data_numpy, step, random_sample=True):
    # input: C,T,V,M
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy, step):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T / step, step, V, M).transpose(
        (0, 1, 3, 2, 4)).reshape(C, T / step, V, step * M)


def mean_subtractor(data_numpy, mean):
    # input: C,T,V,M
    # naive version
    if mean == 0:
        return
    C, T, V, M = data_numpy.shape
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy, size, random_pad=False):
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin:begin + T, :, :] = data_numpy
        return data_numpy_paded
    else:
        return data_numpy


def random_choose(data_numpy, size, auto_pad=True):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    if T == size:
        return data_numpy
    elif T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        else:
            return data_numpy
    else:
        begin = random.randint(0, T - size)
        return data_numpy[:, begin:begin + size, :, :]




def random_move(data_numpy,
                angle_candidate=[-10., -5., 0., 5., 10.],
                scale_candidate=[0.9, 1.0, 1.1],
                transform_candidate=[-0.2, -0.1, 0.0, 0.1, 0.2],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias:bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy):
    C, T, V, M = data_numpy.shape
    assert (C == 3)
    score = data_numpy[2, :, :, :].sum(axis=1)
    # the rank of body confidence in each frame (shape: T-1, M)
    rank = (-score[0:T - 1]).argsort(axis=1).reshape(T - 1, M)

    # data of frame 1
    xy1 = data_numpy[0:2, 0:T - 1, :, :].reshape(2, T - 1, V, M, 1)
    # data of frame 2
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    # square of distance between frame 1&2 (shape: T-1, M, M)
    distance = ((xy2 - xy1)**2).sum(axis=2).sum(axis=0)

    # match pose
    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = (rank == m)
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert (np.all(forward_map >= 0))

    # string data
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    # generate data
    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[
            t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # score sort
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    data_numpy = data_numpy[:, :, :, rank]

    return data_numpy


def top_k_by_category(label, score, top_k):
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for i in range(class_num)]
    for i in range(instance_num):
        l = label[i]
        hit_top_k[l].append(l in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label, score):
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision = []
    recall = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall


def joint_courruption(input_data):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()
    if flip_prob < 0.5:

        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(17, 6,replace=False)
        out[:,:,joint_indicies,:] = 0 
        return out
    
    else:
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
         joint_indicies = np.random.choice(17, 6,replace=False)
         
         temp = out[:,:,joint_indicies,:] 
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
         return out
     
     
def joint_masking(input_data):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()
    if flip_prob < 0.5:

        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(17, 6,replace=False)
        out[:,:,joint_indicies,:] = 0 
    return out
    
def random_noise(input_data):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()
    if flip_prob < 0.5:
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
         joint_indicies = np.random.choice(17, 6,replace=False)
         
         temp = out[:,:,joint_indicies,:] 
         Corruption = np.array([
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)],
                           [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)] ])
         temp = np.dot(temp.transpose([1, 2, 3, 0]), Corruption)
         temp = temp.transpose(3, 0, 1, 2)
         out[:,:,joint_indicies,:] = temp
    return out

     
     
def shear(data_numpy, s1=None, s2=None, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        if s1 != None:
            s1_list = s1
        else:
            s1_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]
        if s2 != None:
            s2_list = s2
        else:
            s2_list = [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]

        R = np.array([[1, s1_list[0], s2_list[0]],
                      [s1_list[1], 1, s2_list[1]],
                      [s1_list[2], s2_list[2], 1]])
        R = R.transpose()
        temp = np.dot(temp.transpose([1, 2, 3, 0]), R)
        temp = temp.transpose(3, 0, 1, 2)
        return temp.astype(np.float32)
    else:
        # print(type(data_numpy[0, 0, 0, 0]))
        return data_numpy.copy()
    
    
def gaussian_noise(data_numpy, mean=0, std=0.05, p=0.5):
    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        # print(type(noise[0, 0, 0, 0]))
        return (temp + noise).astype(np.float32)
    else:
        return data_numpy.copy()
    
    
def _rot(rot):
    """
    rot: T,3
    """
    cos_r, sin_r = rot.cos(), rot.sin()  # T,3
    zeros = torch.zeros(rot.shape[0], 1)  # T,1
    ones = torch.ones(rot.shape[0], 1)  # T,1

    r1 = torch.stack((ones, zeros, zeros),dim=-1)  # T,1,3
    rx2 = torch.stack((zeros, cos_r[:,0:1], sin_r[:,0:1]), dim = -1)  # T,1,3
    rx3 = torch.stack((zeros, -sin_r[:,0:1], cos_r[:,0:1]), dim = -1)  # T,1,3
    rx = torch.cat((r1, rx2, rx3), dim = 1)  # T,3,3

    ry1 = torch.stack((cos_r[:,1:2], zeros, -sin_r[:,1:2]), dim =-1)
    r2 = torch.stack((zeros, ones, zeros),dim=-1)
    ry3 = torch.stack((sin_r[:,1:2], zeros, cos_r[:,1:2]), dim =-1)
    ry = torch.cat((ry1, r2, ry3), dim = 1)

    rz1 = torch.stack((cos_r[:,2:3], sin_r[:,2:3], zeros), dim =-1)
    r3 = torch.stack((zeros, zeros, ones),dim=-1)
    rz2 = torch.stack((-sin_r[:,2:3], cos_r[:,2:3],zeros), dim =-1)
    rz = torch.cat((rz1, rz2, r3), dim = 1)

    rot = rz.matmul(ry).matmul(rx)
    return rot


def random_rot(data_numpy, theta=0.6):
    """
    data_numpy: C,T,V,M
    """
    data_torch = torch.from_numpy(data_numpy)
    C, T, V, M = data_torch.shape
    data_torch = data_torch.permute(1, 0, 2, 3).contiguous().view(T, C, V*M)  # T,3,V*M
    rot = torch.zeros(3).uniform_(-theta, theta)
    rot = torch.stack([rot, ] * T, dim=0)
    rot = _rot(rot)  # T,3,3
    data_torch = torch.matmul(rot, data_torch)
    data_torch = data_torch.view(T, C, V, M).permute(1, 0, 2, 3).contiguous()

    return data_torch.detach().numpy().copy()


def projection(data):
    """
    data_numpy: C,T,V,M
    """
    
    # プロジェクション行列を定義
    P_x = np.array([[0, 0, 0],
                    [0, 1, 0],
                    [0, 0, 1]])

    P_y = np.array([[1, 0, 0],
                    [0, 0, 0],
                    [0, 0, 1]])

    P_z = np.array([[1, 0, 0],
                    [0, 1, 0],
                    [0, 0, 0]])
    
    data = np.transpose(data, (3, 1, 2, 0))
    
    projection_matrices = [P_x, P_y, P_z]
    selected_projection_matrix = projection_matrices[random.choice([0, 1, 2])]
    # 各関節の座標にプロジェクションを適用
    data = np.dot(data, selected_projection_matrix.T)
    
    data = np.transpose(data, (3, 1, 2, 0))

    return data.astype(np.float32)


def random_translation(data, theta=0.05):
    import numpy as np

    # 平行移動の範囲を設定
    translation_range = [-theta, theta]

    data = np.transpose(data, (3, 1, 2, 0))
    
    # x軸およびz軸方向のランダムな平行移動値 Δ を生成
    delta_x = np.random.uniform(translation_range[0], translation_range[1])
    delta_z = np.random.uniform(translation_range[0], translation_range[1])

    # 平行移動ベクトル Δ を作成
    delta = np.array([delta_x, 0, delta_z])

    # 各関節の座標に平行移動を適用
    data = data + delta
    
    data = np.transpose(data, (3, 1, 2, 0))
    return data.astype(np.float32)


def random_scaling(data, scale_range=(0.8, 2.0)):

    data = np.transpose(data, (3, 1, 2, 0))
    m, t, v, c = data.shape
    
    # スケーリングファクターをランダムに生成
    scale_factors = np.random.uniform(scale_range[0], scale_range[1], size=(m, t, 1, 3))
    
    # スケーリングを適用
    data = data * scale_factors
    
    data = np.transpose(data, (3, 1, 2, 0))
    return data.astype(np.float32)


def translate_data(data, trans_cood):

    data = data - trans_cood
    return data


def random_scaling_2d(data, scale_range=(0.8, 2.0), left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12):
        
    data = np.transpose(data, (3, 1, 2, 0))
    m, t, v, c = data.shape
    
    for i in range(m):
        
            # キーポイントの中心をもと得る(両肩と両腰の平均)
        center = np.mean(data[i, 0, [left_shoulder, right_shoulder, left_hip, right_hip],:], axis=0)
        
        # 原点へ移動 
        skeleton = translate_data(data[i], center)
        
        # スケーリングファクターをランダムに生成
        scale_factors = np.random.uniform(scale_range[0], scale_range[1])
    
        # スケーリングを適用
        skeleton = skeleton * scale_factors
        
        data[i] = translate_data(skeleton, -center)
    
    data = np.transpose(data, (3, 1, 2, 0))
    return data.astype(np.float32)


def random_translation_2d(data, theta=255//4):

    data = np.transpose(data, (3, 1, 2, 0))
    m, t, v, c = data.shape
    
    # 平行移動の範囲を設定
    translation_range = [-theta, theta]
        
    # x軸およびy軸方向のランダムな平行移動値 Δ を生成
    delta_x = np.random.uniform(translation_range[0], translation_range[1])
    delta_y = np.random.uniform(translation_range[0], translation_range[1])

    # 平行移動ベクトル Δ を作成
    delta = np.array([delta_x, delta_y])

    # 各関節の座標に平行移動を適用
    data = data + delta
    
    data = np.transpose(data, (3, 1, 2, 0))
    return data.astype(np.float32)


def random_rotation_2d(data, theta=45, left_shoulder=5, right_shoulder=6, left_hip=11, right_hip=12):
    data = np.transpose(data, (3, 1, 2, 0))
    
    m, t, v, c = data.shape    
    
    # 回転角 Δ を生成
    delta = np.random.uniform(-theta, theta)
    
    # 角度をラジアンに変換
    angle_in_radians = math.radians(delta)
    
    
    for i in range(m):
        
        # キーポイントの中心をもと得る(両肩と両腰の平均)
        center = np.mean(data[i, 0, [left_shoulder, right_shoulder, left_hip, right_hip],:], axis=0)
        
        # 原点へ移動 
        skeleton = translate_data(data[i], center)
        
        # 座標を回転させる
        x_rotated = skeleton[..., 0] * math.cos(angle_in_radians) - skeleton[..., 1] * math.sin(angle_in_radians)
        y_rotated = skeleton[..., 0] * math.sin(angle_in_radians) + skeleton[..., 1] * math.cos(angle_in_radians)
        
        skeleton[..., 0] = x_rotated
        skeleton[..., 1] = y_rotated
        
        data[i] = translate_data(skeleton, -center)
        
    data = np.transpose(data, (3, 1, 2, 0))
    return data.astype(np.float32)


def gaussian_noise_2d(data_numpy, mean=0, std=3, p=0.5):

    if random.random() < p:
        temp = data_numpy.copy()
        C, T, V, M = data_numpy.shape
        noise = np.random.normal(mean, std, size=(C, T, V, M))
        # print(type(noise[0, 0, 0, 0]))
        return (temp + noise).astype(np.float32)
    else:
        return data_numpy.copy()
    
    
def joint_courruption_2d(input_data):                                                                     

    out = input_data.copy()

    flip_prob  = random.random()
    if flip_prob < 0.5:
        
        #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(17, 7,replace=False)
        out[:,:,joint_indicies,:] = 255 // 2
        return out
    
    else:

        c, t, v, m = input_data.shape
         #joint_indicies = np.random.choice(25, random.randint(5, 10), replace=False)
        joint_indicies = np.random.choice(17, 7,replace=False)
         
        temp = out[:,:,joint_indicies,:] 
        Corruption = np.random.randint(-255//4, 255//4, size=(c, t, 7, m))
        
        temp = Corruption + temp
        out[:,:,joint_indicies,:] = temp
        return out
    
    
def random_move_2d(data_numpy,
                angle_candidate=[-10., -5, 0., 5, 10.],
                scale_candidate=[0.9, 1, 1.1],
                transform_candidate=[-50, -25, 0.0, 25, 50],
                move_time_candidate=[1]):
    # input: C,T,V,M
    C, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)
    

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    
    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # linspace
    for i in range(num_node - 1):
        a[node[i]:node[i + 1]] = np.linspace(
            A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i]:node[i + 1]] = np.linspace(S[i], S[i + 1],
                                             node[i + 1] - node[i])
        t_x[node[i]:node[i + 1]] = np.linspace(T_x[i], T_x[i + 1],
                                               node[i + 1] - node[i])
        t_y[node[i]:node[i + 1]] = np.linspace(T_y[i], T_y[i + 1],
                                               node[i + 1] - node[i])

    theta = np.array([[np.cos(a) * s, -np.sin(a) * s],
                      [np.sin(a) * s, np.cos(a) * s]])

    # perform transformation
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy






