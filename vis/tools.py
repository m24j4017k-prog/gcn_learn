import imageio
import io
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

def coco_one_person_img(img, keypoints):
    
    orange = (18, 132, 239)
    green = (12, 213, 56)
    blue = (245, 151, 58)
    skeleton_links = [[0, 1], [0, 2], [1, 3], [2, 4], [0, 5], [0, 6], [5, 7], [6, 8],  [7, 9], [8, 10], [5,11],  [6, 12], [11, 13], [12, 14], [13, 15], [14, 16], [11, 12]]
    skeleton_links_colors = np.array([blue, blue, blue, blue, blue, blue, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, green])
    keypoint_colors = np.array([blue, blue, blue, blue, blue, green, orange, green, orange, green, orange, green, orange, green, orange, green, orange, blue])
    

    line_width = 2
    radius = 3


    for sk_id, sk in enumerate(skeleton_links):

        pos1 = (int(keypoints[sk[0], 0]), int(keypoints[sk[0], 1]))
        pos2 = (int(keypoints[sk[1], 0]), int(keypoints[sk[1], 1]))

        color = skeleton_links_colors[sk_id].tolist()
        cv2.line(img, pos1, pos2, color, thickness=line_width)

    for kid, kpt in enumerate(keypoints):

        x_coord, y_coord = int(kpt[0]), int(kpt[1])

        color = keypoint_colors[kid].tolist()
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    color, -1)
        cv2.circle(img, (int(x_coord), int(y_coord)), radius,
                    color)
            
    return img

# input 
def coco_one_person_video(t_keypoints, video_path, width=255, height=255, fps=30, label=None):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, (width, height))
    action_label = ['no talk', 'no active', 'active']
    for keypoints in (t_keypoints):
            
        img = np.ones((255, 255, 3), dtype=np.uint8) * 255
        
        if (label is not None):
            cv2.putText(img, action_label[label], (20, 20), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0,),1,cv2.LINE_AA)
        img = coco_one_person_img(img, keypoints)
        out.write(img)

    out.release()



def get_img_from_fig(fig, dpi=120):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    return img

def motion2video_3d(motion, save_path, fps=25, keep_imgs = False, view=[90, 180]):
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.view_init(elev=view[0], azim=view[1])
        # plt.tick_params(left = False, right = False , labelleft = False ,
        #                 labelbottom = False, bottom = False)
        
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(xs, ys, zs, color=color_left, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(xs, ys, zs, color=color_right, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            else:
                ax.plot(xs, ys, zs, color=color_mid, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()

import numpy as np
import random



def motion2video_3d2(motion, save_path, fps=2, keep_imgs = False, view=[90, 180]):
#     motion: (17,3,N)
    # input (T, V, C)
    # motion (V, C, T)
    motion = np.transpose(motion, (1, 2, 0))
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.view_init(elev=view[0], azim=view[1])
        plt.tick_params(left = True, right = True , labelleft = True ,
                        labelbottom = True, bottom = True)
        plt.xticks(np.arange(-1,1,0.2))

        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(-xs, -zs, -ys, color=color_left, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(-xs, -zs, -ys, color=color_right, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            else:
                ax.plot(-xs, -zs, -ys, color=color_mid, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()
    
    

def motion2video_3d3(motion, save_path, fps=25, keep_imgs = False, view=[90, 180]):
    motion = np.transpose(motion, (1, 2, 0))
    
#     motion: (17,3,N)
    videowriter = imageio.get_writer(save_path, fps=fps)
    vlen = motion.shape[-1]
    save_name = save_path.split('.')[0]
    frames = []
    joint_pairs = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [8, 11], [8, 14], [9, 10], [11, 12], [12, 13], [14, 15], [15, 16]]
    joint_pairs_left = [[8, 11], [11, 12], [12, 13], [0, 4], [4, 5], [5, 6]]
    joint_pairs_right = [[8, 14], [14, 15], [15, 16], [0, 1], [1, 2], [2, 3]]
    
    color_mid = "#00457E"
    color_left = "#02315E"
    color_right = "#2F70AF"
    for f in tqdm(range(vlen)):
        j3d = motion[:,:,f]
        fig = plt.figure(0, figsize=(10, 10))
        ax = plt.axes(projection="3d")
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_zlim(-1, 1)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.view_init(elev=view[0], azim=view[1])
        # plt.tick_params(left = False, right = False , labelleft = False ,
        #                 labelbottom = False, bottom = False)
        
        for i in range(len(joint_pairs)):
            limb = joint_pairs[i]
            xs, ys, zs = [np.array([j3d[limb[0], j], j3d[limb[1], j]]) for j in range(3)]
            if joint_pairs[i] in joint_pairs_left:
                ax.plot(xs, ys, zs, color=color_left, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            elif joint_pairs[i] in joint_pairs_right:
                ax.plot(xs, ys, zs, color=color_right, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            else:
                ax.plot(xs, ys, zs, color=color_mid, lw=5, marker='o', markerfacecolor='w', markersize=12, markeredgewidth=4) # axis transformation for visualization
            
        frame_vis = get_img_from_fig(fig)
        videowriter.append_data(frame_vis)
        plt.close()
    videowriter.close()
    
def random_move(data_numpy,
                angle_candidate=[-100., -50., 0., 50., 100.],
                scale_candidate=[0.2, 1.0, 3.1],
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


if __name__ == '__main__':
    data = np.load("data/VITPOSE/1/data.npy")
    print(data.shape)
    data = np.transpose(data, (0, 4, 2, 3, 1))
    img = np.ones((255, 255, 3), dtype=np.uint8) * 255
    img = coco_one_person_video(data[0, 1], "a.mp4")
