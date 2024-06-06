import train
import os
import time
import csv
import sys
import warnings
import random
import numpy as np
import time
import pprint
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP
from loguru import logger
import smplx

from utils import config, logger_tools, other_tools, metric
from utils import rotation_conversions as rc
from dataloaders import data_tools
from optimizers.optim_factory import create_optimizer
from optimizers.scheduler_factory import create_scheduler
from optimizers.loss_factory import get_loss_func
from scipy.spatial.transform import Rotation
from dataloaders.data_tools import joints_list

class CustomTrainer(train.BaseTrainer):
    """
    motion representation learning
    """
    def __init__(self, args):
        super().__init__(args)
        self.joints = self.train_data.joints
        self.smplx = smplx.create(
            self.args.data_path_1+"smplx_models/", 
            model_type='smplx',
            gender='NEUTRAL_2020', 
            use_face_contour=False,
            num_betas=300,
            num_expression_coeffs=100, 
            ext='npz',
            use_pca=False,
        ).cuda().eval()
        self.tracker = other_tools.EpochTracker(["rec", "vel", "ver", "com", "kl", "acc"], [False, False, False, False, False, False])
        if not self.args.rot6d: #"rot6d" not in args.pose_rep:
            logger.error(f"this script is for rot6d, your pose rep. is {args.pose_rep}")
        self.rec_loss = get_loss_func("GeodesicLoss")
        self.vel_loss = torch.nn.L1Loss(reduction='mean')
        self.vectices_loss = torch.nn.MSELoss(reduction='mean')
        
        
        vq_model_module = __import__(f"models.{args.model}", fromlist=["something"])
        vq_name = args.g_name
        # self.args.vae_layer = 2
        # self.args.vae_length = 256
        self.args.pose_dims = 180
        # self.vq_model_hands = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
        # # print(self.vq_model_face)
        # other_tools.load_checkpoints(self.vq_model_hands, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0118_113544_myvqvae_hands/rec.bin", "vq_model_hands")
        # self.args.pose_dims = 84
        # self.vq_model_upper = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_upper, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0118_113637_myvqvae_upper/rec.bin", "vq_model_upper")
        # self.args.pose_dims = 54
        # self.vq_model_lower = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
        # other_tools.load_checkpoints(self.vq_model_lower, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0118_144700_myvqvae_lower/rec.bin", "vq_model_lower")
        
        if args.model == 'motion_representation':
            self.args.vae_length = 240
            self.args.vae_test_dim = 180
            self.vq_model_hands = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
            # print(self.vq_model_face)
            other_tools.load_checkpoints(self.vq_model_hands, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0119_182938_vqvae_hands/rec.bin", "vq_model_hands")
            self.args.vae_length = 240
            self.args.vae_test_dim = 84
            self.vq_model_upper = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
            other_tools.load_checkpoints(self.vq_model_upper, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0119_230021_vqvae_upper/rec.bin", "vq_model_upper")
            self.args.vae_test_dim = 54
            self.vq_model_lower = getattr(vq_model_module, vq_name)(self.args).to(self.rank)
            other_tools.load_checkpoints(self.vq_model_lower, "/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/outputs/audio2pose/custom/0119_183006_vqvae_lower/rec.bin", "vq_model_lower")
        
            
            
        
        
        self.tar_joint_list_upper = joints_list["my_beat_smplx_upper"]
        self.tar_joint_list_hands = joints_list["my_beat_smplx_hands"]
        self.tar_joint_list_lower = joints_list["my_beat_smplx_lower"]
        
        self.ori_joint_list = joints_list[self.args.ori_joints]
        self.joint_mask_upper = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_upper:
            self.joint_mask_upper[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_hands = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_hands:
            self.joint_mask_hands[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        self.joint_mask_lower = np.zeros(len(list(self.ori_joint_list.keys()))*3)
        for joint_name in self.tar_joint_list_lower:
            self.joint_mask_lower[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
            
        ## 上面是表征方式是三维的位置标号，下面把三维的换成六维
        self.joint_mask_upper_6d = self.joint_mask_upper.repeat(2)
        self.joint_mask_hands_6d = self.joint_mask_hands.repeat(2)
        self.joint_mask_lower_6d = self.joint_mask_lower.repeat(2)
        
        
        
    
    def inverse_selection(self, filtered_t, selection_array, n):
        # 创建一个全为零的数组，形状为 n*165
        original_shape_t = np.zeros((n, selection_array.size))
        
        # 找到选择数组中为1的索引位置
        selected_indices = np.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t
    
    def inverse_selection_tensor(self, filtered_t, selection_array, n):
    # 创建一个全为零的数组，形状为 n*165
        selection_array = torch.from_numpy(selection_array).cuda()
        original_shape_t = torch.zeros((n, 165)).cuda()
        
        # 找到选择数组中为1的索引位置
        selected_indices = torch.where(selection_array == 1)[0]
        
        # 将 filtered_t 的值填充到 original_shape_t 中相应的位置
        for i in range(n):
            original_shape_t[i, selected_indices] = filtered_t[i]
            
        return original_shape_t

    def vis_tsne(self):
        import numpy as np
        import matplotlib.pyplot as plt
        from sklearn.manifold import TSNE
        self.model.eval()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data['pose']
            vid = dict_data['vid'][...,0,0]
            b,s,c = tar_pose.shape    # batch , seq , channel

            tar_pose = tar_pose.reshape(b,-1)
            # example data, randomly generated
            data = tar_pose.cpu().numpy()   # 100 samples, 10 features
            # t-SNE
            plot = '2d'
            if plot == '2d':
                plt.figure()
                # tsne = TSNE(n_components=2, perplexity=30, learning_rate=200)
                # #tsne = PCA(n_components=2)
                # reduced_data = tsne.fit_transform(data)
                # for i in range(b):
                #     if vid[i].item()==99 :
                #         plt.scatter(reduced_data[i, 0], reduced_data[i, 1],color="r",alpha = 0.1)
                #     else :    
                #         plt.scatter(reduced_data[i, 0], reduced_data[i, 1],color="b",alpha = 0.1)
                #         if reduced_data[i, 0]>10:
                #             print(f"find:{i}")
                # plt.title("t-SNE of example data")
                # plt.xlabel("t-SNE component 1")
                # plt.ylabel("t-SNE component 2")
                # plt.show()
                # plt.savefig('before_vq.png')
                
                
                
                plt.clf()
                plt.cla()
                plt.close('all')
                
                plt.figure()
                tar_pose = dict_data['pose']
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(b, s, -1, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(b, s, -1)
                tar_pose_upper = tar_pose[...,self.joint_mask_upper_6d.astype(bool)].cuda()
                tar_pose_lower = tar_pose[...,self.joint_mask_lower_6d.astype(bool)].cuda()
                tar_pose_hands = tar_pose[...,self.joint_mask_hands_6d.astype(bool)].cuda()
                
                rec_pose_upper = self.vq_model_upper(tar_pose_upper)['rec_pose']
                rec_pose_lower = self.vq_model_lower(tar_pose_lower)['rec_pose']
                rec_pose_hands = self.vq_model_hands(tar_pose_hands)['rec_pose']
                
                rec_pose_upper = rec_pose_upper.reshape(b, s, 14, 6) 
                rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
                rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(b*s, 14*3)
        
                rec_pose_lower = rec_pose_lower.reshape(b, s, 9, 6) 
                rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
                rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(b*s, 9*3)
                
                rec_pose_hands = rec_pose_hands.reshape(b, s, 30, 6) 
                rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
                rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(b*s, 30*3)
                
                rec_pose_upper = rec_pose_upper.detach().cpu().numpy()
                rec_pose_lower = rec_pose_lower.detach().cpu().numpy()
                rec_pose_hands = rec_pose_hands.detach().cpu().numpy()
                
                rec_pose_upper = self.inverse_selection(rec_pose_upper, self.joint_mask_upper, rec_pose_upper.shape[0])
                rec_pose_lower = self.inverse_selection(rec_pose_lower, self.joint_mask_lower, rec_pose_lower.shape[0])
                rec_pose_hands = self.inverse_selection(rec_pose_hands, self.joint_mask_hands, rec_pose_hands.shape[0])
                
                rec_pose = rec_pose_upper + rec_pose_lower + rec_pose_hands
                
                tsne = TSNE(n_components=2, perplexity=300, learning_rate=200)
                #tsne = PCA(n_components=2)
                reduced_data = tsne.fit_transform(rec_pose)
                for i in range(b):
                    if vid[i].item()==99 :
                        plt.scatter(reduced_data[i, 0], reduced_data[i, 1],color="r",alpha = 0.1)
                    else :    
                        plt.scatter(reduced_data[i, 0], reduced_data[i, 1],color="b",alpha = 0.1)
                        if reduced_data[i, 0]>10:
                            print(f"find:{i}")
                plt.title("t-SNE of example data")
                plt.xlabel("t-SNE component 1")
                plt.ylabel("t-SNE component 2")
                plt.show()
                plt.savefig('after_vq.png')
            
            
            
            

    def rotate_character1(): # smplx 2 beat
        a = np.load('/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/datasets/AMASS_SMPLX/TotalCapture/s1/acting1_stageii.npz',allow_pickle=True)
        data_dict = {key: a[key] for key in a}
        b = data_dict['poses'][...,:3]
        b = rc.axis_angle_to_matrix(torch.from_numpy(b))
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, 1.0], [0.0, -1.0, 0.0]])
        c = np.einsum('ij,kjl->kil',rot_matrix,b)
        c = rc.matrix_to_axis_angle(torch.from_numpy(c))
        data_dict['poses'][...,:3] = c
        
        trans_matrix1 = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, -1.0], [0.0, 1.0, 0.0]])
        #trans_matrix2 = np.array([[1.0, 0.0, 0.0], [0.0 , -1.0, 0.0], [0.0, 0.0, -1.0]])
        #data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix)
        data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix1)
        #data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix2)
        np.savez('1232.npz', **data_dict)

        np.savez('1234.npz',
            betas=data_dict["betas"],
            poses=data_dict['poses'],
            #expressions=data_dict["expressions"]-data_dict["expressions"],
            trans=data_dict["trans"],
            model='smplx2020',
            gender='male',
            mocap_frame_rate = 30 ,
        )

    def rotate_character2(): # beat 2 smplx 
        a = np.load('/mnt/fu09a/chenbohong/PantoMatrix/scripts/EMAGE_2024/datasets/BEAT_SMPL/beat_v2.0.0/beat_english_v2.0.0/smplxflame_30/1_wayne_0_1_1.npz',allow_pickle=True)
        data_dict = {key: a[key] for key in a}
        b = data_dict['poses'][...,:3]
        b = rc.axis_angle_to_matrix(torch.from_numpy(b))
        rot_matrix = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, -1.0], [0.0, 1.0, 0.0]])
        c = np.einsum('ij,kjl->kil',rot_matrix,b)
        c = rc.matrix_to_axis_angle(torch.from_numpy(c))
        data_dict['poses'][...,:3] = c
        
        trans_matrix1 = np.array([[1.0, 0.0, 0.0], [0.0 , 0.0, 1.0], [0.0, -1.0, 0.0]])
        #trans_matrix2 = np.array([[1.0, 0.0, 0.0], [0.0 , -1.0, 0.0], [0.0, 0.0, -1.0]])
        #data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix)
        data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix1)
        #data_dict['trans'] = np.einsum("bi,ij->bj",data_dict['trans'],trans_matrix2)
        np.savez('1232.npz', **data_dict)

        np.savez('1234.npz',
            betas=data_dict["betas"],
            poses=data_dict['poses'],
            #expressions=data_dict["expressions"]-data_dict["expressions"],
            trans=data_dict["trans"],
            model='smplx2020',
            gender='male',
            mocap_frame_rate = 30 ,
        )




    def train(self, epoch):
        self.vis_tsne()
        self.model.train()
        t_start = time.time()
        self.tracker.reset()
        for its, dict_data in enumerate(self.train_loader):
            tar_pose = dict_data["pose"][:, :, :165]
            #tar_beta = dict_data["beta"].cuda()
            tar_trans = dict_data["trans"].cuda()
            vid = dict_data['vid'][:,0,0]
            #self.vis_tsne(tar_pose,vid)
            tar_pose = tar_pose.cuda()  
            bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
            tar_exps = torch.zeros((bs, n, 100)).cuda()
            tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
            t_data = time.time() - t_start 
            
            self.opt.zero_grad()
            g_loss_final = 0
            net_out = self.model(tar_pose)
            rec_pose = net_out["rec_pose"]
            rec_pose = rec_pose.reshape(bs, n, j, 6)
            rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
            tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
            loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
            self.tracker.update_meter("rec", "train", loss_rec.item())
            g_loss_final += loss_rec

            velocity_loss =  self.vel_loss(rec_pose[:, 1:] - rec_pose[:, :-1], tar_pose[:, 1:] - tar_pose[:, :-1]) * self.args.rec_weight
            acceleration_loss =  self.vel_loss(rec_pose[:, 2:] + rec_pose[:, :-2] - 2 * rec_pose[:, 1:-1], tar_pose[:, 2:] + tar_pose[:, :-2] - 2 * tar_pose[:, 1:-1]) * self.args.rec_weight
            self.tracker.update_meter("vel", "train", velocity_loss.item())
            self.tracker.update_meter("acc", "train", acceleration_loss.item())
            g_loss_final += velocity_loss 
            g_loss_final += acceleration_loss 
             # vertices loss
            if self.args.rec_ver_weight > 0:
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                vertices_rec = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=rec_pose[:, 66:69], 
                    global_orient=rec_pose[:,:3], 
                    body_pose=rec_pose[:,3:21*3+3], 
                    left_hand_pose=rec_pose[:,25*3:40*3], 
                    right_hand_pose=rec_pose[:,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )
                vertices_tar = self.smplx(
                    betas=tar_beta.reshape(bs*n, 300), 
                    transl=tar_trans.reshape(bs*n, 3), 
                    expression=tar_exps.reshape(bs*n, 100), 
                    jaw_pose=tar_pose[:, 66:69], 
                    global_orient=tar_pose[:,:3], 
                    body_pose=tar_pose[:,3:21*3+3], 
                    left_hand_pose=tar_pose[:,25*3:40*3], 
                    right_hand_pose=tar_pose[:,40*3:55*3], 
                    return_verts=True,
                    return_joints=True,
                    leye_pose=tar_pose[:, 69:72], 
                    reye_pose=tar_pose[:, 72:75],
                )  
                vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                self.tracker.update_meter("ver", "train", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight

                vertices_vel_loss = self.vel_loss(vertices_rec['vertices'][:, 1:] - vertices_rec['vertices'][:, :-1], vertices_tar['vertices'][:, 1:] - vertices_tar['vertices'][:, :-1]) * self.args.rec_weight
                vertices_acc_loss = self.vel_loss(vertices_rec['vertices'][:, 2:] + vertices_rec['vertices'][:, :-2] - 2 * vertices_rec['vertices'][:, 1:-1], vertices_tar['vertices'][:, 2:] + vertices_tar['vertices'][:, :-2] - 2 * vertices_tar['vertices'][:, 1:-1]) * self.args.rec_weight
                g_loss_final += vertices_vel_loss * self.args.rec_weight * self.args.rec_ver_weight
                g_loss_final += vertices_acc_loss * self.args.rec_weight * self.args.rec_ver_weight 
            
            # if self.args.vel_weight > 0:  
            #     pos_rec_vel = other_tools.estimate_linear_velocity(vertices_rec['joints'], 1/self.pose_fps)
            #     pos_tar_vel = other_tools.estimate_linear_velocity(vertices_tar['joints'], 1/self.pose_fps)
            #     vel_rec_loss = self.vel_loss(pos_rec_vel, pos_tar_vel)
            #     tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
            #     rec_pose = rc.axis_angle_to_matrix(rec_pose.reshape(bs, n, j, 3))
            #     rot_rec_vel = other_tools.estimate_angular_velocity(rec_pose, 1/self.pose_fps)
            #     rot_tar_vel = other_tools.estimate_angular_velocity(tar_pose, 1/self.pose_fps)
            #     vel_rec_loss += self.vel_loss(pos_rec_vel, pos_tar_vel)
            #     self.tracker.update_meter("vel", "train", vel_rec_loss.item()*self.args.vel_weight)
            #     loss += (vel_rec_loss*self.args.vel_weight)

            # ---------------------- vae -------------------------- #
            if "VQVAE" in self.args.g_name:
                loss_embedding = net_out["embedding_loss"]
                g_loss_final += loss_embedding
                self.tracker.update_meter("com", "train", loss_embedding.item())
            # elif "VAE" in self.args.g_name:
            #     pose_mu, pose_logvar = net_out["pose_mu"], net_out["pose_logvar"] 
            #     KLD = -0.5 * torch.sum(1 + pose_logvar - pose_mu.pow(2) - pose_logvar.exp())
            #     if epoch < 0:
            #         KLD_weight = 0
            #     else:
            #         KLD_weight = min(1.0, (epoch - 0) * 0.05) * 0.01
            #     loss += KLD_weight * KLD
            #     self.tracker.update_meter("kl", "train", KLD_weight * KLD.item())    
            g_loss_final.backward()
            if self.args.grad_norm != 0: 
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.grad_norm)
            self.opt.step()
            t_train = time.time() - t_start - t_data
            t_start = time.time()
            mem_cost = torch.cuda.memory_cached() / 1E9
            lr_g = self.opt.param_groups[0]['lr']
            if its % self.args.log_period == 0:
                self.train_recording(epoch, its, t_data, t_train, mem_cost, lr_g)   
            if self.args.debug:
                if its == 1: break
        self.opt_s.step(epoch)
                    
    def val(self, epoch):
        self.model.eval()
        t_start = time.time()
        with torch.no_grad():
            for its, dict_data in enumerate(self.val_loader):
                tar_pose = dict_data["pose"][:, :, :165]
                #tar_beta = dict_data["beta"].cuda()
                tar_trans = dict_data["trans"].cuda()
                tar_pose = tar_pose.cuda()  
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints
                tar_exps = torch.zeros((bs, n, 100)).cuda()
                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                t_data = time.time() - t_start 

                #self.opt.zero_grad()
                #g_loss_final = 0
                net_out = self.model(tar_pose)
                rec_pose = net_out["rec_pose"]
                rec_pose = rec_pose.reshape(bs, n, j, 6)
                rec_pose = rc.rotation_6d_to_matrix(rec_pose)#
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                loss_rec = self.rec_loss(rec_pose, tar_pose) * self.args.rec_weight * self.args.rec_pos_weight
                self.tracker.update_meter("rec", "val", loss_rec.item())
                #g_loss_final += loss_rec

                 # vertices loss
                if self.args.rec_ver_weight > 0:
                    tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                    rec_pose = rc.matrix_to_axis_angle(rec_pose).reshape(bs*n, j*3)
                    rec_pose = self.inverse_selection_tensor(rec_pose, self.train_data.joint_mask, rec_pose.shape[0])
                    tar_pose = self.inverse_selection_tensor(tar_pose, self.train_data.joint_mask, tar_pose.shape[0])
                    vertices_rec = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=rec_pose[:, 66:69], 
                        global_orient=rec_pose[:,:3], 
                        body_pose=rec_pose[:,3:21*3+3], 
                        left_hand_pose=rec_pose[:,25*3:40*3], 
                        right_hand_pose=rec_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )
                    vertices_tar = self.smplx(
                        betas=tar_beta.reshape(bs*n, 300), 
                        transl=tar_trans.reshape(bs*n, 3), 
                        expression=tar_exps.reshape(bs*n, 100), 
                        jaw_pose=tar_pose[:, 66:69], 
                        global_orient=tar_pose[:,:3], 
                        body_pose=tar_pose[:,3:21*3+3], 
                        left_hand_pose=tar_pose[:,25*3:40*3], 
                        right_hand_pose=tar_pose[:,40*3:55*3], 
                        return_verts=True, 
                        leye_pose=tar_pose[:, 69:72], 
                        reye_pose=tar_pose[:, 72:75],
                    )  
                    vectices_loss = self.vectices_loss(vertices_rec['vertices'], vertices_tar['vertices'])
                    self.tracker.update_meter("ver", "val", vectices_loss.item()*self.args.rec_weight * self.args.rec_ver_weight)
                if "VQVAE" in self.args.g_name:
                    loss_embedding = net_out["embedding_loss"]
                    self.tracker.update_meter("com", "val", loss_embedding.item())
                    #g_loss_final += vectices_loss*self.args.rec_weight*self.args.rec_ver_weight
            self.val_recording(epoch)
            
    def test(self, epoch):

        results_save_path = self.checkpoint_path + f"/{epoch}/"
        if not os.path.exists(results_save_path): 
            os.makedirs(results_save_path)
        start_time = time.time()
        total_length = 0
        test_seq_list = self.test_data.selected_file
        self.model.eval()
        with torch.no_grad():
            for its, dict_data in enumerate(self.test_loader):
                tar_pose = dict_data["pose"][:, :, :165]
                self.vis_tsne(dict_data)
                tar_pose = tar_pose.cuda()
                bs, n, j = tar_pose.shape[0], tar_pose.shape[1], self.joints

                tar_pose = rc.axis_angle_to_matrix(tar_pose.reshape(bs, n, j, 3))
                tar_pose = rc.matrix_to_rotation_6d(tar_pose).reshape(bs, n, j*6)
                remain = n%self.args.pose_length
                tar_pose = tar_pose[:, :n-remain, :]
                n = n - remain
                if n < 8:
                    print("error")
                    continue
                #print(tar_pose.shape)
                if True:
                    tar_pose_upper = tar_pose[...,self.joint_mask_upper_6d.astype(bool)]
                    tar_pose_lower = tar_pose[...,self.joint_mask_lower_6d.astype(bool)]
                    tar_pose_hands = tar_pose[...,self.joint_mask_hands_6d.astype(bool)]
                    
                    rec_pose_upper = self.vq_model_upper(tar_pose_upper)['rec_pose']

                    rec_pose_lower = self.vq_model_lower(tar_pose_lower)['rec_pose']
                    rec_pose_hands = self.vq_model_hands(tar_pose_hands)['rec_pose']
                    
                    
                    rec_pose_upper = rec_pose_upper.reshape(bs, n, 14, 6) 
                    rec_pose_upper = rc.rotation_6d_to_matrix(rec_pose_upper)
                    rec_pose_upper = rc.matrix_to_axis_angle(rec_pose_upper).reshape(bs*n, 14*3)
            
                    rec_pose_lower = rec_pose_lower.reshape(bs, n, 9, 6) 
                    rec_pose_lower = rc.rotation_6d_to_matrix(rec_pose_lower)
                    rec_pose_lower = rc.matrix_to_axis_angle(rec_pose_lower).reshape(bs*n, 9*3)
                    
                    rec_pose_hands = rec_pose_hands.reshape(bs, n, 30, 6) 
                    rec_pose_hands = rc.rotation_6d_to_matrix(rec_pose_hands)
                    rec_pose_hands = rc.matrix_to_axis_angle(rec_pose_hands).reshape(bs*n, 30*3)
                    
                    
                    rec_pose_upper = rec_pose_upper.cpu().numpy()
                    rec_pose_lower = rec_pose_lower.cpu().numpy()
                    rec_pose_hands = rec_pose_hands.cpu().numpy()
                    
                    rec_pose_upper = self.inverse_selection(rec_pose_upper, self.joint_mask_upper, rec_pose_upper.shape[0])
                    rec_pose_lower = self.inverse_selection(rec_pose_lower, self.joint_mask_lower, rec_pose_lower.shape[0])
                    rec_pose_hands = self.inverse_selection(rec_pose_hands, self.joint_mask_hands, rec_pose_hands.shape[0])
                    
                    rec_pose = rec_pose_upper + rec_pose_lower + rec_pose_hands
                    
                            
                tar_pose = rc.rotation_6d_to_matrix(tar_pose.reshape(bs, n, j, 6))
                tar_pose = rc.matrix_to_axis_angle(tar_pose).reshape(bs*n, j*3)
                tar_pose = tar_pose.cpu().numpy()
                
                total_length += n 
                # --- save --- #
                if 'smplx' in self.args.pose_rep:
                    gt_npz = np.load(self.args.data_path+self.args.pose_rep+"/"+test_seq_list.iloc[0]['id']+'.npz', allow_pickle=True)
                    stride = int(30 / self.args.pose_fps)
                    tar_pose = self.inverse_selection(tar_pose, self.test_data.joint_mask, tar_pose.shape[0])
                    np.savez(results_save_path+"gt_"+str(its)+'.npz',
                        betas=gt_npz["betas"],
                        poses=tar_pose[:n],
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=gt_npz["trans"][::stride][:n] - gt_npz["trans"][::stride][:n],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )
                    #rec_pose = self.inverse_selection(rec_pose, self.test_data.joint_mask, rec_pose.shape[0])
                    np.savez(results_save_path+"res_"+str(its)+'.npz',
                        betas=gt_npz["betas"],
                        poses=rec_pose,
                        expressions=gt_npz["expressions"]-gt_npz["expressions"],
                        trans=gt_npz["trans"][::stride][:n] - gt_npz["trans"][::stride][:n],
                        model='smplx2020',
                        gender='neutral',
                        mocap_frame_rate = 30 ,
                    )       

        
        
        end_time = time.time() - start_time
        logger.info(f"total inference time: {int(end_time)} s for {int(total_length/self.args.pose_fps)} s motion")