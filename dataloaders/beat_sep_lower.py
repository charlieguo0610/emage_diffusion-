import os
import pickle
import math
import shutil
import numpy as np
import lmdb as lmdb
import textgrid as tg
import pandas as pd
import torch
import glob
import json
from termcolor import colored
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import torch.distributed as dist
#import pyarrow
import pickle
import librosa
import smplx

from .build_vocab import Vocab
from .utils.audio_features import Wav2Vec2Model
from .data_tools import joints_list
from .utils import rotation_conversions as rc
from .utils import other_tools


# for fastspeech2
import tgt
import pyworld as pw
import sys

import yaml
sys.path.insert(0, '/scratch/year/zixin/2024/emage-diffusion')

from os.path import join
from scipy.interpolate import interp1d
###########
import audio as Audio
def init_for_fastspeech2():
    print("Initializing STFT...")
    # initialize for fastspeech2
    config_root = '/scratch/year/zixin/2024/FastSpeech2/config/en_16_config'
    preprocess_config_path = join(config_root, 'preprocess.yaml')
    preprocess_config = yaml.load(open(preprocess_config_path, "r"), Loader=yaml.FullLoader)
   
    _STFT = Audio.stft.TacotronSTFT(
                preprocess_config["preprocessing"]["stft"]["filter_length"],
                preprocess_config["preprocessing"]["stft"]["hop_length"],
                preprocess_config["preprocessing"]["stft"]["win_length"],
                preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
                preprocess_config["preprocessing"]["audio"]["sampling_rate"],
                preprocess_config["preprocessing"]["mel"]["mel_fmin"],
                preprocess_config["preprocessing"]["mel"]["mel_fmax"],
        )
    return _STFT
# ###########
_STFT = init_for_fastspeech2()
# _STFT = None


class CustomDataset(Dataset):
    def __init__(self, args, loader_type, augmentation=None, kwargs=None, build_cache=True):
        
        self.args = args
        self.loader_type = loader_type # train

        self.rank = dist.get_rank()
        self.ori_stride = self.args.stride # 20
        self.ori_length = self.args.pose_length # 64
        self.alignment = [0,0] # for trinity
        
        self.ori_joint_list = joints_list[self.args.ori_joints] # 'beat_smplx_joints'
        self.tar_joint_list = joints_list[self.args.tar_joints] # 'beat_smplx_full'
        if 'smplx' in self.args.pose_rep:
            self.joint_mask = np.zeros(len(list(self.ori_joint_list.keys()))*3)
            self.joints = len(list(self.tar_joint_list.keys()))  
            for joint_name in self.tar_joint_list:
                self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        else:
            self.joints = len(list(self.ori_joint_list.keys()))+1
            self.joint_mask = np.zeros(self.joints*3)
            for joint_name in self.tar_joint_list:
                if joint_name == "Hips":
                    self.joint_mask[3:6] = 1
                else:
                    self.joint_mask[self.ori_joint_list[joint_name][1] - self.ori_joint_list[joint_name][0]:self.ori_joint_list[joint_name][1]] = 1
        # select trainable joints
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

        split_rule = pd.read_csv(args.data_path+"train_test_split.csv")
        self.selected_file = split_rule.loc[(split_rule['type'] == loader_type) & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
        if args.additional_data and loader_type == 'train':
            split_b = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            #self.selected_file = split_rule.loc[(split_rule['type'] == 'additional') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = pd.concat([self.selected_file, split_b])
        if self.selected_file.empty:
            logger.warning(f"{loader_type} is empty for speaker {self.args.training_speakers}, use train set 0-8 instead")
            self.selected_file = split_rule.loc[(split_rule['type'] == 'train') & (split_rule['id'].str.split("_").str[0].astype(int).isin(self.args.training_speakers))]
            self.selected_file = self.selected_file.iloc[0:8]
        self.data_dir = args.data_path 
        
        if loader_type == "test": 
            self.args.multi_length_training = [1.0]
        self.max_length = int(args.pose_length * self.args.multi_length_training[-1])
        self.max_audio_pre_len = math.floor(args.pose_length / args.pose_fps * self.args.audio_sr)
        if self.max_audio_pre_len > self.args.test_length*self.args.audio_sr: 
            self.max_audio_pre_len = self.args.test_length*self.args.audio_sr
        
        if args.word_rep is not None:
            with open(f"{args.data_path}weights/vocab.pkl", 'rb') as f:
                self.lang_model = pickle.load(f)
                
        preloaded_dir = self.args.root_path + self.args.cache_path + loader_type + f"/{args.pose_rep}_cache"      
        # if args.pose_norm:
        #     # careful for rotation vectors
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_pose()
        #     self.mean_pose = np.load(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy")
        #     self.std_pose = np.load(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_std.npy")
        # if args.audio_norm:
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_audio()
        #     self.mean_audio = np.load(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/npy_mean.npy")
        #     self.std_audio = np.load(args.data_path+args.mean_pose_path+f"{args.audio_rep.split('_')[0]}/npy_std.npy")
        # if args.facial_norm:
        #     if not os.path.exists(args.data_path+args.mean_pose_path+f"{args.pose_rep.split('_')[0]}/bvh_mean.npy"):
        #         self.calculate_mean_face()
        #     self.mean_facial = np.load(args.data_path+args.mean_pose_path+f"{args.facial_rep}/json_mean.npy")
        #     self.std_facial = np.load(args.data_path+args.mean_pose_path+f"{args.facial_rep}/json_std.npy")
        if self.args.beat_align:
            if not os.path.exists(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy"):
                self.calculate_mean_velocity(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            self.avg_vel = np.load(args.data_path+f"weights/mean_vel_{args.pose_rep}.npy")
            
        if build_cache and self.rank == 0:
            self.build_cache(preloaded_dir)
        self.lmdb_env = lmdb.open(preloaded_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"] 

    
    def calculate_mean_velocity(self, save_path):
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
        dir_p = self.data_dir + self.args.pose_rep + "/"
        all_list = []
        from tqdm import tqdm
        for tar in tqdm(os.listdir(dir_p)):
            if tar.endswith(".npz"):
                m_data = np.load(dir_p+tar, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, :55, :].reshape(max_length, 55*3)
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, :55, :].reshape(r, 55*3)
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0)
                joints = joints.permute(1, 0)
                dt = 1/30
            # first steps is forward diff (t+1 - t) / dt
                init_vel = (joints[:, 1:2] - joints[:, :1]) / dt
                # middle steps are second order (t+1 - t-1) / 2dt
                middle_vel = (joints[:, 2:] - joints[:, 0:-2]) / (2 * dt)
                # last step is backward diff (t - t-1) / dt
                final_vel = (joints[:, -1:] - joints[:, -2:-1]) / dt
                #print(joints.shape, init_vel.shape, middle_vel.shape, final_vel.shape)
                vel_seq = torch.cat([init_vel, middle_vel, final_vel], dim=1).permute(1, 0).reshape(n, 55, 3)
                #print(vel_seq.shape)
                #.permute(1, 0).reshape(n, 55, 3)
                vel_seq_np = vel_seq.cpu().numpy()
                vel_joints_np = np.linalg.norm(vel_seq_np, axis=2) # n * 55
                all_list.append(vel_joints_np)
        avg_vel = np.mean(np.concatenate(all_list, axis=0),axis=0) # 55
        np.save(save_path, avg_vel)
        
    
    def build_cache(self, preloaded_dir):
        logger.info(f"Audio bit rate: {self.args.audio_fps}")
        logger.info("Reading data '{}'...".format(self.data_dir))
        logger.info("Creating the dataset cache...")
        if self.args.new_cache:
            if os.path.exists(preloaded_dir):
                shutil.rmtree(preloaded_dir)
        if os.path.exists(preloaded_dir):
            logger.info("Found the cache {}".format(preloaded_dir))
        elif self.loader_type == "test":
            self.cache_generation(
                preloaded_dir, True, 
                0, 0,
                is_test=True)
        else:
            self.cache_generation(
                preloaded_dir, self.args.disable_filtering, 
                self.args.clean_first_seconds, self.args.clean_final_seconds,
                is_test=False)
        
    def __len__(self):
        return self.n_samples
    

    def cache_generation(self, out_lmdb_dir, disable_filtering, clean_first_seconds,  clean_final_seconds, is_test=False):
        # if "wav2vec2" in self.args.audio_rep:
        #     self.wav2vec_model = Wav2Vec2Model.from_pretrained(f"{self.args.data_path_1}/hub/transformer/wav2vec2-base-960h")
        #     self.wav2vec_model.feature_extractor._freeze_parameters()
        #     self.wav2vec_model = self.wav2vec_model.cuda()
        #     self.wav2vec_model.eval()
        
        self.n_out_samples = 0
        # create db for samples
        if not os.path.exists(out_lmdb_dir): os.makedirs(out_lmdb_dir)
        dst_lmdb_env = lmdb.open(out_lmdb_dir, map_size= int(1024 ** 3 * 500))# 500G
        n_filtered_out = defaultdict(int)
    
        for index, file_name in self.selected_file.iterrows():
            f_name = file_name["id"]
            ext = ".npz" if "smplx" in self.args.pose_rep else ".bvh"
            pose_file = self.data_dir + self.args.pose_rep + "/" + f_name + ext
            pose_each_file = []
            trans_each_file = []
            shape_each_file = []
            audio_each_file = []
            facial_each_file = []
            word_each_file = []
            emo_each_file = []
            sem_each_file = []
            vid_each_file = []
            id_pose = f_name #1_wayne_0_1_1
            
            logger.info(colored(f"# ---- Building cache for Pose   {id_pose} ---- #", "blue"))
            if "smplx" in self.args.pose_rep:
                pose_data = np.load(pose_file, allow_pickle=True)
                assert 30%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 30'
                stride = int(30/self.args.pose_fps)
                pose_each_file = pose_data["poses"][::stride] # (1644, 165)
                trans_each_file = pose_data["trans"][::stride] # (1644, 3)
                shape_each_file = np.repeat(pose_data["betas"].reshape(1, 300), pose_each_file.shape[0], axis=0) # (1644, 300)
                
                assert self.args.pose_fps == 30, "should 30"
                m_data = np.load(pose_file, allow_pickle=True)
                betas, poses, trans, exps = m_data["betas"], m_data["poses"], m_data["trans"], m_data["expressions"]
                n, c = poses.shape[0], poses.shape[1]
                betas = betas.reshape(1, 300)
                betas = np.tile(betas, (n, 1))
                betas = torch.from_numpy(betas).cuda().float()
                poses = torch.from_numpy(poses.reshape(n, c)).cuda().float()
                exps = torch.from_numpy(exps.reshape(n, 100)).cuda().float()
                trans = torch.from_numpy(trans.reshape(n, 3)).cuda().float()
                max_length = 128    # 为什么这里需要一个max_length
                s, r = n//max_length, n%max_length
                #print(n, s, r)
                all_tensor = []
                for i in range(s):
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[i*max_length:(i+1)*max_length], 
                            transl=trans[i*max_length:(i+1)*max_length], 
                            expression=exps[i*max_length:(i+1)*max_length], 
                            jaw_pose=poses[i*max_length:(i+1)*max_length, 66:69], 
                            global_orient=poses[i*max_length:(i+1)*max_length,:3], 
                            body_pose=poses[i*max_length:(i+1)*max_length,3:21*3+3], 
                            left_hand_pose=poses[i*max_length:(i+1)*max_length,25*3:40*3], 
                            right_hand_pose=poses[i*max_length:(i+1)*max_length,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[i*max_length:(i+1)*max_length, 69:72], 
                            reye_pose=poses[i*max_length:(i+1)*max_length, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(max_length, 4, 3).cpu()
                    all_tensor.append(joints)
                if r != 0:
                    with torch.no_grad():
                        joints = self.smplx(
                            betas=betas[s*max_length:s*max_length+r], 
                            transl=trans[s*max_length:s*max_length+r], 
                            expression=exps[s*max_length:s*max_length+r], 
                            jaw_pose=poses[s*max_length:s*max_length+r, 66:69], 
                            global_orient=poses[s*max_length:s*max_length+r,:3], 
                            body_pose=poses[s*max_length:s*max_length+r,3:21*3+3], 
                            left_hand_pose=poses[s*max_length:s*max_length+r,25*3:40*3], 
                            right_hand_pose=poses[s*max_length:s*max_length+r,40*3:55*3], 
                            return_verts=True,
                            return_joints=True,
                            leye_pose=poses[s*max_length:s*max_length+r, 69:72], 
                            reye_pose=poses[s*max_length:s*max_length+r, 72:75],
                        )['joints'][:, (7,8,10,11), :].reshape(r, 4, 3).cpu()
                    all_tensor.append(joints)
                joints = torch.cat(all_tensor, axis=0) # all, 4, 3
                # print(joints.shape)
                feetv = torch.zeros(joints.shape[1], joints.shape[0])
                joints = joints.permute(1, 0, 2)
                #print(joints.shape, feetv.shape)
                feetv[:, :-1] = (joints[:, 1:] - joints[:, :-1]).norm(dim=-1)
                #print(feetv.shape)
                contacts = (feetv < 0.01).numpy().astype(float)
                # print(contacts.shape, contacts)
                contacts = contacts.transpose(1, 0)
                pose_each_file = pose_each_file * self.joint_mask
                pose_each_file = pose_each_file[:, self.joint_mask.astype(bool)]
                pose_each_file = np.concatenate([pose_each_file, contacts], axis=1)
                # print(pose_each_file.shape)
                
                
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_each_file = pose_data["expressions"][::stride]
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                    
            else:
                assert 120%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                stride = int(120/self.args.pose_fps)
                with open(pose_file, "r") as pose_data:
                    for j, line in enumerate(pose_data.readlines()):
                        if j < 431: continue     
                        if j%stride != 0:continue
                        data = np.fromstring(line, dtype=float, sep=" ")
                        rot_data = rc.euler_angles_to_matrix(torch.from_numpy(np.deg2rad(data)).reshape(-1, self.joints,3), "XYZ")
                        rot_data = rc.matrix_to_axis_angle(rot_data).reshape(-1, self.joints*3) 
                        rot_data = rot_data.numpy() * self.joint_mask
                        
                        pose_each_file.append(rot_data)
                        trans_each_file.append(data[:3])
                        
                pose_each_file = np.array(pose_each_file)
                # print(pose_each_file.shape)
                trans_each_file = np.array(trans_each_file)
                shape_each_file = np.repeat(np.array(-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
                if self.args.facial_rep is not None:
                    logger.info(f"# ---- Building cache for Facial {id_pose} and Pose {id_pose} ---- #")
                    facial_file = pose_file.replace(self.args.pose_rep, self.args.facial_rep).replace("bvh", "json")
                    assert 60%self.args.pose_fps == 0, 'pose_fps should be an aliquot part of 120'
                    stride = int(60/self.args.pose_fps)
                    if not os.path.exists(facial_file):
                        logger.warning(f"# ---- file not found for Facial {id_pose}, skip all files with the same id ---- #")
                        self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                        continue
                    with open(facial_file, 'r') as facial_data_file:
                        facial_data = json.load(facial_data_file)
                        for j, frame_data in enumerate(facial_data['frames']):
                            if j%stride != 0:continue
                            facial_each_file.append(frame_data['weights'])
                    facial_each_file = np.array(facial_each_file)
                    if self.args.facial_norm: 
                        facial_each_file = (facial_each_file - self.mean_facial) / self.std_facial
                        
            if self.args.id_rep is not None:
                vid_each_file = np.repeat(np.array(int(f_name.split("_")[0])-1).reshape(1, 1), pose_each_file.shape[0], axis=0)
      
            if self.args.audio_rep is not None:
                logger.info(f"# ---- Building cache for Audio  {id_pose} and Pose {id_pose} ---- #")
                audio_file = pose_file.replace(self.args.pose_rep, 'wave16k').replace(ext, ".wav")
                if not os.path.exists(audio_file):
                    logger.warning(f"# ---- file not found for Audio  {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                audio_save_path = audio_file.replace("wave16k", "onset_amplitude").replace(".wav", ".npy")
                if self.args.audio_rep == "onset+amplitude" and os.path.exists(audio_save_path):
                    audio_each_file = np.load(audio_save_path)
                    logger.warning(f"# ---- file found cache for Audio   {id_pose}   ---- #")
                elif self.args.audio_rep == "onset+amplitude":
                    audio_each_file, sr = librosa.load(audio_file)
                    audio_each_file = librosa.resample(audio_each_file, orig_sr=sr, target_sr=self.args.audio_sr)
                    from numpy.lib import stride_tricks
                    frame_length = 1024
                    # hop_length = 512
                    shape = (audio_each_file.shape[-1] - frame_length + 1, frame_length)
                    strides = (audio_each_file.strides[-1], audio_each_file.strides[-1])
                    rolling_view = stride_tricks.as_strided(audio_each_file, shape=shape, strides=strides)
                    amplitude_envelope = np.max(np.abs(rolling_view), axis=1)
                    # pad the last frame_length-1 samples
                    amplitude_envelope = np.pad(amplitude_envelope, (0, frame_length-1), mode='constant', constant_values=amplitude_envelope[-1])
                    audio_onset_f = librosa.onset.onset_detect(y=audio_each_file, sr=self.args.audio_sr, units='frames')
                    onset_array = np.zeros(len(audio_each_file), dtype=float)
                    onset_array[audio_onset_f] = 1.0
                    # print(amplitude_envelope.shape, audio_each_file.shape, onset_array.shape)
                    audio_each_file = np.concatenate([amplitude_envelope.reshape(-1, 1), onset_array.reshape(-1, 1)], axis=1)
                elif self.args.audio_rep == "mfcc":
                    audio_each_file = librosa.feature.melspectrogram(y=audio_each_file, sr=self.args.audio_sr, n_mels=128, hop_length=int(self.args.audio_sr/self.args.audio_fps))
                    audio_each_file = audio_each_file.transpose(1, 0)
                    # print(audio_each_file.shape, pose_each_file.shape)
                if self.args.audio_norm and self.args.audio_rep == "wave16k": 
                    audio_each_file = (audio_each_file - self.mean_audio) / self.std_audio
                    
            time_offset = 0
            if self.args.word_rep is not None:
                logger.info(f"# ---- Building cache for Word   {id_pose} and Pose {id_pose} ---- #")
                word_file = f"{self.data_dir}{self.args.word_rep}/{id_pose}.TextGrid"
                if not os.path.exists(word_file):
                    logger.warning(f"# ---- file not found for Word   {id_pose}, skip all files with the same id ---- #")
                    self.selected_file = self.selected_file.drop(self.selected_file[self.selected_file['id'] == id_pose].index)
                    continue
                word_save_path = f"{self.data_dir}{self.args.t_pre_encoder}/{id_pose}.npy"
                if os.path.exists(word_save_path):
                    word_each_file = np.load(word_save_path)
                    logger.warning(f"# ---- file found cache for Word   {id_pose}   ---- #")
                else:
                    tgrid = tg.TextGrid.fromFile(word_file)
                    if self.args.t_pre_encoder == "bert":
                        from transformers import AutoTokenizer, BertModel
                        tokenizer = AutoTokenizer.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True)
                        model = BertModel.from_pretrained(self.args.data_path_1 + "hub/bert-base-uncased", local_files_only=True).eval()
                        list_word = []
                        all_hidden = []
                        max_len = 400
                        last = 0
                        word_token_mapping = []
                        first = True
                        for i, word in enumerate(tgrid[0]):
                            last = i
                            if (i%max_len != 0) or (i==0):
                                if word.mark == "":
                                    list_word.append(".")
                                else:
                                    list_word.append(word.mark)
                            else:
                                max_counter = max_len
                                str_word = ' '.join(map(str, list_word))
                                if first:
                                    global_len = 0
                                end = -1
                                offset_word = []
                                for k, wordvalue in enumerate(list_word):
                                    start = end+1 
                                    end = start+len(wordvalue)
                                    offset_word.append((start, end))
                                #print(offset_word)
                                token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                                #print(token_scan)
                                for start, end in offset_word:
                                    sub_mapping = []
                                    for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                        if int(start) <= int(start_t) and int(end_t) <= int(end):
                                            #print(i+global_len)
                                            sub_mapping.append(i+global_len)
                                    word_token_mapping.append(sub_mapping)
                                #print(len(word_token_mapping))
                                global_len = word_token_mapping[-1][-1] + 1    
                                list_word = []
                                if word.mark == "":
                                    list_word.append(".")
                                else:
                                    list_word.append(word.mark)
                                
                                with torch.no_grad():
                                    inputs = tokenizer(str_word, return_tensors="pt")
                                    outputs = model(**inputs)
                                    last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                                all_hidden.append(last_hidden_states)
                        
                        #list_word = list_word[:10]
                        if list_word == []:
                            pass
                        else:
                            if first: 
                                global_len = 0
                            str_word = ' '.join(map(str, list_word))
                            end = -1
                            offset_word = []
                            for k, wordvalue in enumerate(list_word):
                                start = end+1 
                                end = start+len(wordvalue)
                                offset_word.append((start, end))
                            #print(offset_word)
                            token_scan = tokenizer.encode_plus(str_word, return_offsets_mapping=True)['offset_mapping']
                            #print(token_scan)
                            for start, end in offset_word:
                                sub_mapping = []
                                for i, (start_t, end_t) in enumerate(token_scan[1:-1]):
                                    if int(start) <= int(start_t) and int(end_t) <= int(end):
                                        sub_mapping.append(i+global_len)
                                        #print(sub_mapping)
                                word_token_mapping.append(sub_mapping)
                            #print(len(word_token_mapping))
                            with torch.no_grad():
                                inputs = tokenizer(str_word, return_tensors="pt")
                                outputs = model(**inputs)
                                last_hidden_states = outputs.last_hidden_state.reshape(-1, 768).cpu().numpy()[1:-1, :]
                            all_hidden.append(last_hidden_states)
                        last_hidden_states = np.concatenate(all_hidden, axis=0)
                
                    for i in range(pose_each_file.shape[0]):
                        found_flag = False
                        current_time = i/self.args.pose_fps + time_offset
                        j_last = 0
                        for j, word in enumerate(tgrid[0]): 
                            word_n, word_s, word_e = word.mark, word.minTime, word.maxTime
                            if word_s<=current_time and current_time<=word_e:
                                if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                                    mapping_index = word_token_mapping[j]
                                    #print(mapping_index, word_s, word_e)
                                    s_t = np.linspace(word_s, word_e, len(mapping_index)+1)
                                    #print(s_t)
                                    for tt, t_sep in enumerate(s_t[1:]):
                                        if current_time <= t_sep:
                                            #if len(mapping_index) > 1: print(mapping_index[tt])
                                            word_each_file.append(last_hidden_states[mapping_index[tt]])
                                            break
                                else:
                                    if word_n == " ":
                                        word_each_file.append(self.lang_model.PAD_token)
                                    else:
                                        word_each_file.append(self.lang_model.get_word_index(word_n))
                                found_flag = True
                                j_last = j
                                break
                            else: continue   
                        if not found_flag: 
                            if self.args.word_cache and self.args.t_pre_encoder == 'bert':
                                word_each_file.append(last_hidden_states[j_last])
                            else:
                                word_each_file.append(self.lang_model.UNK_token)
                    word_each_file = np.array(word_each_file)
                    #print(word_each_file.shape)
                
            if self.args.emo_rep is not None:
                logger.info(f"# ---- Building cache for Emo    {id_pose} and Pose {id_pose} ---- #")
                rtype, start = int(id_pose.split('_')[3]), int(id_pose.split('_')[3])
                if rtype == 0 or rtype == 2 or rtype == 4 or rtype == 6:
                    if start >= 1 and start <= 64:
                        score = 0
                    elif start >= 65 and start <= 72:
                        score = 1
                    elif start >= 73 and start <= 80:
                        score = 2
                    elif start >= 81 and start <= 86:
                        score = 3
                    elif start >= 87 and start <= 94:
                        score = 4
                    elif start >= 95 and start <= 102:
                        score = 5
                    elif start >= 103 and start <= 110:
                        score = 6
                    elif start >= 111 and start <= 118:
                        score = 7
                    else: pass
                else:
                    # you may denote as unknown in the future
                    score = 0
                emo_each_file = np.repeat(np.array(score).reshape(1, 1), pose_each_file.shape[0], axis=0)    
                #print(emo_each_file)
                
            if self.args.sem_rep is not None:
                logger.info(f"# ---- Building cache for Sem    {id_pose} and Pose {id_pose} ---- #")
                sem_file = f"{self.data_dir}{self.args.sem_rep}/{id_pose}.txt" 
                sem_all = pd.read_csv(sem_file, 
                    sep='\t', 
                    names=["name", "start_time", "end_time", "duration", "score", "keywords"])
                # we adopt motion-level semantic score here. 
                for i in range(pose_each_file.shape[0]):
                    found_flag = False
                    for j, (start, end, score) in enumerate(zip(sem_all['start_time'],sem_all['end_time'], sem_all['score'])):
                        current_time = i/self.args.pose_fps + time_offset
                        if start<=current_time and current_time<=end: 
                            sem_each_file.append(score)
                            found_flag=True
                            break
                        else: continue 
                    if not found_flag: sem_each_file.append(0.)
                sem_each_file = np.array(sem_each_file)
                #print(sem_each_file)
            
            # filtered_result = self._sample_from_clip(
            #     dst_lmdb_env,
            #     audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
            #     vid_each_file, emo_each_file, sem_each_file,
            #     disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
            #     ) 
            if True:
                # compute data for fastspeech2
                info, pitch_each_file, energy_each_file, mel_each_file, duration_each_file = self.process_utterance_original(id_pose)
                print(f"Processing {info}")

            filtered_result = self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
                vid_each_file, emo_each_file, sem_each_file,
                disable_filtering, clean_first_seconds, clean_final_seconds, is_test,
                pitch_each_file, energy_each_file, mel_each_file, duration_each_file
            ) 
            for type in filtered_result.keys():
                n_filtered_out[type] += filtered_result[type]
                                
        with dst_lmdb_env.begin() as txn:
            logger.info(colored(f"no. of samples: {txn.stat()['entries']}", "cyan"))
            n_total_filtered = 0
            for type, n_filtered in n_filtered_out.items():
                logger.info("{}: {}".format(type, n_filtered))
                n_total_filtered += n_filtered
            logger.info(colored("no. of excluded samples: {} ({:.1f}%)".format(
                n_total_filtered, 100 * n_total_filtered / (txn.stat()["entries"] + n_total_filtered)), "cyan"))
        dst_lmdb_env.sync()
        dst_lmdb_env.close()
    
    def _sample_from_clip(
        self, dst_lmdb_env, audio_each_file, pose_each_file, trans_each_file, shape_each_file, facial_each_file, word_each_file,
        vid_each_file, emo_each_file, sem_each_file,
        disable_filtering, clean_first_seconds, clean_final_seconds, is_test, 
        pitch_each_file = None, energy_each_file = None, mel_each_file = None, duration_each_file = None
        ):
        """
        for data cleaning, we ignore the data for first and final n s
        for test, we return all data 
        """
        # audio_start = int(self.alignment[0] * self.args.audio_fps)
        # pose_start = int(self.alignment[1] * self.args.pose_fps)
        #logger.info(f"before: {audio_each_file.shape} {pose_each_file.shape}")
        # audio_each_file = audio_each_file[audio_start:]
        # pose_each_file = pose_each_file[pose_start:]
        # trans_each_file = 
        #logger.info(f"after alignment: {audio_each_file.shape} {pose_each_file.shape}")
        #print(pose_each_file.shape)
        round_seconds_skeleton = pose_each_file.shape[0] // self.args.pose_fps  # 1644 frames / 30 fps = 54 s
        #print(round_seconds_skeleton)
        if audio_each_file != []:
            if self.args.audio_rep != "wave16k":
                round_seconds_audio = len(audio_each_file) // self.args.audio_fps # assume 880000 / 16,000 = 55 s
            elif self.args.audio_rep == "mfcc":
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_fps
            else:
                round_seconds_audio = audio_each_file.shape[0] // self.args.audio_sr
            if facial_each_file != []:
                round_seconds_facial = facial_each_file.shape[0] // self.args.pose_fps
                logger.info(f"audio: {round_seconds_audio}s, pose: {round_seconds_skeleton}s, facial: {round_seconds_facial}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                max_round = max(round_seconds_audio, round_seconds_skeleton, round_seconds_facial)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")  
            else:
                logger.info(f"pose: {round_seconds_skeleton}s, audio: {round_seconds_audio}s")
                round_seconds_skeleton = min(round_seconds_audio, round_seconds_skeleton)
                max_round = max(round_seconds_audio, round_seconds_skeleton)
                if round_seconds_skeleton != max_round: 
                    logger.warning(f"reduce to {round_seconds_skeleton}s, ignore {max_round-round_seconds_skeleton}s")
        
        clip_s_t, clip_e_t = clean_first_seconds, round_seconds_skeleton - clean_final_seconds # assume [10, 90]s
        clip_s_f_audio, clip_e_f_audio = self.args.audio_fps * clip_s_t, clip_e_t * self.args.audio_fps # [160,000,90*160,000]
        clip_s_f_pose, clip_e_f_pose = clip_s_t * self.args.pose_fps, clip_e_t * self.args.pose_fps # [150,90*15]
        
        
        for ratio in self.args.multi_length_training:
            if is_test:# stride = length for test
                cut_length = clip_e_f_pose - clip_s_f_pose
                self.args.stride = cut_length
                self.max_length = cut_length
            else:
                self.args.stride = int(ratio*self.ori_stride)
                cut_length = int(self.ori_length*ratio)
                
            num_subdivision = math.floor((clip_e_f_pose - clip_s_f_pose - cut_length) / self.args.stride) + 1
            logger.info(f"pose from frame {clip_s_f_pose} to {clip_e_f_pose}, length {cut_length}")
            logger.info(f"{num_subdivision} clips is expected with stride {self.args.stride}")
            
            if audio_each_file != []:
                audio_short_length = math.floor(cut_length / self.args.pose_fps * self.args.audio_fps)
                """
                for audio sr = 16000, fps = 15, pose_length = 34, 
                audio short length = 36266.7 -> 36266
                this error is fine.
                """
                logger.info(f"audio from frame {clip_s_f_audio} to {clip_e_f_audio}, length {audio_short_length}")
             
            n_filtered_out = defaultdict(int)
            sample_pose_list = []
            sample_audio_list = []
            sample_facial_list = []
            sample_shape_list = []
            sample_word_list = []
            sample_emo_list = []
            sample_sem_list = []
            sample_vid_list = []
            sample_trans_list = []

            sample_pitch_list = []
            sample_energy_list = []
            sample_duration_list = []
            sample_mel_list = []
           
            for i in range(num_subdivision): # cut into around 2s chip, (self npose)
                start_idx = clip_s_f_pose + i * self.args.stride
                fin_idx = start_idx + cut_length 
                sample_pose = pose_each_file[start_idx:fin_idx]

                sample_trans = trans_each_file[start_idx:fin_idx]
                sample_shape = shape_each_file[start_idx:fin_idx]
                # print(sample_pose.shape)
                if self.args.audio_rep is not None:
                    audio_start = clip_s_f_audio + math.floor(i * self.args.stride * self.args.audio_fps / self.args.pose_fps)  # 0
                    audio_end = audio_start + audio_short_length # 34133
                    sample_audio = audio_each_file[audio_start:audio_end] 
                else:
                    sample_audio = np.array([-1])
                sample_facial = facial_each_file[start_idx:fin_idx] if self.args.facial_rep is not None else np.array([-1])
                sample_word = word_each_file[start_idx:fin_idx] if self.args.word_rep is not None else np.array([-1])
                sample_emo = emo_each_file[start_idx:fin_idx] if self.args.emo_rep is not None else np.array([-1])
                sample_sem = sem_each_file[start_idx:fin_idx] if self.args.sem_rep is not None else np.array([-1])
                sample_vid = vid_each_file[start_idx:fin_idx] if self.args.id_rep is not None else np.array([-1])
                
                if sample_pose.any() != None:
                    # filtering motion skeleton data
                    sample_pose, filtering_message = MotionPreprocessor(sample_pose).get()
                    is_correct_motion = (sample_pose != [])
                    if is_correct_motion or disable_filtering:
                        sample_pose_list.append(sample_pose)
                        sample_audio_list.append(sample_audio)
                        sample_facial_list.append(sample_facial)
                        sample_shape_list.append(sample_shape)
                        sample_word_list.append(sample_word)
                        sample_vid_list.append(sample_vid)
                        sample_emo_list.append(sample_emo)
                        sample_sem_list.append(sample_sem)
                        sample_trans_list.append(sample_trans)
                    else:
                        n_filtered_out[filtering_message] += 1

                # clip energy, pitch, etc based on start_idx and fin_idx
                if pitch_each_file is not None:
                    sample_pitch= pitch_each_file[start_idx:fin_idx]
                    sample_energy = energy_each_file[start_idx:fin_idx]
                    sample_duration = duration_each_file[start_idx:fin_idx]
                    sample_mel = mel_each_file[start_idx:fin_idx]

                    sample_pitch_list.append(sample_pitch)
                    sample_energy_list.append(sample_energy)
                    sample_duration_list.append(sample_duration)
                    sample_mel_list.append(sample_mel)



            if len(sample_pose_list) > 0:
                with dst_lmdb_env.begin(write=True) as txn:
                    for pose, audio, facial, shape, word, vid, emo, sem, trans, pitch, energy, duration, mel in zip(
                        sample_pose_list,
                        sample_audio_list,
                        sample_facial_list,
                        sample_shape_list,
                        sample_word_list,
                        sample_vid_list,
                        sample_emo_list,
                        sample_sem_list,
                        sample_trans_list,
                        sample_pitch_list,
                        sample_energy_list,
                        sample_duration_list,
                        sample_mel_list):
                        k = "{:005}".format(self.n_out_samples).encode("ascii")
                        # v = [pose, audio, facial, shape, word, emo, sem, vid, trans]
                        # now we have data for each 2s clip
                        v = [pose, audio, facial, shape, word, emo, sem, vid, trans, pitch, energy, np.array(duration), mel]
                        v = pickle.dumps(v,5)
                        txn.put(k, v)
                        self.n_out_samples += 1
         
        return n_filtered_out


    ################################################
    # borrowed from fastspeech2

    def remove_outlier(self, values):
        values = np.array(values)
        p25 = np.percentile(values, 25)
        p75 = np.percentile(values, 75)
        lower = p25 - 1.5 * (p75 - p25)
        upper = p75 + 1.5 * (p75 - p25)
        normal_indices = np.logical_and(values > lower, values < upper)

        return values[normal_indices]

    def get_phone_and_duration(self, tier, start_time, end_time):
        """
        Given start_time and end_time in seconds, extract all the phonemes
        where start_time <= t.start_time and t.end_time <= end_time, excluding silent phonemes.
        Also calculate the overlapping duration with the given time window for each phoneme.

        Parameters:
        - tier: IntervalTier object with phoneme intervals.
        - start_time: float, start of the time range in seconds.
        - end_time: float, end of the time range in seconds.

        Returns:
        - filtered_phonemes: list of phonemes within the specified time range, excluding silent phonemes.
        - durations: list of durations, in seconds, for each phoneme, adjusted for the specified time window.
        """
        sil_phones = ["sil", "sp", "spn"]  # Define which phonemes are considered silent
        filtered_phonemes = []
        durations = []

        # Iterate through each phoneme interval in the tier.
        for interval in tier._objects:
            if (interval.start_time < end_time and interval.end_time > start_time):
                if interval.text not in sil_phones:  # Check if the phoneme is not a silent one
                    # Calculate effective start and end times within the given window
                    effective_start = max(interval.start_time, start_time)
                    effective_end = min(interval.end_time, end_time)
                    # Calculate duration within the window
                    duration = effective_end - effective_start
                    if duration > 0:  # Only add phonemes with a positive duration

                        duration = int(
                            np.round(effective_end * self.sampling_rate / self.hop_length) # second multiply by fps = starting frame
                            - np.round(effective_start * self.sampling_rate / self.hop_length)
                        )
                        filtered_phonemes.append(interval.text)
                        durations.append(duration)

        return filtered_phonemes, durations

    
    def process_utterance_original(self, basename):
        self.in_dir = "/scratch/year/zixin/2024/PantoMatrix/BEAT2/beat_english_v2.0.0"
        self.pitch_phoneme_averaging = True
        self.energy_phoneme_averaging = True
        speaker = basename.split('_')[0]
        wav_path = os.path.join(self.in_dir, 'wave16k', "{}.wav".format(basename))
        # text_path = os.path.join(self.in_dir, speaker, "{}.lab".format(basename))
        tg_path = os.path.join(
            self.in_dir, "textgrid", "{}.TextGrid".format(basename)
        )
        
        wav, self.sampling_rate = librosa.load(wav_path)
        self.hop_length = self.sampling_rate / 30.0
       

        # Get alignments
        # textgrid = tgt.io.read_textgrid(tg_path)
        import textgrid
        textgrid = textgrid.TextGrid.fromFile(tg_path)
        phone, duration, start, end = self.get_alignment(
            textgrid.tiers[1] # phonemes by default
        )
        text = "{" + " ".join(phone) + "}"
        if start >= end:
            return None

        wav = wav[
            int(self.sampling_rate * start) : int(self.sampling_rate * end)
        ].astype(np.float32)

        # Read and trim wav files
        

        # Read raw text
        # with open(text_path, "r") as f:
        #     raw_text = f.readline().strip("\n")

        # Compute fundamental frequency
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(duration)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, _STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(duration)]
        energy = energy[: sum(duration)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch_avg = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(duration):
                if d > 0:
                    pitch_avg[i] = np.mean(pitch_avg[pos : pos + d])
                else:
                    pitch_avg[i] = 0
                pos += d
            pitch_avg = pitch_avg[: len(duration)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            energy_avg = np.copy(energy)
            for i, d in enumerate(duration):
                if d > 0:
                    energy_avg[i] = np.mean(energy_avg[pos : pos + d])
                else:
                    energy_avg[i] = 0
                pos += d
            energy_avg = energy_avg[: len(duration)]

        # Save files
        # dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.in_dir, "duration", dur_filename), duration)

        # pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.in_dir, "pitch", pitch_filename), pitch)

        # energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.in_dir, "energy", energy_filename), energy)

        # mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        # np.save(
        #     os.path.join(self.in_dir, "mel", mel_filename),
        #     mel_spectrogram.T,
        # )
        duration, pitch, energy = self.process_for_frame(duration, pitch_avg, energy_avg)

        return (
            # "|".join([basename, speaker, text, raw_text]),
            "|".join([basename, speaker, text]),
            pitch, # need to process outliers
            energy, # need to process outliers
            mel_spectrogram.T,
            duration
        )
    
    def process_for_frame(self, duration, pitch_avg, energy_avg):
        """
        duration: Take in [55, 8, 1, 2, ...], outputs: repeat each element by its value
        pitch_avg: repeat based on element-wise duration value
        """
        repeats = np.array(duration)
        frame_duration = np.repeat(repeats, repeats)
        frame_pitch = np.repeat(pitch_avg, repeats)
        frame_energy = np.repeat(energy_avg, repeats)
       
        return frame_duration.tolist(), frame_pitch, frame_energy

    def process_utterance(self, basename, start_time, end_time):
        """
        Expected Output for input to fastspeech2:
            speakers, # (64,)
            texts (phonome), # (64, 158)
            text_lens, # (64,)
            max(text_lens), # 158
            mels, # (64, 1298, 80)
            mel_lens, # (64,)
            max(mel_lens), # 1298
            pitches, # (64, 158)
            energies, # (64, 158)
            durations, # (64, 158)
        """
        self.in_dir = "/scratch/year/zixin/2024/PantoMatrix/BEAT2/beat_english_v2.0.0"
        self.pitch_phoneme_averaging = True
        self.energy_phoneme_averaging = True

        wav_path = os.path.join(self.in_dir, 'wave16k', "{}.wav".format(basename))
        tg_path = os.path.join(
            self.in_dir, "textgrid", "{}.TextGrid".format(basename)
        )

        wav, self.sampling_rate = librosa.load(wav_path)
        self.hop_length = self.sampling_rate / 30.0
        # Read and trim wav files
        wav = wav[
            int(self.sampling_rate * start_time) : int(self.sampling_rate * end_time)
        ].astype(np.float32)

        # Get alignments
        textgrid = tgt.io.read_textgrid(tg_path)

        # given start_time and end_time of the current clip, extract the phone, duration
        phones, durations = self.get_phone_and_duration(textgrid.get_tier_by_name("phones"), start_time, end_time)
   
        text = "{" + " ".join(phones) + "}"

        if len(durations) == 0: return None
        
    
        # Compute fundamental frequency based on cropped wav
        pitch, t = pw.dio(
            wav.astype(np.float64),
            self.sampling_rate,
            frame_period=self.hop_length / self.sampling_rate * 1000,
        )
        pitch = pw.stonemask(wav.astype(np.float64), pitch, t, self.sampling_rate)

        pitch = pitch[: sum(durations)]
        if np.sum(pitch != 0) <= 1:
            return None

        # Compute mel-scale spectrogram and energy
        mel_spectrogram, energy = Audio.tools.get_mel_from_wav(wav, _STFT)
        mel_spectrogram = mel_spectrogram[:, : sum(durations)]
        energy = energy[: sum(durations)]

        if self.pitch_phoneme_averaging:
            # perform linear interpolation
            nonzero_ids = np.where(pitch != 0)[0]
            interp_fn = interp1d(
                nonzero_ids,
                pitch[nonzero_ids],
                fill_value=(pitch[nonzero_ids[0]], pitch[nonzero_ids[-1]]),
                bounds_error=False,
            )
            pitch = interp_fn(np.arange(0, len(pitch)))

            # Phoneme-level average
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    pitch[i] = np.mean(pitch[pos : pos + d])
                else:
                    pitch[i] = 0
                pos += d
            pitch = pitch[: len(durations)]

        if self.energy_phoneme_averaging:
            # Phoneme-level average
            pos = 0
            for i, d in enumerate(durations):
                if d > 0:
                    energy[i] = np.mean(energy[pos : pos + d])
                else:
                    energy[i] = 0
                pos += d
            energy = energy[: len(durations)]


        self.remove_outlier(pitch)
        self.remove_outlier(energy)

        # # Save files
        # dur_filename = "{}-duration-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "duration", dur_filename), duration)

        # pitch_filename = "{}-pitch-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)

        # energy_filename = "{}-energy-{}.npy".format(speaker, basename)
        # np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)

        # mel_filename = "{}-mel-{}.npy".format(speaker, basename)
        # np.save(
        #     os.path.join(self.out_dir, "mel", mel_filename),
        #     mel_spectrogram.T,
        # )

        # return (
        #     "|".join([basename, speaker, text, raw_text]),
        #     self.remove_outlier(pitch),
        #     self.remove_outlier(energy),
        #     mel_spectrogram.shape[1],
        # )
        return 

    def get_alignment(self, tier):
        phones = []
        durations = []
        starts = 0
        ends = 0
        for interval in tier.intervals:
            
            phones.append(interval.mark)
            ends = interval.maxTime
            durations.append(
            int(
                np.round(interval.maxTime * self.sampling_rate / self.hop_length)
                - np.round(interval.minTime * self.sampling_rate / self.hop_length)
            )
        )
        return phones, durations, starts, ends

    def get_alignment_original(self, tier):
        # sil_phones = ["sil", "sp", "spn"]
        sil_phones = []
        phones = []
        durations = []
        start_time = 0
        end_time = 0
        for t in tier._objects:
            s, e, p = t.start_time, t.end_time, t.text

            # # Trim leading silences
            # if phones == []:
            #     if p in sil_phones:
            #         continue
            #     else:
            #         start_time = s

    
            phones.append(p)
            end_time = e
                
            durations.append(
                int(
                    np.round(e * self.sampling_rate / self.hop_length)
                    - np.round(s * self.sampling_rate / self.hop_length)
                )
            )

        # Trim tailing silences
        # phones = phones[:end_idx]
        # durations = durations[:end_idx]

        return phones, durations, start_time, end_time


    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)
            sample = pickle.loads(sample)
            # tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans = sample
            tar_pose, in_audio, in_facial, in_shape, in_word, emo, sem, vid, trans, pitch, energy, duration, mel = sample # current is frame-level

            # (64, 169), (34133, 2), (64, 100), (64, 300), (64, ), 
       
            #vid = torch.from_numpy(vid).int()
            emo = torch.from_numpy(emo).int()
            sem = torch.from_numpy(sem).float() 
            in_audio = torch.from_numpy(in_audio).float() 
            in_word = torch.from_numpy(in_word).float() if self.args.word_cache else torch.from_numpy(in_word).int() 
            if self.loader_type == "test":
                tar_pose = torch.from_numpy(tar_pose).float()
                trans = torch.from_numpy(trans).float()
                in_facial = torch.from_numpy(in_facial).float()
                vid = torch.from_numpy(vid).float()
                in_shape = torch.from_numpy(in_shape).float()
            else:
                in_shape = torch.from_numpy(in_shape).reshape((in_shape.shape[0], -1)).float()
                trans = torch.from_numpy(trans).reshape((trans.shape[0], -1)).float()
                vid = torch.from_numpy(vid).reshape((vid.shape[0], -1)).float()
                tar_pose = torch.from_numpy(tar_pose).reshape((tar_pose.shape[0], -1)).float()
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()
            return {"pose":tar_pose, "audio":in_audio, "facial":in_facial, "beta": in_shape, "word":in_word, "id":vid, "emo":emo, "sem":sem, "trans":trans}

         
class MotionPreprocessor:
    def __init__(self, skeletons):
        self.skeletons = skeletons
        #self.mean_pose = mean_pose
        self.filtering_message = "PASS"

    def get(self):
        assert (self.skeletons is not None)

        # filtering
        if self.skeletons != []:
            if self.check_pose_diff():
                self.skeletons = []
                self.filtering_message = "pose"
            # elif self.check_spine_angle():
            #     self.skeletons = []
            #     self.filtering_message = "spine angle"
            # elif self.check_static_motion():
            #     self.skeletons = []
            #     self.filtering_message = "motion"

        # if self.skeletons != []:
        #     self.skeletons = self.skeletons.tolist()
        #     for i, frame in enumerate(self.skeletons):
        #         assert not np.isnan(self.skeletons[i]).any()  # missing joints

        return self.skeletons, self.filtering_message

    def check_static_motion(self, verbose=True):
        def get_variance(skeleton, joint_idx):
            wrist_pos = skeleton[:, joint_idx]
            variance = np.sum(np.var(wrist_pos, axis=0))
            return variance

        left_arm_var = get_variance(self.skeletons, 6)
        right_arm_var = get_variance(self.skeletons, 9)

        th = 0.0014  # exclude 13110
        # th = 0.002  # exclude 16905
        if left_arm_var < th and right_arm_var < th:
            if verbose:
                print("skip - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return True
        else:
            if verbose:
                print("pass - check_static_motion left var {}, right var {}".format(left_arm_var, right_arm_var))
            return False


    def check_pose_diff(self, verbose=False):
#         diff = np.abs(self.skeletons - self.mean_pose) # 186*1
#         diff = np.mean(diff)

#         # th = 0.017
#         th = 0.02 #0.02  # exclude 3594
#         if diff < th:
#             if verbose:
#                 print("skip - check_pose_diff {:.5f}".format(diff))
#             return True
# #         th = 3.5 #0.02  # exclude 3594
# #         if 3.5 < diff < 5:
# #             if verbose:
# #                 print("skip - check_pose_diff {:.5f}".format(diff))
# #             return True
#         else:
#             if verbose:
#                 print("pass - check_pose_diff {:.5f}".format(diff))
        return False


    def check_spine_angle(self, verbose=True):
        def angle_between(v1, v2):
            v1_u = v1 / np.linalg.norm(v1)
            v2_u = v2 / np.linalg.norm(v2)
            return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

        angles = []
        for i in range(self.skeletons.shape[0]):
            spine_vec = self.skeletons[i, 1] - self.skeletons[i, 0]
            angle = angle_between(spine_vec, [0, -1, 0])
            angles.append(angle)

        if np.rad2deg(max(angles)) > 30 or np.rad2deg(np.mean(angles)) > 20:  # exclude 4495
        # if np.rad2deg(max(angles)) > 20:  # exclude 8270
            if verbose:
                print("skip - check_spine_angle {:.5f}, {:.5f}".format(max(angles), np.mean(angles)))
            return True
        else:
            if verbose:
                print("pass - check_spine_angle {:.5f}".format(max(angles)))
            return False