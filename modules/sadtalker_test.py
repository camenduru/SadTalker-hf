import torch
from time import gmtime, strftime
import os, sys, shutil
from argparse import ArgumentParser
from src.utils.preprocess import CropAndExtract
from src.test_audio2coeff import Audio2Coeff  
from src.facerender.animate import AnimateFromCoeff
from src.generate_batch import get_data
from src.generate_facerender_batch import get_facerender_data

from modules.text2speech import text2speech

class SadTalker():

    def __init__(self, checkpoint_path='checkpoints'):

        if torch.cuda.is_available() :
            device = "cuda"
        else:
            device = "cpu"
        
        current_code_path = sys.argv[0]
        modules_path = os.path.split(current_code_path)[0]

        current_root_path = './'

        os.environ['TORCH_HOME']=os.path.join(current_root_path, 'checkpoints')

        path_of_lm_croper = os.path.join(current_root_path, 'checkpoints', 'shape_predictor_68_face_landmarks.dat')
        path_of_net_recon_model = os.path.join(current_root_path, 'checkpoints', 'epoch_20.pth')
        dir_of_BFM_fitting = os.path.join(current_root_path, 'checkpoints', 'BFM_Fitting')
        wav2lip_checkpoint = os.path.join(current_root_path, 'checkpoints', 'wav2lip.pth')

        audio2pose_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2pose_00140-model.pth')
        audio2pose_yaml_path = os.path.join(current_root_path, 'config', 'auido2pose.yaml')
    
        audio2exp_checkpoint = os.path.join(current_root_path, 'checkpoints', 'auido2exp_00300-model.pth')
        audio2exp_yaml_path = os.path.join(current_root_path, 'config', 'auido2exp.yaml')

        free_view_checkpoint = os.path.join(current_root_path, 'checkpoints', 'facevid2vid_00189-model.pth.tar')
        mapping_checkpoint = os.path.join(current_root_path, 'checkpoints', 'mapping_00229-model.pth.tar')
        facerender_yaml_path = os.path.join(current_root_path, 'config', 'facerender.yaml')

        #init model
        print(path_of_lm_croper)
        self.preprocess_model = CropAndExtract(path_of_lm_croper, path_of_net_recon_model, dir_of_BFM_fitting, device)

        print(audio2pose_checkpoint)
        self.audio_to_coeff = Audio2Coeff(audio2pose_checkpoint, audio2pose_yaml_path, 
                                audio2exp_checkpoint, audio2exp_yaml_path, wav2lip_checkpoint, device)
        print(free_view_checkpoint)
        self.animate_from_coeff = AnimateFromCoeff(free_view_checkpoint, mapping_checkpoint, 
                                            facerender_yaml_path, device)
        self.device = device

    def test(self, source_image, driven_audio, still_mode, use_enhancer, result_dir='./'):

        time_tag = strftime("%Y_%m_%d_%H.%M.%S")
        save_dir = os.path.join(result_dir, time_tag)
        os.makedirs(save_dir, exist_ok=True)

        input_dir = os.path.join(save_dir, 'input')
        os.makedirs(input_dir, exist_ok=True)

        print(source_image)
        pic_path = os.path.join(input_dir, os.path.basename(source_image)) 
        shutil.move(source_image, input_dir)

        if os.path.isfile(driven_audio):
            audio_path = os.path.join(input_dir, os.path.basename(driven_audio))  
            shutil.move(driven_audio, input_dir)
        else:
            text2speech


        os.makedirs(save_dir, exist_ok=True)
        pose_style = 0
        #crop image and extract 3dmm from image
        first_frame_dir = os.path.join(save_dir, 'first_frame_dir')
        os.makedirs(first_frame_dir, exist_ok=True)
        first_coeff_path, crop_pic_path = self.preprocess_model.generate(pic_path, first_frame_dir)
        if first_coeff_path is None:
            raise AttributeError("No face is detected")

        #audio2ceoff
        batch = get_data(first_coeff_path, audio_path, self.device)
        coeff_path = self.audio_to_coeff.generate(batch, save_dir, pose_style)
        #coeff2video
        batch_size = 8
        data = get_facerender_data(coeff_path, crop_pic_path, first_coeff_path, audio_path, batch_size, still_mode=still_mode)
        self.animate_from_coeff.generate(data, save_dir, enhancer='gfpgan' if use_enhancer else None)
        video_name = data['video_name']
        print(f'The generated video is named {video_name} in {save_dir}')
        
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        if use_enhancer:
            return os.path.join(save_dir, video_name+'_enhanced.mp4'), os.path.join(save_dir, video_name+'_enhanced.mp4')

        else:
            return os.path.join(save_dir, video_name+'.mp4'), os.path.join(save_dir, video_name+'.mp4')
    