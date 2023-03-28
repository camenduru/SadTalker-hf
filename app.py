import pickle
import time
import numpy as np
import scipy, cv2, os, sys, argparse
from tqdm import tqdm
import torch
import librosa
from networks import define_G
from pcavs.config.AudioConfig import AudioConfig

sys.path.append('spectre')
from config import cfg as spectre_cfg
from src.spectre import SPECTRE

from audio2mesh_helper import *
from pcavs.models import create_model, networks

torch.manual_seed(0)
from scipy.signal import savgol_filter


class SimpleWrapperV2(nn.Module):
    def __init__(self, cfg, use_ref=True, exp_dim=53, noload=False) -> None:
        super().__init__()
       
        self.audio_encoder = networks.define_A_sync(cfg)

        self.mapping1 = nn.Linear(512+exp_dim, exp_dim)
        nn.init.constant_(self.mapping1.weight, 0.)
        nn.init.constant_(self.mapping1.bias, 0.)
        self.use_ref = use_ref

    def forward(self, x, ref, use_tanh=False):
        x = self.audio_encoder.forward_feature(x).view(x.size(0), -1)
        ref_reshape = ref.reshape(x.size(0), -1)  #20, -1
        
        y = self.mapping1(torch.cat([x, ref_reshape], dim=1)) 

        if self.use_ref:
            out = y.reshape(ref.shape[0], ref.shape[1], -1) + ref # resudial
        else:
            out = y.reshape(ref.shape[0], ref.shape[1], -1)

        if use_tanh:
            out[:, :50] = torch.tanh(out[:, :50]) * 3
            
        return out

class Audio2Mesh(object):
    def __init__(self, args) -> None:
        self.args = args

        spectre_cfg.model.use_tex = True
        spectre_cfg.model.mask_type = args.mask_type
        spectre_cfg.debug = self.args.debug
        spectre_cfg.model.netA_sync = 'ressesync'
        spectre_cfg.model.gpu_ids = [0]

        self.spectre = SPECTRE(spectre_cfg)
        self.spectre.eval()
        self.face_tracker = None #FaceTrackerV2() # face landmark detection
        self.mel_step_size = 16
        self.fps = args.fps
        self.Nw = args.tframes
        self.device = self.args.device
        self.image_size = self.args.image_size

        ### only audio
        args.netA_sync = 'ressesync'
        args.gpu_ids = [0]
        args.exp_dim = 53
        args.use_tanh = False
        args.K = 20 

        self.audio2exp = 'pcavs'

        # 
        self.avmodel = SimpleWrapperV2(args, exp_dim=args.exp_dim).cuda() 
        self.avmodel.load_state_dict(torch.load('../packages/pretrained/audio2expression_v2_model.tar')['opt'])

        # 5, 160 = 25fps
        self.audio = AudioConfig(frame_rate=args.fps, num_frames_per_clip=5, hop_size=160)
        
        with open(os.path.join(args.source_dir, 'deca_infos.pkl'), 'rb') as f: # ?
            self.fitting_coeffs = pickle.load(f, encoding='bytes')

        self.coeffs_dict  = { key: torch.Tensor(self.fitting_coeffs[key]).cuda().squeeze(1) for key in ['cam', 'pose', 'light', 'tex', 'shape', 'exp']}

        #### find the close month
        exp_tensors = torch.sum(self.coeffs_dict['exp'], dim=1)
        ssss, sorted_indices = torch.sort(exp_tensors)
        self.exp_id = sorted_indices[0].item()

        if '.ts' in args.render_path:
            self.render = torch.jit.load(args.render_path).cuda()
            self.trt = True
        else:
            self.render = define_G(self.Nw*6, 3, args.ngf, args.netR).eval().cuda()
            self.render.load_state_dict(torch.load(args.render_path))
            self.trt = False

        print('loaded cached images...')

    @torch.no_grad()
    def cg2real(self, rendedimages, start_frame=0):

        ## load original image and the mask
        self.source_images = np.concatenate(load_image_from_dir(os.path.join(self.args.source_dir, 'original_frame'),\
             resize=self.image_size, limit=len(rendedimages)+start_frame))[start_frame:]
        self.source_masks = np.concatenate(load_image_from_dir(os.path.join(self.args.source_dir, 'original_mask'),\
             resize=self.image_size, limit=len(rendedimages)+start_frame))[start_frame:]

        self.source_masks = torch.FloatTensor(np.transpose(self.source_masks,(0,3,1,2))/255.)
        self.padded_real_tensor = torch.FloatTensor(np.transpose(self.source_images,(0,3,1,2))/255.)

        ## padding the rended_imgs
        paded_tensor = torch.cat([rendedimages[0:1]]* (self.Nw // 2) + [rendedimages] + [rendedimages[-1:]]* (self.Nw // 2)).contiguous()
        paded_mask_tensor = torch.cat([self.source_masks[0:1]]* (self.Nw // 2) + [self.source_masks] + [self.source_masks[-1:]]* (self.Nw // 2)).contiguous()
        paded_real_tensor = torch.cat([self.padded_real_tensor[0:1]]* (self.Nw // 2) + [self.padded_real_tensor] + [self.padded_real_tensor[-1:]]* (self.Nw // 2)).contiguous()

        # paded_mask_tensor = maskErosion(paded_mask_tensor, offY=self.args.mask)
        padded_input = ((paded_real_tensor-0.5)*2 ) # *(1-paded_mask_tensor)
        padded_input = torch.nn.functional.interpolate(padded_input, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
        paded_tensor = torch.nn.functional.interpolate(paded_tensor, (self.image_size, self.image_size), mode='bilinear', align_corners=False)
        paded_tensor = (paded_tensor-0.5)*2

        result = []
        for index in tqdm(range(0, len(rendedimages), self.args.renderbs), desc='CG2REAL:'):
            list_A = []
            list_R = []
            list_M = []
            for i in range(self.args.renderbs):
                idx = index + i
                if idx+self.Nw > len(padded_input):
                    list_A.append(torch.zeros(self.Nw*3,self.image_size,self.image_size).unsqueeze(0))
                    list_R.append(torch.zeros(self.Nw*3,self.image_size,self.image_size).unsqueeze(0))
                    list_M.append(torch.zeros(self.Nw*3,self.image_size,self.image_size).unsqueeze(0))
                else:
                    list_A.append(padded_input[idx:idx+self.Nw].view(-1, self.image_size, self.image_size).unsqueeze(0))
                    list_R.append(paded_tensor[idx:idx+self.Nw].view(-1, self.image_size, self.image_size).unsqueeze(0))
                    list_M.append(paded_mask_tensor[idx:idx+self.Nw].view(-1, self.image_size, self.image_size).unsqueeze(0))

            list_A = torch.cat(list_A)
            list_R = torch.cat(list_R)
            list_M = torch.cat(list_M)

            idx = (self.Nw//2) * 3
            mask = list_M[:, idx:idx+3]

            # list_A = padded_input
            mask = maskErosion(mask, offY=self.args.mask)
            list_A = list_A * (1 - mask[:,0:1])
            A = torch.cat([list_A, list_R], 1)

            if self.trt:
                B = self.render(A.half().cuda())
            elif self.args.netR == 'unet_256':
                # import pdb; pdb.set_trace()
                idx = (self.Nw//2) * 3
                mask = list_M[:, idx:idx+3].cuda()
                mask = maskErosion(mask, offY=self.args.mask)
                B0 = list_A[:, idx:idx+3].cuda()
                B = self.render(A.cuda()) * mask[:,0:1] + (1 - mask[:,0:1]) * B0
            elif  self.args.netR == 's2am':
                # import pdb; pdb.set_trace()
                idx = (self.Nw//2) * 3
                mask = list_M[:, idx:idx+3].cuda()
                mask = maskErosion(mask, offY=self.args.mask)
                B0 = list_A[:, idx:idx+3].cuda()
                B = self.render(A.cuda(), mask[:,0:1] ) * mask[:,0:1] + (1 - mask[:,0:1]) * B0
            else:
                B = self.render(A.cuda()) 

            result.append((B.cpu() + 1) * 0.5) # -1,1 -> 0,1
        
        return torch.cat(result)[:len(rendedimages)]

    @torch.no_grad()
    def coeffs_to_img(self, vertices, coeffs, zero_pose=False, XK = 20):

        xlen = vertices.shape[0]
        all_shape_images = []
        landmark2d = []

        #### find the most larger pose 51 in the coeffs.
        max_pose_51 = torch.max(self.coeffs_dict['pose'][..., 3:4].squeeze(-1))

        for i in tqdm(range(0, xlen, XK)):
            
            if i + XK > xlen:
                XK = xlen - i

            codedictdecoder = {}
            codedictdecoder['shape'] = torch.zeros_like(self.coeffs_dict['shape'][i:i+XK].cuda())
            codedictdecoder['tex'] = self.coeffs_dict['tex'][i:i+XK].cuda()
            codedictdecoder['exp'] =  torch.zeros_like(self.coeffs_dict['exp'][i:i+XK].cuda()) #  all_exps[i:i+XK, :50].cuda()  # # # vid_exps[i:i+1].cuda() i:i+XK
            codedictdecoder['pose'] = self.coeffs_dict['pose'][i:i+XK]  # vid_poses[i:i+1].cuda()
            codedictdecoder['cam'] =  self.coeffs_dict['cam'][i:i+XK].cuda() # vid_poses[i:i+1].cuda()
            codedictdecoder['light'] = self.coeffs_dict['light'][i:i+XK].cuda() # vid_poses[i:i+1].cuda()
            codedictdecoder['images'] = torch.zeros((XK,3,256,256)).cuda()

            codedictdecoder['pose'][..., 3:4] = torch.clip(coeffs[i:i+XK, 50:51], 0, max_pose_51*0.9) # torch.zeros_like(self.coeffs_dict['pose'][i:i+XK, 3:])
            codedictdecoder['pose'][..., 4:6] = 0 # coeffs[i:i+XK, 50:]*( - 0.25) # torch.zeros_like(self.coeffs_dict['pose'][i:i+XK, 3:])

            sub_vertices = vertices[i:i+XK].cuda()

            opdict = self.spectre.decode_verts(codedictdecoder, sub_vertices, rendering=True, vis_lmk=False, return_vis=False)

            landmark2d.append(opdict['landmarks2d'].cpu())

            all_shape_images.append(opdict['rendered_images'].cpu())

        rendedimages = torch.cat(all_shape_images)

        lmk2d = torch.cat(landmark2d)

        return rendedimages, lmk2d

    
    @torch.no_grad()
    def run_spectre_v3(self, wav=None, ds_features=None, L=20):

        wav = audio_normalize(wav)
        all_mel = self.audio.melspectrogram(wav).astype(np.float32).T
        frames_from_audio = np.arange(2, len(all_mel) // self.audio.num_bins_per_frame - 2) # 2,[]mmmmmmmmmmmmmmmmmmmmmmmmmmmm
        audio_inds = frame2audio_indexs(frames_from_audio, self.audio.num_frames_per_clip, self.audio.num_bins_per_frame)

        vid_exps = self.coeffs_dict['exp'][self.exp_id:self.exp_id+1]
        vid_poses = self.coeffs_dict['pose'][self.exp_id:self.exp_id+1]
        
        ref = torch.cat([vid_exps.view(1, 50), vid_poses[:, 3:].view(1, 3)], dim=-1)
        ref = ref[...,:self.args.exp_dim]

        K = 20
        xlens = len(audio_inds) # len(self.coeffs_dict['exp'])

        exps = []
        for i in tqdm(range(0, xlens, K), desc='S2 DECODER:'+ str(xlens) + ' '):
            
            mels = []
            for j in range(K):
                if i + j < xlens:
                    idx = i+j # //3 * 3 
                    mel = load_spectrogram(all_mel, audio_inds[idx], self.audio.num_frames_per_clip * self.audio.num_bins_per_frame).cuda()
                    mel = mel.view(-1, 1, 80, self.audio.num_frames_per_clip * self.audio.num_bins_per_frame)
                    mels.append(mel)
                else:
                    break

            mels = torch.cat(mels, dim=0)
            new_exp = self.avmodel(mels, ref.repeat(mels.shape[0], 1, 1).cuda(), self.args.use_tanh) # exp 53
            exps+= [new_exp.view(-1, 53)]
            
        all_exps = torch.cat(exps,axis=0)
        
        return all_exps

    @torch.no_grad()
    def test_model(self, wav_path):   
        
        sys.path.append('../FaceFormer')
        from faceformer import Faceformer
        from transformers import Wav2Vec2FeatureExtractor,Wav2Vec2Processor
        from faceformer import PeriodicPositionalEncoding, init_biased_mask
        
        #build model
        self.args.train_subjects = " ".join(["A"]*8) # suitable for pre-trained faceformer checkpoint
        model = Faceformer(self.args)
        model.load_state_dict(torch.load('/apdcephfs/private_shadowcun/Avatar2dFF/medias/videos/c8/mask5000_l2/6_model.pth'))  # ../packages/pretrained/28_ff_model.pth
        model = model.to(torch.device(self.device))
        model.eval()

        # hacking for long audio generation
        model.PPE = PeriodicPositionalEncoding(self.args.feature_dim, period = self.args.period, max_seq_len=6000).cuda()
        model.biased_mask = init_biased_mask(n_head = 4, max_seq_len = 6000, period=self.args.period).cuda()

        train_subjects_list = ["A"] * 8

        one_hot_labels = np.eye(len(train_subjects_list))
        one_hot = one_hot_labels[0]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        one_hot = torch.FloatTensor(one_hot).to(device=self.device)

        vertices_npy = np.load(self.args.source_dir + '/mesh_pose0.npy')
        vertices_npy = np.array(vertices_npy).reshape(-1, 5023*3)

        temp = vertices_npy[33] # 829

        template = temp.reshape((-1))
        template = np.reshape(template,(-1,template.shape[0]))
        template = torch.FloatTensor(template).to(device=self.device)

        speech_array, sampling_rate = librosa.load(os.path.join(wav_path), sr=16000)
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
        audio_feature = np.squeeze(processor(speech_array,sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(device=self.device)
 
        prediction = model.predict(audio_feature, template, one_hot, 1.0) # (1, seq_len, V*3)
        
        return prediction.squeeze()

    @torch.no_grad()
    def run(self, face, audio, start_frame=0):

        wav, sr = librosa.load(audio, sr=16000) # 16*80 ? 20*80
        wav_tensor = torch.FloatTensor(wav).unsqueeze(0) if len(wav.shape) == 1 else torch.FloatTensor(wav)
        _, frames = parse_audio_length(wav_tensor.shape[1], 16000, self.args.fps)

        #####  audio-guided, only use the jaw movement
        all_exps = self.run_spectre_v3(wav)

        # #### temp. interpolation
        all_exps = torch.nn.functional.interpolate(all_exps.unsqueeze(0).permute([0,2,1]), size=frames, mode='linear')
        all_exps = all_exps.permute([0,2,1]).squeeze(0)

        # run faceformer for face mesh generation
        predicted_vertices = self.test_model(audio)
        predicted_vertices = predicted_vertices.view(-1, 5023*3)

        #### temp. interpolation
        predicted_vertices = torch.nn.functional.interpolate(predicted_vertices.unsqueeze(0).permute([0,2,1]), size=frames, mode='linear')
        predicted_vertices = predicted_vertices.permute([0,2,1]).squeeze(0).view(-1, 5023, 3)

        all_exps = torch.Tensor(savgol_filter(all_exps.cpu().numpy(), 5, 3, axis=0)).cpu() # smooth GT
        
        rendedimages, lm2d = self.coeffs_to_img(predicted_vertices, all_exps, zero_pose=True)
        debug_video_gen(rendedimages, self.args.result_dir+"/debug_before_ff.mp4", wav_tensor, self.args.fps, sr)

        # cg2real
        debug_video_gen(self.cg2real(rendedimages, start_frame=start_frame), self.args.result_dir+"/debug_cg2real_raw.mp4", wav_tensor, self.args.fps, sr)
        
        exit()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stylization and Seamless Video Dubbing')
    parser.add_argument('--face', default='examples', type=str, help='')
    parser.add_argument('--audio', default='examples', type=str, help='')
    parser.add_argument('--source_dir', default='examples', type=str,help='TODO')
    parser.add_argument('--result_dir', default='examples', type=str,help='TODO')
    parser.add_argument('--backend', default='wav2lip', type=str,help='wav2lip or pcavs')
    parser.add_argument('--result_tag', default='result', type=str,help='TODO')
    parser.add_argument('--netR', default='unet_256', type=str,help='TODO')
    parser.add_argument('--render_path', default='', type=str,help='TODO')
    parser.add_argument('--ngf', default=16, type=int,help='TODO')
    parser.add_argument('--fps', default=20, type=int,help='TODO')
    parser.add_argument('--mask', default=100, type=int,help='TODO')
    parser.add_argument('--mask_type', default='v3', type=str,help='TODO')
    parser.add_argument('--image_size', default=256, type=int,help='TODO')
    parser.add_argument('--input_nc', default=21, type=int,help='TODO')
    parser.add_argument('--output_nc', default=3, type=int,help='TODO')
    parser.add_argument('--renderbs', default=16, type=int,help='TODO')
    parser.add_argument('--tframes', default=1, type=int,help='TODO')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--enhance', action='store_true')
    parser.add_argument('--phone', action='store_true')

    #### faceformer
    parser.add_argument("--model_name", type=str, default="VOCA")
    parser.add_argument("--dataset", type=str, default="vocaset", help='vocaset or BIWI')
    parser.add_argument("--feature_dim", type=int, default=64, help='64 for vocaset; 128 for BIWI')
    parser.add_argument("--period", type=int, default=30, help='period in PPE - 30 for vocaset; 25 for BIWI')
    parser.add_argument("--vertice_dim", type=int, default=5023*3, help='number of vertices - 5023*3 for vocaset; 23370*3 for BIWI')
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_subjects", type=str, default="FaceTalk_170728_03272_TA ")
    parser.add_argument("--test_subjects", type=str, default="FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA")
    parser.add_argument("--condition", type=str, default="FaceTalk_170904_00128_TA", help='select a conditioning subject from train_subjects')
    parser.add_argument("--subject", type=str, default="FaceTalk_170731_00024_TA", help='select a subject from test_subjects or train_subjects')
    parser.add_argument("--background_black", type=bool, default=True, help='whether to use black background')
    parser.add_argument("--template_path", type=str, default="templates.pkl", help='path of the personalized templates')
    parser.add_argument("--render_template_path", type=str, default="templates", help='path of the mesh in BIWI/FLAME topology')

    opt = parser.parse_args()

    opt.img_size = 96
    opt.static = True
    opt.device = torch.device("cuda")

    a2m = Audio2Mesh(opt)

    print('link start!')
    t = time.time()
    # 02780
    a2m.run(opt.face, opt.audio, 0)
    print(time.time() - t)