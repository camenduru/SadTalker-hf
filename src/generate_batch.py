import os
import torch
import numpy as np
import random
import scipy.io as scio
import src.utils.audio as audio
import subprocess, platform

def crop_pad_audio(wav, audio_length):
    if len(wav) > audio_length:
        wav = wav[:audio_length]
    elif len(wav) < audio_length:
        wav = np.pad(wav, [0, audio_length - len(wav)], mode='constant', constant_values=0)
    return wav

def parse_audio_length(audio_length, sr, fps):
    bit_per_frames = sr / fps

    num_frames = int(audio_length / bit_per_frames)
    audio_length = int(num_frames * bit_per_frames)

    return audio_length, num_frames

def generate_blink_seq(num_frames):
    ratio = np.zeros((num_frames,1))
    frame_id = 0
    while frame_id in range(num_frames):
        #start = random.choice(range(60,70))
        start = 80
        if frame_id+start+9<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+9, 0] = [0.5,0.6,0.7,0.9,1, 0.9, 0.7,0.6,0.5]
            frame_id = frame_id+start+9
        else:
            break
    return ratio 

def generate_blink_seq_randomly(num_frames):
    ratio = np.zeros((num_frames,1))
    if num_frames<=20:
        return ratio
    frame_id = 0
    while frame_id in range(num_frames):
        #start = random.choice(range(60,70))
        start = random.choice(range(min(10,num_frames), min(int(num_frames/2), 70))) 
        if frame_id+start+5<=num_frames - 1:
            ratio[frame_id+start:frame_id+start+5, 0] = [0.5, 0.9, 1.0, 0.9, 0.5]
            frame_id = frame_id+start+5
        else:
            break
    return ratio

def get_data(first_coeff_path, audio_path, device):

    syncnet_mel_step_size = 16
    syncnet_T = 5
    MAX_FRAME = 32
    fps = 25

    pic_name = os.path.splitext(os.path.split(first_coeff_path)[-1])[0]
    audio_name = os.path.splitext(os.path.split(audio_path)[-1])[0]

    source_semantics_path = first_coeff_path
    source_semantics_dict = scio.loadmat(source_semantics_path)
    ref_coeff = source_semantics_dict['coeff_3dmm'][:1,:70]         #1 70

    if '.mp3' in audio_path:
        new_audio = audio_path.replace('.mp3', '.wav')
        command = 'ffmpeg -i '+ audio_path + ' -acodec pcm_s16le -ac 1 -ar 16000 ' + new_audio
        subprocess.call(command, shell=platform.system() != 'Windows')
    else:
        new_audio = audio_path

    wav = audio.load_wav(new_audio, 16000) 
            
    wav_length, num_frames = parse_audio_length(len(wav), 16000, 25)
    wav = crop_pad_audio(wav, wav_length)
    orig_mel = audio.melspectrogram(wav).T
    spec = orig_mel.copy()         # nframes 80
    indiv_mels = []

    for i in range(num_frames):
        start_frame_num = i-2
        start_idx = int(80. * (start_frame_num / float(fps)))
        end_idx = start_idx + syncnet_mel_step_size
        seq = list(range(start_idx, end_idx))
        seq = [ min(max(item, 0), orig_mel.shape[0]-1) for item in seq ]
        m = spec[seq, :]
        indiv_mels.append(m.T)
    indiv_mels = np.asarray(indiv_mels)         # T 80 16
    ratio = generate_blink_seq_randomly(num_frames)      # T

    
    indiv_mels = torch.FloatTensor(indiv_mels).unsqueeze(1).unsqueeze(0) # bs T 1 80 16
    ratio = torch.FloatTensor(ratio).unsqueeze(0)                        # bs T
    ref_coeff = torch.FloatTensor(ref_coeff).unsqueeze(0)                # bs 1 70

    indiv_mels = indiv_mels.to(device)
    ratio = ratio.to(device)
    ref_coeff = ref_coeff.to(device)

    return {'indiv_mels': indiv_mels,  
            'ref': ref_coeff, 
            'num_frames': num_frames, 
            'ratio_gt': ratio,
            'audio_name': audio_name, 'pic_name': pic_name}

