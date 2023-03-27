import os,sys

def gfpgan(scale, origin_mp4_path):
    current_code_path = sys.argv[0]
    current_root_path = os.path.split(current_code_path)[0]
    print(current_root_path)
    gfpgan_code_path = current_root_path+'/repositories/GFPGAN/inference_gfpgan.py'
    print(gfpgan_code_path)
    
    #video2pic
    result_dir = os.path.split(origin_mp4_path)[0]
    video_name = os.path.split(origin_mp4_path)[1]
    video_name = video_name.split('.')[0]
    print(video_name)
    str_scale = str(scale).replace('.', '_')
    output_mp4_path = os.path.join(result_dir, video_name+'##'+str_scale+'.mp4')
    temp_output_mp4_path = os.path.join(result_dir, 'temp_'+video_name+'##'+str_scale+'.mp4')

    audio_name = video_name.split('##')[-1]
    audio_path = os.path.join(result_dir, audio_name+'.wav')
    temp_pic_dir1 = os.path.join(result_dir, video_name)
    temp_pic_dir2 = os.path.join(result_dir, video_name+'##'+str_scale)
    os.makedirs(temp_pic_dir1, exist_ok=True)
    os.makedirs(temp_pic_dir2, exist_ok=True)
    cmd1 = 'ffmpeg -i \"{}\" -start_number 0  \"{}\"/%06d.png -loglevel error -y'.format(origin_mp4_path, temp_pic_dir1)
    os.system(cmd1)
    cmd2 = f'python {gfpgan_code_path} -i {temp_pic_dir1} -o {temp_pic_dir2} -s {scale}'
    os.system(cmd2)
    cmd3 = f'ffmpeg -r 25 -f image2 -i {temp_pic_dir2}/%06d.png  -vcodec libx264 -crf 25  -pix_fmt yuv420p {temp_output_mp4_path}'
    os.system(cmd3)
    cmd4 = f'ffmpeg -y -i {temp_output_mp4_path}  -i {audio_path}  -vcodec copy {output_mp4_path}'
    os.system(cmd4)
    #shutil.rmtree(temp_pic_dir1)
    #shutil.rmtree(temp_pic_dir2)

    return output_mp4_path
