import os, sys
import tempfile
import gradio as gr
from modules.text2speech import text2speech 
from modules.sadtalker_test import SadTalker  

def get_driven_audio(audio):  
    if os.path.isfile(audio):
        return audio
    else:
        save_path = tempfile.NamedTemporaryFile(
                delete=False,
                suffix=("." + "wav"),
            )
        gen_audio = text2speech(audio, save_path.name)
        return gen_audio, gen_audio 

def get_source_image(image):   
        return image

def sadtalker_demo(result_dir='./tmp/'):

    sad_talker = SadTalker()
    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        gr.Markdown("<div align='center'> <h3> 😭 SadTalker: Learning Realistic 3D Motion Coefficients for Stylized Audio-Driven Single Image Talking Face Animation (CVPR 2023) </h3> \
                    <a style='font-size:18px;color: #efefef' href='https://arxiv.org/abs/2211.12194'>Arxiv</a> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                    <a style='font-size:18px;color: #efefef' href='https://sadtalker.github.io'>Homepage</a>  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; \
                     <a style='font-size:18px;color: #efefef' href='https://github.com/Winfredy/SadTalker'> Github </a> </div>")
        
        with gr.Row():
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            source_image = gr.Image(label="Source image", source="upload", type="filepath").style(height=256)
 
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    with gr.TabItem('Upload audio(wav/mp3 only currently)'):
                        with gr.Column(variant='panel'):
                            driven_audio = gr.Audio(label="Input audio", source="upload", type="filepath")

            with gr.Column(variant='panel'): 
                with gr.Tabs(elem_id="sadtalker_checkbox"):
                    with gr.TabItem('Settings'):
                        with gr.Column(variant='panel'):
                            is_still_mode = gr.Checkbox(label="Still Mode (fewer head motion)").style(container=True)
                            is_resize_mode = gr.Checkbox(label="Resize Mode (⚠️ Resize mode need manually crop the image firstly, can handle larger image crop)").style(container=True)
                            is_enhance_mode = gr.Checkbox(label="Enhance Mode (better face quality )").style(container=True)
                            submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')

                with gr.Tabs(elem_id="sadtalker_genearted"):
                        gen_video = gr.Video(label="Generated video", format="mp4").style(width=256)
                        gen_text = gr.Textbox(visible=False)
                    
        with gr.Row():
            examples = [
                [
                    'examples/source_image/art_10.png',
                    'examples/driven_audio/deyu.wav',
                    True,
                    False,
                    False
                ],
                [
                    'examples/source_image/art_1.png',
                    'examples/driven_audio/fayu.wav',
                    True,
                    True,
                    False
                ],
                [
                    'examples/source_image/art_9.png',
                    'examples/driven_audio/itosinger1.wav',
                    True,
                    False,
                    True
                ]
            ]
            gr.Examples(examples=examples,
                        inputs=[
                            source_image,
                            driven_audio,
                            is_still_mode,
                            is_resize_mode,
                            is_enhance_mode,
                            gr.Textbox(value=result_dir, visible=False)], 
                        outputs=[gen_video, gen_text],
                        fn=sad_talker.test,
                        cache_examples=os.getenv('SYSTEM') == 'spaces')

        submit.click(
                    fn=sad_talker.test, 
                    inputs=[source_image,
                            driven_audio,
                            is_still_mode,
                            is_resize_mode,
                            is_enhance_mode,
                            gr.Textbox(value=result_dir, visible=False)], 
                    outputs=[gen_video, gen_text]
                    )

    return sadtalker_interface
 

if __name__ == "__main__":

    sadtalker_result_dir = os.path.join('./', 'results')
    demo = sadtalker_demo(sadtalker_result_dir)
    demo.launch()


