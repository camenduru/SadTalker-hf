import os, sys
import tempfile
import gradio as gr
from modules.text2speech import text2speech 
from modules.gfpgan_inference import gfpgan
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

def sadtalker_demo(result_dir):

    sad_talker = SadTalker()
    with gr.Blocks(analytics_enabled=False) as sadtalker_interface:
        with gr.Row().style(equal_height=False):
            with gr.Column(variant='panel'):
                with gr.Tabs(elem_id="sadtalker_source_image"):
                    source_image = gr.Image(visible=False, type="filepath")
                    with gr.TabItem('Upload image'):
                        with gr.Row():
                            input_image = gr.Image(label="Source image", source="upload", type="filepath").style(height=256,width=256)
                        submit_image = gr.Button('Submit', variant='primary')
                    submit_image.click(fn=get_source_image, inputs=input_image, outputs=source_image)
 
                with gr.Tabs(elem_id="sadtalker_driven_audio"):
                    driven_audio = gr.Audio(visible=False, type="filepath")
                    with gr.TabItem('Upload audio'):
                        with gr.Column(variant='panel'):
                            input_audio1 = gr.Audio(label="Input audio", source="upload", type="filepath")
                            submit_audio_1 = gr.Button('Submit', variant='primary')
                        submit_audio_1.click(fn=get_driven_audio, inputs=input_audio1, outputs=driven_audio)

                    with gr.TabItem('Microphone'):
                        with gr.Column(variant='panel'):
                            input_audio2 = gr.Audio(label="Recording audio", source="microphone", type="filepath")
                            submit_audio_2 = gr.Button('Submit', variant='primary')
                        submit_audio_2.click(fn=get_driven_audio, inputs=input_audio2, outputs=driven_audio)
                    
                    with gr.TabItem('TTS'):
                        with gr.Column(variant='panel'):
                            with gr.Row().style(equal_height=False):
                                input_text = gr.Textbox(label="Input text", lines=5, value="Please enter some text in English")
                                input_audio3 = gr.Audio(label="Generated audio",  type="filepath")
                            submit_audio_3 = gr.Button('Submit', variant='primary')
                        submit_audio_3.click(fn=get_driven_audio, inputs=input_text, outputs=[input_audio3, driven_audio])

            with gr.Column(variant='panel'): 
                gen_video = gr.Video(label="Generated video", format="mp4").style(height=256,width=256)
                gen_text = gr.Textbox(visible=False)
                submit = gr.Button('Generate', elem_id="sadtalker_generate", variant='primary')
                scale = gr.Slider(minimum=1, maximum=8, step=1, label="GFPGAN scale", value=1) 
                new_video = gr.Video(label="New video", format="mp4").style(height=512,width=512)
                change_scale = gr.Button('Restore video', elem_id="restore_video", variant='primary')

        submit.click(
                    fn=sad_talker.test, 
                    inputs=[source_image,
                            driven_audio,
                            gr.Textbox(value=result_dir, visible=False)], 
                    outputs=[gen_video, gen_text]
                    )
        change_scale.click(gfpgan,  [scale, gen_text], new_video)

    return sadtalker_interface
 

if __name__ == "__main__":

    current_code_path = sys.argv[0]
    current_root_dir = os.path.split(current_code_path)[0] 
    sadtalker_result_dir = os.path.join(current_root_dir, 'results', 'sadtalker')
    demo = sadtalker_demo(sadtalker_result_dir)
    demo.launch(share=True)


