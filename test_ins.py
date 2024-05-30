from mindone.diffusers import  StableDiffusionInstructPix2PixPipeline, StableDiffusionXLInstructPix2PixPipeline
import mindspore as ms

from mindone.diffusers.utils import load_image

# def test_sdxl_ins_pipelie():
#     resolution = 768
#     image = load_image("mountain.png").resize((resolution, resolution))
#     edit_instruction = "Turn sky into a cloudy one"

#     pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
#         "sdxl-instructpix2pix-768", 
#         mindspore_dtype=ms.float16,
#     )

#     # edited_image = pipe(
#     #     prompt=edit_instruction,
#     #     image=image,
#     #     height=resolution,
#     #     width=resolution,
#     #     guidance_scale=3.0,
#     #     image_guidance_scale=1.5,
#     #     num_inference_steps=2,
#     # )[0][0]
#     # edited_image.save("res_pipe_ins.png")

def test_sd_ins_pipelie():
    image = load_image("mountain.png").resize((512, 512))
    pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
        "timbrooksinstruct-pix2pix", mindspore_dtype=ms.float32
    )
    prompt = "make the mountains snowy"
    # image = pipe(prompt=prompt, image=image)[0][0]
    # image.save("res_sd_instrcut.png")

if __name__ == "__main__":
    test_sd_ins_pipelie()