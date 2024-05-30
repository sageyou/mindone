
import mindspore as ms
from mindone.diffusers import StableDiffusionXLInstructPix2PixPipeline
from diffusers.utils import load_image

def test_sdxl_ins_pipelie():
    resolution = 768
    image = load_image("mountain.png").resize((resolution, resolution))
    edit_instruction = "Turn sky into a cloudy one"

    pipe = StableDiffusionXLInstructPix2PixPipeline.from_pretrained(
        "sdxl-instructpix2pix-768", 
        mindspore_dtype=ms.float16,
    )

    # edited_image = pipe(
    #     prompt=edit_instruction,
    #     image=image,
    #     height=resolution,
    #     width=resolution,
    #     guidance_scale=3.0,
    #     image_guidance_scale=1.5,
    #     num_inference_steps=2,
    # )[0][0]
    # edited_image.save("res_pipe_ins.png")

if __name__ == "__main__":
    test_sdxl_ins_pipelie()