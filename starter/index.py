from PIL import Image
from diffusers import DDPMPipeline
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", DEVICE)
ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to(DEVICE)
image = ddpm(num_inference_steps=25).images[0]
image.show()

from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, PNDMScheduler

vae = AutoencoderKL.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="vae"
).to(DEVICE)
tokenizer = CLIPTokenizer.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="tokenizer"
).to(DEVICE)
text_encoder = CLIPTextModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="text_encoder"
).to(DEVICE)
unet = UNet2DConditionModel.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="unet"
).to(DEVICE)

from diffusers import UniPCMultistepScheduler

scheduler = UniPCMultistepScheduler.from_pretrained(
    "CompVis/stable-diffusion-v1-4", subfolder="scheduler"
).to(DEVICE)

#  Define the prompt and other parameters
prompt = ["a photograph of an astronaut riding a horse"]
height = 512
width = 512
num_inference_steps = 25
guidance_scale = 7.5
generator = torch.manual_seed(0)
batch_size = len(prompt)

text_input = tokenizer(
    prompt,
    padding="max_length",
    max_length=tokenizer.model_max_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)
with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids).to(DEVICE)

max_length = text_input.input_ids.shape[1]
uncond_input = tokenizer(
    [""] * batch_size,
    padding="max_length",
    max_length=max_length,
    truncation=True,
    return_tensors="pt",
).to(DEVICE)
uncond_embeddings = text_encoder(uncond_input.input_ids).to(DEVICE)

text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

latents = torch.randn(
    (batch_size, unet.in_channels, height // 8, width // 8), generator=generator
)
latents = latents.to(DEVICE)

latents = latents * scheduler.init_noise_sigma
from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)
for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(
            latent_model_input, t, encoder_hidden_states=text_embeddings
        ).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (
        noise_pred_text - noise_pred_uncond
    )

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    pil_images = [Image.fromarray(img) for img in images]


from diffusers import AutoPipelineForText2Image

pipe_txt2img = AutoPipelineForText2Image.from_pretrained(
    "dreamlike-art/dreamlike-photoreal-2.0",
    torch_dtype=torch.float16,
    use_safetensor=True,
).to(DEVICE)
prompt = "cinematic photo of Godzilla in the city"
generator = torch.Generator(device=DEVICE).manual_seed(0)
# The AutoPipeline supports Stable Diffusion, Stable Diffusion XL, ControlNet, Kandinsky 2.1, Kandinsky 2.2, and DeepFloyd IF checkpoints.
# If you try to load an unsupported checkpoint, you’ll get an error.
image = pipe_txt2img(prompt, generator=generator)

# Reuse components
from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline

model_id = "runway/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id)
components = stable_diffusion_txt2img.components
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(**components)

model_id = "runwayml/stable-diffusion-v1-5"
stable_diffusion_txt2img = StableDiffusionPipeline.from_pretrained(model_id)
stable_diffusion_img2img = StableDiffusionImg2ImgPipeline(
    vae=stable_diffusion_txt2img.vae,
    text_encoder=stable_diffusion_txt2img.text_encoder,
    tokenizer=stable_diffusion_txt2img.tokenizer,
    unet=stable_diffusion_txt2img.unet,
    scheduler=stable_diffusion_txt2img.scheduler,
    safety_checker=None,
    feature_extractor=None,
    requires_safety_checker=False,
)

from diffusers import DiffusionPipeline
import torch

# load fp16 variant
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16
)
stable_diffusion = DiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="non_ema"
)
# load non_ema variant
stable_diffusion.save_pretrained("./stable-diffusion-v1-5", variant="non_ema")

# save as fp16 variant
stable_diffusion.save_pretrained(
    "runwayml/stable-diffusion-v1-5", variant="fp16", torch_dtype=torch.float16
)
# save as non_ema variant
stable_diffusion.save_pretrained("runwayml/stable-diffusion-v1-5", variant="non_ema")

from diffusers import DiffusionPipeline

pipe = DiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4", revision="fp16"
)  # for github use


# Models
# 模型从 ModelMixin.from_pretrained()方法加载，该方法下载并缓存模型权重和配置的最新版本。
# 如果最新文件已存在于本地缓存中，from_pretrained()将重用缓存中的文件而不是重新下载
# 模型可以从具有 subfolder 参数的子文件夹中加载。
# 例如， runwayml/stable-diffusion-v1-5 的模型权重存储在 unet 子文件夹中
from diffusers import UNet2DModel, UNet2DConditionModel

repo_id = "runwayml/stable-diffusion-v1-5"
model = UNet2DConditionModel.from_pretrained(repo_id, subfolder="unet")

repo_id = "google/ddpm-cifar10-32"
model = UNet2DModel.from_pretrained(repo_id)

model = UNet2DConditionModel.from_pretrained(
    "runwayml/stable-diffusion-v1-5", subfolder="unet", variant="non-ema"
)
model.save_pretrained("./local-unet", variant="non-ema")

# schedulers
# 调度器从 SchedulerMixin.from_pretrained()方法加载，
# 与模型不同，调度器没有参数化或训练；它们由配置文件定义。
# 加载调度器不会消耗大量内存，并且相同的配置文件可用于各种不同的调度器。
# 例如，以下调度器与 StableDiffusionPipeline 兼容，这意味着您可以在这些类中的任何一个中加载相同的调度器配置文件
from diffusers import StableDiffusionPipeline
from diffusers import (
    DDPMScheduler,
    DDIMScheduler,
    PNDMScheduler,
    LMSDiscreteScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    DPMSolverMultistepScheduler,
)

repo_id = "runwayml/stable-diffusion-v1-5"
ddpm = DDPMScheduler.from_pretrained(repo_id, subfolder="scheduler")
ddim = DDIMScheduler.from_pretrained(repo_id, subfolder="scheduler")
pndm = PNDMScheduler.from_pretrained(repo_id, subfolder="scheduler")
lms = LMSDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
euler_anc = EulerAncestralDiscreteScheduler.from_pretrained(
    repo_id, subfolder="scheduler"
)
euler = EulerDiscreteScheduler.from_pretrained(repo_id, subfolder="scheduler")
dpm = DPMSolverMultistepScheduler.from_pretrained(repo_id, subfolder="scheduler")

pipeline = StableDiffusionPipeline.from_pretrained(repo_id, scheduler=dpm)

from diffusers import DiffusionPipeline

repo_id = "runwayml/stable-diffusion-v1-5"
pipeline = DiffusionPipeline.from_pretrained(repo_id)
pipeline.scheduler = LMSDiscreteScheduler.from_config(pipeline.scheduler.config)
image = pipeline("a photo of a cat").images[0]

# Models
unet = UNet2DConditionModel.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    subfolder="unet",
    use_safetensors=True,
)
# Safetensors 是一种安全且快速的文件格式，用于安全存储和加载张量。
# Safetensors 限制头部大小以限制某些类型的攻击，支持懒加载（适用于分布式设置），并且通常具有更快的加载速度。
pipeline = DiffusionPipeline.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", use_safetensors=True
)

from diffusers import StableDiffusionXLPipeline

# LoRA files
# LoRA 是一种轻量级的适配器，训练速度快且简单，因此特别适合以特定方式或风格生成图像。
# 这些适配器通常存储在 safetensors 文件中，在模型共享平台如 civitai 上非常受欢迎。
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "Lykon/dreamshaper-xl-1-0", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pipeline.load_lora_weights(".", weight_name="blueprintify.safetensors")
prompt = "a photo of a cat"
negative_prompt = "a photo of a dog"
image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    generator=torch.manual_seed(0),
).images[0]

pipeline = StableDiffusionPipeline.from_single_file(
    "https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/blob/main/v1-5-pruned.ckpt"
)
