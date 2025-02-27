# 有几种训练技巧可以个性化扩散模型以生成特定主题或特定风格的图像。
# 这些训练方法中的每一种都产生不同类型的适配器。一些适配器生成一个全新的模型，而其他适配器仅修改较小的嵌入或权重集。这意味着每个适配器的加载过程也不同

# DreamBooth
# DreamBooth 通过仅对主题的几张图片进行微调，来生成该主题的新风格和场景的图像。
# 这种方法通过在提示词中使用一个特殊的词，让模型学会将其与主题图像关联起来。在所有训练方法中，DreamBooth 产生的文件大小最大（通常几 GB），因为它是一个完整的检查点模型。
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "sd-dreambooth-library/herge-styled", torch_dtype=torch.float16
).to("cuda")


prompt = "A detailed description of the desired image"
image = pipeline(prompt).images[0]


from diffusers import AutoPipelineForText2Image

# textual inversion
# 文本反转与 DreamBooth 非常相似，它也可以通过少量图像来个性化扩散模型以生成特定概念（风格、物体）。
# 该方法通过训练和寻找新的嵌入来表示您提供的图像，并在提示中使用一个特殊词汇。因此，扩散模型的权重保持不变，训练过程产生一个相对较小的（几 KB）文件。

# 因为文本反转创建嵌入，它不能单独使用，如 DreamBooth，需要另一个模型。
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipeline.load_textual_inversion("sd-concepts-library/gta5-artwork")
prompt = "A detailed description of the desired image"
image = pipeline(prompt).images[0]
# 文本反转也可以用于训练不希望出现的事物，以创建负面嵌入，从而阻止模型生成包含这些不希望出现的事物（如模糊的图像或手上的多余手指）的图像。
# 这可以是一种快速提高你的提示的方法。你还将使用 load_textual_inversion() 函数加载嵌入，但这次需要两个额外的参数
# weight_name:指定加载的权重文件，如果文件以特定名称保存在 🤗 Diffusers 格式或存储在 A1111 格式中
# token: 指定在提示中使用的特殊词以触发嵌入

prompt = "A cute cat with three eyes"
negative_prompt = "A cute cat with four eyes"
image = pipeline(
    prompt, negative_prompt=negative_prompt, num_inference_steps=50
).images[0]

# LoRA 是一种流行的训练技术，因为它速度快且生成的文件大小更小（约几百 MB）。
# 与LoRA 可以使模型仅从少量图像中学习新风格。它通过向扩散模型中插入新权重来实现，然后只训练新权重而不是整个模型。这使得 LoRA 的训练速度更快，存储更容易
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")
pipeline.load_lora_weights(
    "ostris/super-cereal-sdxl-lora", weight_name="cereal_box_sdxl_v1.safetensors"
)
prompt = "bears, pizza bites"
image = pipeline(prompt).images[0]


# The load_lora_weights() 方法将 LoRA 权重加载到 UNet 和文本编码器中。
# 这是加载 LoRAs 的首选方法，因为它可以处理以下情况：
# LoRA 权重没有为 UNet 和文本编码器分别设置标识符
# LoRA 权重为 UNet 和文本编码器分别设置了标识符

# 使用 weight_name 参数指定特定的权重文件，并使用 prefix 参数筛选适当的 state dicts（在这种情况下为 "unet" ）以加载。

from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

pipeline.unet.load_lora_adapter(
    "jbilcke-hf/sdxl-cinematic-1",
    weight_name="pytorch_lora_weights.safetensors",
    prefix="unet",
)

#  use cnmt in the prompt to trigger the LoRA
prompt = "A cute cnmt eating a slice"
image = pipeline(prompt).images[0]

# 对于 load_lora_weights() 和 load_attn_procs()，
# 可以通过 cross_attention_kwargs={"scale": 0.5} 参数来调整使用多少 LoRA 权重。 0 的值等同于仅使用基础模型权重，而 1 的值相当于使用完全微调的 LoRA。


# IP-适配器是一个轻量级的适配器，它使任何扩散模型都能进行图像提示。
# 该适配器通过解耦图像和文本特征的交叉注意力层来工作。所有其他模型组件都被冻结，只有 UNet 中的嵌入图像特征被训练。因此，IP-适配器文件通常只有约 100MB。
from diffusers import AutoPipelineForText2Image
import torch
from diffusers import load_image

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5", torch_dtype=torch.float16
).to("cuda")
pipeline.load_ip_adapter(
    "h94/IP-Adapter", subfolder="models", weight_name="ip-adapter_sd15.bin"
)
image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/load_neg_embed.png"
)
images = pipeline(
    prompt="best quality, high quality",
    ip_adapter_image=image,
    negative_prompt="worst quality, low quality",
    num_inference_steps=50,
).images[0]
from transformers import CLIPVisionModelWithProjection

image_encoder = CLIPVisionModelWithProjection.from_pretrained(
    "h94/IP-Adapter", subfolder="models/image_encoder", torch_dtype=torch.float16
)

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0",
    torch_dtype=torch.float16,
    image_encoder=image_encoder,
).to("cuda")
pipeline.load_ip_adapter(
    "h94/IP-Adapter",
    subfolder="sdxl_models",
    weight_name="ip-adapter-plus_sdxl_vit-h.safetensors",
)

# CnotrolNet
# 控制网模型是辅助模型或适配器，它们在文本到图像模型（如 Stable Diffusion v1.5）之上进行微调。使用控制网模型与文本到图像模型结合，提供了更多显式控制图像生成方式的选择。使用控制网时，您向模型添加一个额外的条件输入图像。
# 例如，如果您提供一个表示人类姿态的图像（通常表示为多个连接成骨骼的关键点）作为条件输入，模型将生成遵循该姿态的图像。查看更深入的控制网指南，了解其他条件输入及其使用方法

from diffusers import ControlNetModel, AutoPipelineForText2Image
from diffusers.utils import load_image

controlnet = ControlNetModel.from_pretrained(
    "llyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, variant="fp16"
).to("cuda")

pose_image = load_image(
    "https://huggingface.co/lllyasviel/control_v11p_sd15_openpose/resolve/main/images/control.png"
)
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

# guidance scale
# guidance_scale 参数影响提示对图像生成的影响程度。
# 较低的值赋予模型“创造力”，生成与提示更松散相关的图像。较高的 guidance_scale 值推动模型更紧密地遵循提示，如果这个值太高，你可能会在生成的图像中观察到一些伪影。
