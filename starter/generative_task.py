# unconditional image generation

from diffusers import DiffusionPipeline

MODEL_NAME = "anton-l/ddpm-butterflies-128"
generator = DiffusionPipeline.from_pretrained("anton-l/ddpm-butterflies-128").to("cuda")
# 输出图像是一个可保存的 PIL.Image 对象
image = generator().images[0]

# 也可以尝试使用 num_inference_steps 参数进行实验
# 参数控制去噪步骤的数量。通常，更多的去噪步骤会产生更高品质的图像，但生成时间会更长。请随意调整此参数，看看它如何影响图像质量。
image = generator(num_inference_steps=100).images[0]


# Text-to-image
from diffusers import AutoPipelineForText2Image
import torch

pipeline = AutoPipelineForText2Image.from_pretrained(
    "stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16
).to("cuda")

# 从非常高的层面来看，扩散模型接受一个提示和一些随机的初始噪声，并通过迭代去除噪声来构建图像。
# 去噪过程由提示引导，一旦经过预定的时间步数后去噪过程结束，图像表示就被解码成图像
prompt = "A beautiful landscape with mountains and a clear blue sky"
image = pipeline(prompt).images[0]

# 最常见的文本到图像模型是 Stable Diffusion v1.5、Stable Diffusion XL（SDXL）和 Kandinsky 2.2。还有可用于与文本到图像模型结合使用以实现更直接控制的 ControlNet 模型或适配器。
# 由于架构和训练过程的不同，每个模型的结果略有差异，但无论选择哪个模型，它们的用法都大同小异。让我们为每个模型使用相同的提示并比较它们的结果。

from diffusers import ControlNetModel, AutoPipelineForText2Image
from diffusers.utils import load_image
import torch

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_openpose", torch_dtype=torch.float16, variant="fp16"
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
generator = torch.Generator("cuda").manual_seed(31)
image = pipeline(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    image=pose_image,
    generator=generator,
).images[0]
# 管道中可以配置多个参数，这些参数会影响图像的生成方式。您可以更改图像的输出大小，指定负提示以提升图像质量等。本节将深入探讨如何使用这些参数。
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
image = pipeline(
    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
    height=768,
    width=512,
).images[0]
# guidance scale
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
image = pipeline(
    "Astronaut in a jungle cold color , muted colors, detailed, 8k", guidance_scale=3.5
).images[0]


pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")

prompt = "A beautiful night sky filled with stars"
negative_prompt = "blurry, low resolution, out of focus"
image = pipeline(prompt, negative_prompt=negative_prompt).images[0]

# 有几种方法可以在配置管道参数之外，更多地控制图像的生成，例如提示权重和控制网络模型。

# 提示权重是一种技术，用于增加或减少提示中概念的重要性，以强调或最小化图像中的某些特征。建议使用 Compel 库来帮助您生成加权提示嵌入。

# 扩散模型很大，图像去噪的迭代性质计算成本高且密集。但这并不意味着您需要访问强大的——甚至许多——GPU 来使用它们。有许多优化技术可以在消费级和免费层资源上运行扩散模型。例如，您可以以半精度加载模型权重以节省 GPU 内存并提高速度，或者将整个模型卸载到 GPU 上以节省更多内存。

# PyTorch 2.0 还支持一种更节省内存的注意力机制，称为缩放点积注意力，如果您使用 PyTorch 2.0，则会自动启用。您可以将此与 torch.compile 结合使用，以进一步加快您的代码速度：
pipeline = AutoPipelineForText2Image.from_pretrained(
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    torch_dtype=torch.float16,
    variant="fp16",
).to("cuda")
pipeline.unet = torch.compile(pipeline.unet, mode="reduce-overhead", fullgraph=True)

# image-to-image task
from diffusers import AutoPipelineForImage2Image
from diffusers.utils import load_image, make_image_grid

pipeline = AutoPipelineForImage2Image.from_pretrained(
    "kandinsky-community/kandinsky-2-2-decoder",
    torch_dtype=torch.float16,
    use_safetensors=True,
)
init_image = load_image(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/diffusers/cat.png"
)
prompt = "cat wizard, gandalf, lord of the rings, detailed, fantasy, cute, adorable, Pixar, Disney, 8k"
image = pipeline(prompt, image=init_image).images[0]
