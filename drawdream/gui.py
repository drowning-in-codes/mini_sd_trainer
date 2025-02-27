import gradio as gr
from diffusers import (
    DiffusionPipeline,
    AutoPipelineForText2Image,
    ControlNetModel,
    AutoPipelineForImage2Image,
    StableDiffusionUpscalePipeline,
    StableDiffusionLatentUpscalePipeline,
    AutoPipelineForInpainting,
)
import os
from transformers import AutoTokenizer
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_community.document_loaders import PyPDFLoader
from diffusers import CogVideoXImageToVideoPipeline
from diffusers.utils import export_to_video, load_image
from PIL import Image, ImageChops
from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser
import torch
from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def check_setting():
    if not torch.cuda.is_available():
        gr.Warning("没有检测到可使用的CUDA设备,为了更好地使用模型,请使用有CUDA的设备.")


with gr.Blocks() as app:
    with gr.Group():
        gr.Markdown("# 设置")
        enable_optimized = gr.Checkbox(label="启用优化", value=False)
    gr.Markdown("# 文生图应用")
    with gr.Tab("无条件图像生成"):
        model_name = gr.Textbox(
            value="", placeholder="anton-l/ddpm-butterflies-128", label="模型名称"
        )
        num_inference_steps = gr.Textbox(value=10, placeholder="10", label="推理步数")
        generate_button = gr.Button("生成图像")

        output_image = gr.Image()

        def generate_image(inference_steps):
            if not inference_steps.isdigit() or int(inference_steps) <= 0:
                return gr.Error("请输入一个有效的正整数")
            try:
                pipeline = DiffusionPipeline.from_pretrained(model_name.value).to(
                    DEVICE
                )
                image = pipeline(num_inference_steps=int(inference_steps)).images[0]
                return image
            except Exception as e:
                return gr.Error(f"生成图像时遇到了错误: {str(e)}")

        generate_button.click(
            generate_image, inputs=num_inference_steps, outputs=output_image
        )

        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: output_image.update(value=None), None, output_image)

        generate_button.click(
            lambda: output_image.update(value=None, tooltips="生成中..."),
            inputs=num_inference_steps,
            outputs=output_image,
        )
    with gr.Tab("文本生成图像"):
        model_name = gr.Textbox(
            value="",
            placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
            label="模型名称",
        )
        prompt = gr.Textbox(placeholder="请输入图像描述", label="提示文本")
        t2i_examples = gr.Examples(
            examples=[
                [
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "stabilityai/stable-diffusion-xl-base-1.0",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "kandinsky-community/kandinsky-2-2-decoder",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
            ],
            inputs=[model_name, prompt],
        )
        negative_prompt = gr.Textbox(placeholder="请输入负面描述", label="负面提示文本")
        height_width_config = {
            "与模型配置相同": 0,
            "512x512": 512,
            "768x768": 768,
            "1024x1024": 1024,
        }
        height_width = gr.Radio(
            list(height_width_config.keys()),
            value="与模型配置相同",
            type="index",
            label="图像大小",
        )
        with gr.Row():
            with gr.Column():
                enable_guidance = gr.Checkbox(
                    value=False,
                    label="启用引导",
                    info="影响文本提示对图像生成的影响程度",
                )
                guidance_scale = gr.Textbox(
                    value="7.5",
                    placeholder="请输入导引比例",
                    label="导引比例",
                    info="较低的值赋予模型“创造力”，生成与提示更松散相关的图像",
                )
        with gr.Column():
            enable_repro = gr.Checkbox(
                value=False, label="启用复现", info="固定随机种子"
            )
            seed = gr.Textbox(
                value=2025, placeholder="请输入随机种子", label="随机种子"
            )

        generate_button = gr.Button("生成图像")
        output_image = gr.Image()
        enable_controlnet = gr.Checkbox(value=False, label="启用controlnet")

        controlnet_model_name = gr.Textbox(
            value="lllyasviel/control_v11p_sd15_openpose",
            placeholder="lllyasviel/control_v11p_sd15_openpose",
            label="controlnet模型",
        )
        ref_image = gr.Image(
            label="上传controlnet图像", interactive=False, visible=False
        )

        def update_image_upload(enable_controlnet):
            if enable_controlnet:
                # 如果复选框被选中，则启用图像上传组件
                return gr.Image(interactive=True, visible=True)
            else:
                # 如果复选框未被选中，则禁用图像上传组件
                return gr.Image(interactive=False, visible=False)

        # 当复选框的值发生变化时，调用update_image_upload函数，并更新图像上传组件的状态
        enable_controlnet.change(
            fn=update_image_upload, inputs=enable_controlnet, outputs=ref_image
        )

        def generate_image(
            model_name,
            prompt,
            enable_controlnet,
            controlnet_model_name,
            ref_image,
            negative_prompt,
            height_width,
            enable_guidance,
            guidance_scale,
            enable_repro,
            seed,
        ):
            if prompt.strip() == "":
                return gr.Error("请输入有效的提示文本")
            try:
                if enable_controlnet:
                    controlnet = ControlNetModel.from_pretrained(
                        controlnet_model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    ).to(DEVICE)

                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_name,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        image=ref_image,
                    ).to(DEVICE)
                else:
                    pipeline = AutoPipelineForText2Image.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        negative_prompt=negative_prompt,
                    ).to(DEVICE)
                if height_width != 0:
                    if enable_repro:
                        image = pipeline(
                            prompt,
                            height=height_width,
                            width=height_width,
                            guidance_scale=(
                                float(guidance_scale) if enable_guidance else None
                            ),
                            negative_prompt=negative_prompt,
                            generator=torch.Generator("cpu").manual_seed(seed),
                        ).images[0]
                    else:
                        image = pipeline(
                            prompt,
                            height=height_width,
                            width=height_width,
                            guidance_scale=(
                                float(guidance_scale) if enable_guidance else None
                            ),
                            negative_prompt=negative_prompt,
                        ).images[0]
                else:
                    if enable_repro:
                        image = pipeline(
                            prompt,
                            negative_prompt=negative_prompt,
                            generator=torch.Generator("cpu").manual_seed(seed),
                        ).images[0]
                    else:
                        image = pipeline(
                            prompt, negative_prompt=negative_prompt
                        ).images[0]
                return image

            except Exception as e:
                return gr.Error(f"生成图像时遇到了错误: {str(e)}")

        generate_button.click(
            generate_image,
            inputs=[
                model_name,
                prompt,
                enable_controlnet,
                controlnet_model_name,
                ref_image,
                negative_prompt,
                height_width,
                enable_guidance,
                guidance_scale,
                enable_repro,
                seed,
            ],
            outputs=output_image,
        )

        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: output_image.update(value=None), None, output_image)
    with gr.Tab("图像生成图像"):
        model_name = gr.Textbox(
            value="",
            placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
            label="模型名称",
        )
        prompt = gr.Textbox(value="", placeholder="请输入提示文本", label="提示文本")
        negative_prompt = gr.Textbox(
            value="", placeholder="输入负面提示", label="负面提示"
        )
        i2i_examples = gr.Examples(
            examples=[
                [
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "stabilityai/stable-diffusion-xl-refiner-1.0",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "kandinsky-community/kandinsky-2-2-decoder",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
            ],
            inputs=[model_name, prompt],
        )
        init_image = gr.Image(label="初始图像")

        height_width_config = {
            "与模型配置相同": 0,
            "512x512": 512,
            "768x768": 768,
            "1024x1024": 1024,
        }
        height_width = gr.Radio(
            list(height_width_config.keys()),
            value="与模型配置相同",
            type="index",
            label="图像大小",
        )
        with gr.Column():
            with gr.Row():
                strength = gr.Slider(
                    minimum=0,
                    maximum=1,
                    value=0.5,
                    label="强度",
                    info="决定了生成图像与初始图像的相似程度",
                )
                guidance_scale = gr.Slider(
                    minimum=0,
                    maximum=20,
                    step=0.1,
                    value=7.5,
                    label="引导强度",
                    info="用于调节生成图像的细节与多样性",
                )
            with gr.Row():
                enable_repro = gr.Checkbox(
                    value=False, label="启用重现", info="设置固定种子"
                )
                seed = gr.Textbox(value=2025, label="固定种子")
            with gr.Row():
                enable_controlnet = gr.Checkbox(value=False, label="启用ControlNet")
                controlnet_model_name = gr.Textbox(
                    value="lllyasviel/control_v11f1p_sd15_depth",
                    placeholder="lllyasviel/control_v11f1p_sd15_depth",
                    label="Controlnet模型",
                )

            ref_img = gr.Image(label="Controlnet图像")
        enable_controlnet.change(
            fn=lambda enable_: gr.Image(interactive=enable_, visible=enable_),
            inputs=enable_controlnet,
            outputs=ref_img,
        )
        generate_button = gr.Button("生成图像")
        output_image = gr.Image()

        def generate_image(
            model_name,
            prompt,
            init_image,
            negative_prompt,
            height_width,
            guidance_scale,
            enable_repro,
            seed,
        ):

            if prompt.strip() == "" or init_image == None:
                gr.Error("文本提示和图像输入错误")
                return
            else:
                try:
                    pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    ).to(DEVICE)
                    if height_width != 0:
                        if enable_repro:
                            image = pipeline(
                                prompt,
                                image=init_image,
                                height=height_width,
                                width=height_width,
                                strength=strength,
                                guidance_scale=float(guidance_scale),
                                negative_prompt=negative_prompt,
                                generator=torch.Generator("cpu").manual_seed(seed),
                            ).images[0]
                        else:
                            image = pipeline(
                                prompt,
                                image=init_image,
                                strength=strength,
                                guidance_scale=float(guidance_scale),
                                negative_prompt=negative_prompt,
                            ).images[0]
                    else:
                        if enable_repro:
                            image = pipeline(
                                prompt,
                                image=init_image,
                                strength=strength,
                                negative_prompt=negative_prompt,
                                generator=torch.Generator("cpu").manual_seed(seed),
                            ).images[0]
                        else:
                            image = pipeline(
                                prompt,
                                image=init_image,
                                strength=strength,
                                negative_prompt=negative_prompt,
                            ).images[0]
                    return image

                except Exception as e:
                    return gr.Error(f"生成图像时遇到了错误: {str(e)}")

        generate_button.click(
            generate_image,
            inputs=[
                model_name,
                prompt,
                init_image,
                negative_prompt,
                height_width,
                guidance_scale,
                enable_repro,
                seed,
            ],
            outputs=output_image,
        )

        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: output_image.update(value=None), None, output_image)

        # upscale image
        with gr.Group():
            gr.Markdown("# 高清修复")
            upscaler_model = gr.Textbox(
                value="stabilityai/sd-x2-latent-upscaler",
                placeholder="stabilityai/sd-x2-latent-upscaler",
                label="超分模型",
            )
            enhance_res_model = gr.Textbox(
                value="stabilityai/stable-diffusion-x4-upscaler",
                placeholder="stabilityai/stable-diffusion-x4-upscaler",
                label="修复模型",
            )
            before_scale_image = gr.Image(label="修复前图像")
            upscale_image = gr.Image(label="修复后图像")
            upscale_button = gr.Button("修复图像")

            def upscale_function(prompt, image):
                upscaler = StableDiffusionLatentUpscalePipeline.from_pretrained(
                    "",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )
                upscale_img = upscaler(
                    prompt, image=image, output_type="latent"
                ).images[0]
                super_res = StableDiffusionUpscalePipeline.from_pretrained(
                    "",
                    torch_dtype=torch.float16,
                    variant="fp16",
                    use_safetensors=True,
                )

                result = super_res(prompt, upscale_img)

                return result

            upscale_button.click(
                upscale_function,
                inputs=[before_scale_image, upscaler_model, enhance_res_model],
                outputs=upscale_image,
            )
    with gr.Tab("Inpainting"):
        gr.Markdown("# 内绘")
        model_name = gr.Textbox(
            value="kandinsky-community/kandinsky-2-2-decoder-inpaint",
            placeholder="kandinsky-community/kandinsky-2-2-decoder-inpaint",
            label="模型名称",
        )
        prompt = gr.Textbox(placeholder="请输入图像描述", label="提示文本")
        t2i_examples = gr.Examples(
            examples=[
                [
                    "runwayml/stable-diffusion-inpainting",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "kandinsky-community/kandinsky-2-2-decoder-inpaint",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
                [
                    "diffusers/stable-diffusion-xl-1.0-inpainting-0.1",
                    "Astronaut in a jungle, cold color palette, muted colors, detailed, 8k",
                ],
            ],
            inputs=[model_name, prompt],
        )
        negative_prompt = gr.Textbox(placeholder="请输入负面描述", label="负面提示文本")
        with gr.Row():
            with gr.Column():
                enable_guidance = gr.Checkbox(
                    value=False,
                    label="启用引导",
                    info="影响文本提示对图像生成的影响程度",
                )
                guidance_scale = gr.Textbox(
                    value="7.5",
                    placeholder="请输入导引比例",
                    label="导引比例",
                    info="较低的值赋予模型“创造力”，生成与提示更松散相关的图像",
                )
                enable_strength = gr.Checkbox(
                    value=False, label="强度", info="衡量添加到基础图像的噪声程度的指标"
                )
                strength = gr.Textbox(value="0.5", placeholder="0.5", label="强度1")
                """
                模糊量由 blur_factor 参数确定。增加 blur_factor 会增加应用于蒙版边缘的模糊量，使原始图像和修复区域之间的过渡更加柔和。
                低或零 blur_factor 会保留蒙版的锐利边缘。
                """
                enable_blur = gr.Checkbox(value=False, label="启用mask模糊因子")
                blur_factor = gr.Textbox(
                    value="0.5",
                    placeholder="请输入模糊因子",
                    label="模糊因子",
                    info="控制图像的模糊程度，范围从 0 到 1",
                )
                editor_output = gr.ImageEditor(label="mask图片")

        with gr.Column():
            enable_repro = gr.Checkbox(
                value=False, label="启用复现", info="固定随机种子"
            )
            seed = gr.Textbox(
                value=2025, placeholder="请输入随机种子", label="随机种子"
            )

        generate_button = gr.Button("生成图像")
        output_image = gr.Image()
        enable_controlnet = gr.Checkbox(value=False, label="启用controlnet")

        controlnet_model_name = gr.Textbox(
            value="lllyasviel/control_v11p_sd15_openpose",
            placeholder="lllyasviel/control_v11p_sd15_openpose",
            label="controlnet模型",
        )
        ref_image = gr.Image(
            label="上传controlnet图像", interactive=False, visible=False
        )

        def update_image_upload(enable_controlnet):
            if enable_controlnet:
                # 如果复选框被选中，则启用图像上传组件
                return gr.Image(interactive=True, visible=True)
            else:
                # 如果复选框未被选中，则禁用图像上传组件
                return gr.Image(interactive=False, visible=False)

        # 当复选框的值发生变化时，调用update_image_upload函数，并更新图像上传组件的状态
        enable_controlnet.change(
            fn=update_image_upload, inputs=enable_controlnet, outputs=ref_image
        )

        def create_mask_from_editor(editor_output):
            # editor_output 是一个字典，包含 'background'（原始图像）和 'layers'（编辑层）
            if "layers" in editor_output and len(editor_output["layers"]) > 0:
                # 假设只有一个编辑层
                layer = editor_output["layers"][0]
                # 将编辑层转换为灰度图像
                layer_gray = layer.convert("L")
                # 创建掩码：绘画部分为白色（255），透明部分为黑色（0）
                mask = Image.new("L", layer.size, 0)
                mask.paste(layer_gray, (0, 0), layer)
                # 反转掩码，使绘画区域为白色，未绘画区域为黑色
                mask = ImageChops.invert(mask)
            else:
                # 如果没有编辑层，返回全黑掩码（无修复）
                mask = Image.new("L", editor_output["background"].size, 0)
            return mask

        def generate_image(
            model_name,
            prompt,
            editor_output,
            enable_controlnet,
            controlnet_model_name,
            ref_image,
            negative_prompt,
            enable_strength,
            strength,
            enable_guidance,
            guidance_scale,
            enable_blur,
            blur_factor,
            enable_repro,
            seed,
        ):
            if prompt.strip() == "":
                return gr.Error("请输入有效的提示文本")
            try:
                init_img = editor_output["background"]
                mask_img = create_mask_from_editor(editor_output)
                if enable_controlnet:
                    controlnet = ControlNetModel.from_pretrained(
                        controlnet_model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                    ).to(DEVICE)

                    pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_name,
                        image=init_img,
                        mask_image=mask_img,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        control_image=ref_image,
                    ).to(DEVICE)
                else:
                    pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_name,
                        image=init_img,
                        mask_image=mask_img,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        negative_prompt=negative_prompt,
                    ).to(DEVICE)
                if enable_repro:
                    image = pipeline(
                        prompt,
                        image=init_img,
                        mask_image=mask_img,
                        guidance_scale=(
                            float(guidance_scale) if enable_guidance else None
                        ),
                        negative_prompt=negative_prompt,
                        generator=torch.Generator("cpu").manual_seed(seed),
                        strength=strength if enable_strength else None,
                    ).images[0]
                else:
                    image = pipeline(
                        prompt,
                        image=init_img,
                        mask_image=mask_img,
                        guidance_scale=(
                            float(guidance_scale) if enable_guidance else None
                        ),
                        negative_prompt=negative_prompt,
                        strength=strength if enable_strength else None,
                    ).images[0]
                return image

            except Exception as e:
                return gr.Error(f"生成图像时遇到了错误: {str(e)}")

        generate_button.click(
            generate_image,
            inputs=[
                model_name,
                prompt,
                editor_output,
                enable_controlnet,
                controlnet_model_name,
                ref_image,
                negative_prompt,
                enable_strength,
                strength,
                enable_guidance,
                guidance_scale,
                enable_blur,
                blur_factor,
                enable_repro,
                seed,
            ],
            outputs=output_image,
        )
        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: output_image.update(value=None), None, output_image)
    with gr.Tab("文本/图像生成视频"):
        models_examples = {
            0: "CogVideoXImageToVideoPipeline",
            1: "stabilityai/stable-video-diffusion-img2vid-xt",
            2: "ali-vilab/i2vgen-xl",
            3: "guoyww/animatediff-motion-adapter-v1-5-2",
            4: "damo-vilab/text-to-video-ms-1.7b",
        }
        choices = list(models_examples.values())
        model_index = gr.Dropdown(
            choices=choices,
            value="CogVideoXImageToVideoPipeline",
            label="模型名称",
            type="index",
        )
        image = gr.Image()
        prompt = gr.Textbox(placeholder="请输入图像描述", label="提示文本")

        negative_prompt = gr.Textbox(placeholder="请输入负面描述", label="负面提示文本")
        with gr.Row():
            with gr.Column():
                num_frames = gr.Textbox(
                    value="25",
                    placeholder="25",
                    label="帧数",
                    info="请输入想要生成的帧数",
                )
        with gr.Column():
            enable_guidance = gr.Checkbox(
                value=False, label="启用引导", info="启用引导效果"
            )
            guidance_scale = gr.Textbox(
                value=1.0, placeholder="文本引导强度", label="引导强度"
            )
        with gr.Column():
            enable_repro = gr.Checkbox(
                value=False, label="启用复现", info="固定随机种子"
            )
            seed = gr.Textbox(
                value=2025, placeholder="请输入随机种子", label="随机种子"
            )
        generate_button = gr.Button("生成视频")
        output_video = gr.Video()

        def update_image_upload(enable_controlnet):
            if enable_controlnet:
                # 如果复选框被选中，则启用图像上传组件
                return gr.Image(interactive=True, visible=True)
            else:
                # 如果复选框未被选中，则禁用图像上传组件
                return gr.Image(interactive=False, visible=False)

        def generate_video(
            model_index,
            prompt,
            image,
            negative_prompt,
            num_frames,
            enable_guidance,
            guidance_scale,
            enable_repro,
            seed,
        ):
            if prompt.strip() == "":
                return gr.Error("请输入有效的提示文本")
            try:
                if model_index == 0:
                    pipe = CogVideoXImageToVideoPipeline.from_pretrained(
                        "THUDM/CogVideoX-5b-I2V", torch_dtype=torch.bfloat16
                    )
                    frames = pipe(
                        prompt=prompt,
                        image=image,
                        num_videos_per_prompt=1,
                        num_inference_steps=50,
                        num_frames=num_frames,
                        guidance_scale=guidance_scale if enable_guidance else None,
                        generator=(
                            torch.Generator(DEVICE).manual_seed(seed)
                            if enable_repro
                            else None
                        ),
                    ).frames[0]
                elif model_index == 1:
                    pipeline = StableVideoDiffusionPipeline.from_pretrained(
                        "stabilityai/stable-video-diffusion-img2vid-xt",
                        torch_dtype=torch.float16,
                        variant="fp16",
                    )
                    frames = pipeline(
                        image,
                        decode_chunk_size=8,
                        frames=int(num_frames),
                        guidance_scale=guidance_scale if enable_guidance else None,
                        generator=(
                            torch.Generator(DEVICE).manual_seed(seed)
                            if enable_repro
                            else None
                        ),
                    ).frames[0]
                elif model_index == 2:
                    pipeline = I2VGenXLPipeline.from_pretrained(
                        "ali-vilab/i2vgen-xl", torch_dtype=torch.float16, variant="fp16"
                    )
                    frames = pipeline(
                        prompt=prompt,
                        image=image,
                        negative_prompt=negative_prompt,
                        guidance_scale=guidance_scale if enable_guidance else None,
                        generator=(
                            torch.Generator(DEVICE).manual_seed(seed)
                            if enable_repro
                            else None
                        ),
                    ).frames[0]
                elif model_index == 3:
                    adapter = MotionAdapter.from_pretrained(
                        "guoyww/animatediff-motion-adapter-v1-5-2",
                        torch_dtype=torch.float16,
                    )
                    pipeline = AnimateDiffPipeline.from_pretrained(
                        "emilianJR/epiCRealism",
                        motion_adapter=adapter,
                        torch_dtype=torch.float16,
                    )
                    scheduler = DDIMScheduler.from_pretrained(
                        "emilianJR/epiCRealism",
                        subfolder="scheduler",
                        clip_sample=False,
                        timestep_spacing="linspace",
                        beta_schedule="linear",
                        steps_offset=1,
                    )
                    frames = pipeline(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        frames=int(num_frames),
                        guidance_scale=guidance_scale if enable_guidance else None,
                        generator=(
                            torch.Generator(DEVICE).manual_seed(seed)
                            if enable_repro
                            else None
                        ),
                    )
                else:
                    pipeline = DiffusionPipeline.from_pretrained(
                        "damo-vilab/text-to-video-ms-1.7b",
                        torch_dtype=torch.float16,
                        variant="fp16",
                    )
                    frames = pipeline(
                        prompt,
                        negative_prompt=negative_prompt,
                        frames=int(num_frames),
                        guidance_scale=guidance_scale if enable_guidance else None,
                        generator=(
                            torch.Generator(DEVICE).manual_seed(seed)
                            if enable_repro
                            else None
                        ),
                    ).frames[0]
                return frames
            except Exception as e:
                return gr.Error(f"生成图像时遇到了错误: {str(e)}")

        generate_button.click(
            generate_video,
            inputs=[
                model_index,
                prompt,
                image,
                negative_prompt,
                num_frames,
                enable_guidance,
                guidance_scale,
                enable_repro,
                seed,
            ],
            outputs=output_video,
        )

        reset_button = gr.Button("重置视频")
        reset_button.click(lambda: output_video.update(value=None), None, output_video)
    with gr.Tab("使用LLM辅助生成图像"):
        gr.Markdown("# 使用LLM")
        llm_source = gr.Dropdown(
            choices=["ollama", "huggingface"],
            value="ollama",
            type="index",
            label="模型来源",
        )
        # # if use huggingface, use api token
        # huggingface_api_token = gr.Textbox(
        #     value="", placeholder="use hf api token to download model", visible=False
        # )
        llm_model = gr.Textbox(value="", placeholder="LLM模型名称", label="LLM模型名称")
        input_prompt = gr.Textbox(
            value="", placeholder="请输入提示文本", label="提示文本"
        )
        temp = gr.Slider(minimum=0, maximum=1, value=0.5)
        files = gr.Files(file_types=["image", "pdf"])
        llm_button = gr.Button("生成描述文字")
        llm_output_prompt = gr.Textbox(value="", label="生成prompt")
        img_gen_btn = gr.Button("生成图像")
        output_img = gr.Image(label="生成图像")

        def process_files(files):
            prompts = []
            for file in files:
                file_path = file.name
                if file_path.lower().endswith("pdf"):
                    tokenizer = AutoTokenizer.from_pretrained(
                        "HuggingFaceH4/zephyr-7b-beta"
                    )
                    model = AutoModelForCausalLM.from_pretrained(
                        "HuggingFaceH4/zephyr-7b-beta"
                    )
                    summarizer = pipeline(
                        "text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        device=DEVICE,
                    )
                    # 处理 PDF 文件
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=1000, chunk_overlap=200
                    )
                    chunks = text_splitter.split_documents(docs)
                    chunked_docs = [chunk.page_content for chunk in chunks]
                    db = FAISS.from_documents(
                        chunked_docs,
                        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5"),
                    )
                    retriever = db.as_retriever(
                        search_type="similarity", search_kwargs={"k": 4}
                    )
                    # 3. 检索相关内容并生成总结
                    query = "Summarize the document"  # 可以根据需要调整查询
                    relevant_chunks = retriever.get_relevant_documents(query)
                    relevant_text = " ".join(
                        [chunk.page_content for chunk in relevant_chunks]
                    )

                    # 使用 Zephyr-7b-beta 生成总结
                    summary = summarizer(
                        relevant_text, max_length=150, min_length=50, do_sample=True
                    )[0]["generated_text"].strip()
                    prompts.append(summary)
                elif file_path.lower().endswith((".png", ".jpg", ".jpeg")):
                    # 处理图片文件
                    # 加载图像描述模型
                    caption_model = pipeline(
                        "image-to-text", model="Salesforce/blip-image-captioning-base"
                    )
                    image = Image.open(file_path)
                    caption = caption_model(image)[0]["generated_text"]
                    prompts.append(caption)
                else:
                    continue  # 跳过不支持的文件类型
            # 合并所有提示
            if not prompts:
                return "No valid files processed."
            combined_prompt = ".".join(prompts)
            return combined_prompt

        llm_button.click(process_files, files, llm_output_prompt)

        def generate_image(
            llm_source, llm_model, prompt, llm_output_prompt, temp, files
        ):
            combined_prompt = prompt + llm_output_prompt
            messages = [
                (
                    "system",
                    "You are a helpful assistant and help me to convert the following query to the prompt for image generation.",
                ),
                ("human", combined_prompt),
            ]
            if llm_source == 0:
                # use ollama
                try:
                    llm = ChatOllama(
                        model=llm_model,
                        temperature=temp,
                        # other params...
                    )
                except Exception as e:
                    return gr.Error(f"发生错误: {str(e)}")

                response = llm.invoke(messages)
                return response
            else:
                llm = HuggingFacePipeline.from_model_id(
                    model_id=llm_model,
                    task="text-generation",
                )
                template = """Question: convert the following query to the prompt for image generation.
                Query: {query}
                """
                prompt = PromptTemplate.from_template(template)

                chain = prompt | llm
                response = chain.invoke({"query": combined_prompt})
                return response

        img_gen_btn.click(
            generate_image,
            inputs=[llm_source, llm_model, prompt, llm_output_prompt, temp, files],
            outputs=output_img,
        )
        reset_btn = gr.Button("重置图像")
        reset_btn.click(fn=lambda: "", outputs=output_img)


app.launch()
