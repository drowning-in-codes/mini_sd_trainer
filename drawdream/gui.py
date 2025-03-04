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
import os, re
import ollama
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
            value="anton-l/ddpm-butterflies-128",
            placeholder="anton-l/ddpm-butterflies-128",
            label="模型名称",
        )
        num_inference_steps = gr.Textbox(value=10, placeholder="10", label="推理步数")
        generate_button = gr.Button("生成图像")

        output_image = gr.Image()

        def generate_image(inference_steps, model):
            if not inference_steps.isdigit() or int(inference_steps) <= 0:
                return gr.Error("请输入一个有效的正整数")
            try:
                pipeline = DiffusionPipeline.from_pretrained(model).to(DEVICE)
                print(model)
                image = pipeline(num_inference_steps=int(inference_steps)).images[0]
                return image
            except Exception as e:
                gr.Error(f"生成图像时遇到了错误: {str(e)}")
                print(f"生成图像时遇到了错误: {str(e)}")
                return None

        generate_button.click(
            generate_image,
            inputs=[num_inference_steps, model_name],
            outputs=output_image,
        )

        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: None, None, output_image)

    with gr.Tab("文本生成图像"):
        t2i_model_name = gr.Textbox(
            value="stable-diffusion-v1-5/stable-diffusion-v1-5",
            placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
            label="模型名称",
        )
        prompt = gr.Textbox(value="", placeholder="请输入图像描述", label="提示文本")
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

        def t2i_generate_image(
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
                gr.Error("请输入有效的提示文本")
                print("请输入有效的提示文本")
                return None
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
            t2i_generate_image,
            inputs=[
                t2i_model_name,
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
        reset_button.click(lambda: None, None, output_image)
    with gr.Tab("图像生成图像"):
        model_name = gr.Textbox(
            value="stable-diffusion-v1-5/stable-diffusion-v1-5",
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

        def i2i_generate_image(
            model_name,
            prompt,
            init_image,
            negative_prompt,
            height_width,
            strength,
            guidance_scale,
            enable_repro,
            seed,
        ):

            if prompt.strip() == "" or len(init_image) == 0:
                gr.Error("文本提示和图像输入错误")
                return
            else:
                try:
                    pipeline = AutoPipelineForImage2Image.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        safety_checker=None,
                    ).to(DEVICE)
                    # print(height_width)
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
                                guidance_scale=float(guidance_scale),
                                negative_prompt=negative_prompt,
                                generator=torch.Generator("cpu").manual_seed(seed),
                            ).images[0]
                        else:
                            print(prompt)
                            image = pipeline(
                                prompt,
                                image=init_image,
                                strength=strength,
                                guidance_scale=float(guidance_scale),
                                negative_prompt=negative_prompt,
                            ).images[0]
                    return image

                except Exception as e:
                    gr.Error(f"生成图像时遇到了错误: {str(e)}")
                    print(f"生成图像时遇到了错误: {str(e)}")
                    return None

        generate_button.click(
            i2i_generate_image,
            inputs=[
                model_name,
                prompt,
                init_image,
                negative_prompt,
                height_width,
                strength,
                guidance_scale,
                enable_repro,
                seed,
            ],
            outputs=output_image,
        )

        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: None, None, output_image)

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
        reset_button.click(lambda: None, None, output_image)

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
        reset_button.click(lambda: None, None, output_video)
    with gr.Tab("使用LLM辅助生成图像"):
        gr.Markdown("# 使用LLM")
        llm_source = gr.Dropdown(
            choices=["ollama", "huggingface"],
            value="huggingface",
            type="index",
            label="模型来源",
        )
        # # if use huggingface, use api token
        # huggingface_api_token = gr.Textbox(
        #     value="", placeholder="use hf api token to download model", visible=False
        # )
        llm_model = gr.Textbox(value="", placeholder="LLM模型名称", label="LLM模型名称")
        llm_model_examples = gr.Examples(
            examples=[["deepseek-ai/DeepSeek-R1"]],
            inputs=[llm_model],
            cache_examples=False,
        )

        def llm_source_change(source_index):
            if source_index == 0:
                examples = ollama.list()
                # print(examples)
                model_name_list = [[m.model] for m in examples.models]
                # print(model_name_list)
                return gr.Dataset(
                    samples=model_name_list,
                )
            else:
                return gr.Dataset(samples=[["deepseek-ai/DeepSeek-R1"]])

        llm_source.change(llm_source_change, llm_source, llm_model_examples.dataset)
        input_prompt = gr.Textbox(
            value="", placeholder="请输入提示文本", label="提示文本"
        )
        temp = gr.Slider(minimum=0, maximum=1, value=0.5)
        files = gr.Files(file_types=["image", "pdf"])
        llm_button = gr.Button("生成描述文字")
        llm_content = gr.Textbox(value="", label="图像内容")
        llm_output_prompt = gr.Textbox(value="", label="生成prompt")
        llm_output_negative_prompt = gr.Textbox(value="", label="生成负向prompt")
        img_gen_btn = gr.Button("生成图像")
        output_img = gr.Image(label="生成图像")

        def process_files(files):
            file_prompts = ""
            if files:
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
                            "image-to-text",
                            model="Salesforce/blip-image-captioning-base",
                        )
                        image = Image.open(file_path)
                        caption = caption_model(image)[0]["generated_text"]
                        prompts.append(caption)
                    else:
                        continue  # 跳过不支持的文件类型
                    # 合并所有提示
                    if not prompts:
                        # No valid files processed.
                        return ""
                    file_prompts = ".".join(prompts)
            return file_prompts

        def generate_prompt(
            llm_source,
            llm_model,
            prompt,
            files,
            temp,
        ):
            combined_prompt = prompt + process_files(files)
            # from https://geekdaxue.co/read/jianxu@aigc/ls78zitgbmw0k9z1#1tdk7c
            system_prompts = """
            从现在开始你将扮演一个stable diffusion的提示词工程师，你的任务是帮助我设计stable diffusion的文生图提示词。你需要按照如下流程完成工作。1、我将给你发送一段图片情景，你需要将这段图片情景更加丰富和具象生成一段图片描述。
            并且按照“<图片内容>具像化的图片描述</图片内容>”格式输出出来；2、你需要结合stable diffusion的提示词规则，将你输出的图片描述翻译为英语，并且加入诸如高清图片、高质量图片等描述词来生成标准的提示词，提示词为英语，以“<正向提示>提示词</正向提示>”格式输出出来；3、你需要根据上面的内容，设计反向提示词，你应该设计一些不应该在图片中出现的元素，例如低质量内容、多余的鼻子、多余的手等描述，这个描述用英文并且生成一个标准的stable diffusion提示词，以“<反向提示>提示词</反向提示>”格式输出出来。
            例如：我发送：一个二战时期的护士。你回复：
            <图片内容>一个穿着二战期间德国护士服的护士，手里拿着一个酒瓶，带着听诊器坐在附近的桌子上，衣服是白色的，背后有桌子</图片内容>;<正向提示>A nurse wearing a German nurse's uniform during World War II, holding a wine bottle and a stethoscope, sat on a nearby table with white clothes and a table behind,full shot body photo of the most beautiful artwork in the world featuring ww2 nurse holding a liquor bottle sitting on a desk nearby, smiling, freckles, white outfit, nostalgia, sexy, stethoscope, heart professional majestic oil painting by Ed Blinkey, Atey Ghailan, Studio Ghibli, by Jeremy Mann, Greg Manchess, Antonio Moro, trending on ArtStation, trending on CGSociety, Intricate, High Detail, Sharp focus, dramatic, photorealistic painting art by midjourney and greg rutkowski</正向提示>;<反向提示>cartoon, 3d, ((disfigured)), ((bad art)), ((deformed)),((extra limbs)),((close up)),((b&w)), wierd colors, blurry, (((duplicate))), ((morbid)), ((mutilated)), [out of frame], extra fingers, mutated hands, ((poorly drawn hands)), ((poorly drawn face)), (((mutation))), (((deformed))), ((ugly)), blurry, ((bad anatomy)), (((bad proportions))), ((extra limbs)), cloned face, (((disfigured))), out of frame, ugly, extra limbs, (bad anatomy), gross proportions, (malformed limbs), ((missing arms)), ((missing legs)), (((extra arms))), (((extra legs))), mutated hands, (fused fingers), (too many fingers), (((long neck))), Photoshop, video game, ugly, tiling, poorly drawn hands, poorly drawn feet, poorly drawn face, out of frame, mutation, mutated, extra limbs, extra legs, extra arms, disfigured, deformed, cross-eye, body out of frame, blurry, bad art, bad anatomy, 3d rende</反向提示>.
            """
            messages = [
                (
                    "system",
                    system_prompts,
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

                response = llm.invoke(messages).content.strip('"')
                print(response)
                content = re.search(r"<图片内容>(.*?)</图片内容>", response).group(1)
                prompt = re.search(r"<正向提示>(.*?)</正向提示>", response).group(1)
                negative_prompt = re.search(
                    r"<反向提示>(.*?)</反向提示>", response
                ).group(1)
                return content, prompt, negative_prompt
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
                # response = llm.invoke(messages).content.strip('"')
                content = re.search(r"<图片内容>(.*?)</图片内容>", response).group(1)
                prompt = re.search(r"<正向提示>(.*?)</正向提示>", response).group(1)
                negative_prompt = re.search(
                    r"<反向提示>(.*?)</反向提示>", response
                ).group(1)
                return content, prompt, negative_prompt

        llm_button.click(
            generate_prompt,
            inputs=[llm_source, llm_model, input_prompt, files, temp],
            outputs=[llm_content, llm_output_prompt, llm_output_negative_prompt],
        )
        img_model = gr.Textbox(
            value="stable-diffusion-v1-5/stable-diffusion-v1-5",
            placeholder="stable-diffusion-v1-5/stable-diffusion-v1-5",
            label="模型名称",
        )

        examples = gr.Examples(
            examples=[
                [
                    "stable-diffusion-v1-5/stable-diffusion-v1-5",
                ],
                [
                    "stabilityai/stable-diffusion-xl-base-1.0",
                ],
                [
                    "kandinsky-community/kandinsky-2-2-decoder",
                ],
            ],
            inputs=[img_model],
        )

        def generate_img(model, llm_output_prompt, llm_output_negative_prompt):
            pipeline = AutoPipelineForText2Image.from_pretrained(
                model,
            ).to(DEVICE)
            image = pipeline(
                llm_output_prompt, negative_prompt=llm_output_negative_prompt
            ).images[0]
            return image

        img_gen_btn.click(
            generate_img,
            inputs=[img_model, llm_output_prompt, llm_output_negative_prompt],
            outputs=output_img,
        )

        reset_btn = gr.Button("重置图像")
        reset_btn.click(fn=lambda: "", outputs=output_img)
    with gr.Blocks():
        gr.Markdown("# 训练LoRA")

        def train_lora(**kwargs):
            # 这里可以添加调用实际训练函数的代码，使用输入的参数进行训练。
            return f"训练已启动，使用模型: {kwargs['pretrained_model']}, 训练数据目录: {kwargs['train_data_dir']}"

        with gr.Tab("基础设置"):
            with gr.Row():
                pretrained_model = gr.Textbox(
                    label="预训练模型路径", placeholder="./sd-models/model.ckpt"
                )
                model_type = gr.Radio(
                    choices=["sd1.5", "sd2.0", "sdxl", "flux"], label="模型类型"
                )
                parameterization = gr.Number(
                    label="参数化(仅在model_type为sd2.0时有效)", value=0
                )

            with gr.Row():
                train_data_dir = gr.Textbox(
                    label="训练数据集路径", placeholder="./train/aki"
                )
                reg_data_dir = gr.Textbox(label="正则化数据集路径(可选)")
                resolution = gr.Textbox(label="分辨率(w,h)", placeholder="512,512")

        with gr.Tab("网络设置"):
            with gr.Row():
                network_module = gr.Dropdown(
                    choices=["networks.lora", "lycoris.kohya"], label="网络模块"
                )
                network_weights = gr.Textbox(label="LoRA网络权重路径(可选)")

            with gr.Row():
                network_dim = gr.Slider(
                    minimum=4, maximum=128, step=1, label="网络维度", value=32
                )
                network_alpha = gr.Slider(
                    minimum=1, maximum=128, step=1, label="网络alpha值", value=32
                )

        with gr.Tab("训练设置"):
            with gr.Row():
                batch_size = gr.Slider(
                    minimum=1, maximum=64, step=1, label="批量大小", value=1
                )
                max_train_epoches = gr.Slider(
                    minimum=1, maximum=100, step=1, label="最大训练epoch数", value=10
                )
                save_every_n_epochs = gr.Slider(
                    minimum=1, maximum=10, step=1, label="每N个epoch保存一次", value=2
                )

            with gr.Row():
                train_unet_only = gr.Checkbox(label="仅训练U-Net")
                train_text_encoder_only = gr.Checkbox(label="仅训练文本编码器")
                stop_text_encoder_training = gr.Number(
                    label="停止文本编码器训练于第N步", value=0
                )

        with gr.Tab("优化器与学习率"):
            with gr.Row():
                lr = gr.Textbox(label="学习率", placeholder="1e-4")
                unet_lr = gr.Textbox(label="U-Net学习率", placeholder="1e-4")
                text_encoder_lr = gr.Textbox(
                    label="文本编码器学习率", placeholder="1e-5"
                )

            with gr.Row():
                lr_scheduler = gr.Dropdown(
                    choices=[
                        "linear",
                        "cosine",
                        "cosine_with_restarts",
                        "polynomial",
                        "constant",
                        "constant_with_warmup",
                        "adafactor",
                    ],
                    label="学习率调度器",
                )
                lr_warmup_steps = gr.Number(label="学习率预热步数", value=0)
                lr_restart_cycles = gr.Number(label="余弦重启次数", value=1)

        with gr.Tab("高级设置"):
            with gr.Row():
                optimizer_type = gr.Dropdown(
                    choices=[
                        "AdamW8bit",
                        "AdamW",
                        "Lion",
                        "Lion8bit",
                        "SGDNesterov",
                        "SGDNesterov8bit",
                        "DAdaptation",
                        "AdaFactor",
                        "prodigy",
                    ],
                    label="优化器类型",
                )
                output_name = gr.Textbox(label="输出模型名称", placeholder="aki")
                save_model_as = gr.Dropdown(
                    choices=["safetensors", "ckpt", "pt"], label="模型保存格式"
                )

            with gr.Row():
                noise_offset = gr.Number(label="噪声偏移", value=0)
                keep_tokens = gr.Number(label="保留前N个tokens不变", value=0)
                min_snr_gamma = gr.Number(label="最小信噪比(SNR)伽马值", value=0)

        btn_start_training = gr.Button("开始训练")
        output = gr.Textbox(label="输出信息", interactive=False)

        btn_start_training.click(
            fn=train_lora,
            inputs=[
                pretrained_model,
                model_type,
                parameterization,
                train_data_dir,
                reg_data_dir,
                network_module,
                network_weights,
                network_dim,
                network_alpha,
                resolution,
                batch_size,
                max_train_epoches,
                save_every_n_epochs,
                train_unet_only,
                train_text_encoder_only,
                stop_text_encoder_training,
                noise_offset,
                keep_tokens,
                min_snr_gamma,
                lr,
                unet_lr,
                text_encoder_lr,
                lr_scheduler,
                lr_warmup_steps,
                lr_restart_cycles,
                optimizer_type,
                output_name,
                save_model_as,
            ],
            outputs=output,
        )

app.launch()
