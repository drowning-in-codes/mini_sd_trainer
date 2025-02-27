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
import torch
from dataclasses import dataclass

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
                """
                模糊量由 blur_factor 参数确定。增加 blur_factor 会增加应用于蒙版边缘的模糊量，使原始图像和修复区域之间的过渡更加柔和。
                低或零 blur_factor 会保留蒙版的锐利边缘。
                """
                blur_factor = gr.Textbox(
                    value="0.5",
                    placeholder="请输入模糊因子",
                    label="模糊因子",
                    info="控制图像的模糊程度，范围从 0 到 1",
                )
                mask_img = gr.ImageEditor(label="mask图片")

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
            mask_img,
            enable_controlnet,
            controlnet_model_name,
            ref_image,
            negative_prompt,
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

                    pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_name,
                        image=init_image,
                        mask_image=mask_img,
                        controlnet=controlnet,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        control_image=ref_image,
                    ).to(DEVICE)
                else:
                    pipeline = AutoPipelineForInpainting.from_pretrained(
                        model_name,
                        image=init_image,
                        mask_image=mask_img,
                        torch_dtype=torch.float16,
                        variant="fp16",
                        negative_prompt=negative_prompt,
                    ).to(DEVICE)
                if enable_repro:
                    image = pipeline(
                        prompt,
                        image=init_image,
                        mask_image=mask_img,
                        guidance_scale=(
                            float(guidance_scale) if enable_guidance else None
                        ),
                        negative_prompt=negative_prompt,
                        generator=torch.Generator("cpu").manual_seed(seed),
                    ).images[0]
                else:
                    image = pipeline(
                        prompt,
                        image=init_image,
                        mask_image=mask_img,
                        guidance_scale=(
                            float(guidance_scale) if enable_guidance else None
                        ),
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
                mask_img,
                enable_controlnet,
                controlnet_model_name,
                ref_image,
                negative_prompt,
                enable_guidance,
                guidance_scale,
                enable_repro,
                seed,
            ],
            outputs=output_image,
        )
        reset_button = gr.Button("重置图像")
        reset_button.click(lambda: output_image.update(value=None), None, output_image)


app.launch()
