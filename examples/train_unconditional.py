import argparse
import inspect
import logging
import math
import os
import shutil
from datetime import timedelta
from pathlib import Path

import accelerate
import datasets
import torch
import torch.nn.functional as F
from accelerate import Accelerator, InitProcessGroupKwargs
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from torchvision import transforms
from tqdm.auto import tqdm

import diffusers
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.utils import (
    check_min_version,
    is_accelerate_version,
    is_tensorboard_available,
    is_wandb_available,
)
from diffusers.training_utils import EMAModel
from diffusers.utils.import_utils import is_xformers_available


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.33.0.dev0")

logger = get_logger(__name__, log_level="INFO")


def main(args):
    logging_dir = os.path.join(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(
        project_dir=args.output_dir, logging_dir=logging_dir
    )

    kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=7200))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.logger,
        project_config=accelerator_project_config,
        kwargs_handlers=[kwargs],
    )

    if args.logger == "tensorboard":
        if not is_tensorboard_available():
            raise ImportError(
                "Tensorboard is not installed. Please install it using `pip install tensorboard`."
            )
    elif args.logger == "wandb":
        if not is_wandb_available():
            raise ImportError("Make sure to install wandb if you want to use it")
        import wandb
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                if args.use_ema:
                    ema_model.save_pretrained(os.path.join(output_dir, "unet_ema"))
                for i, model in enumerate(models):
                    model.save_pretrained(os.path.join(output_dir, "unet"))
                    weights.pop()

        def load_model_hook(models, input_dir):
            if args.use_ema:
                load_model = EMAModel.from_pretrained(
                    os.path.join(input_dir, "unet_ema"), UNet2DModel
                )
                ema_model.load_state_dict(load_model.state_dict())
                ema_model.to(accelerator.device)
                del ema_model

            for i in range(len(models)):
                # pop models so that they are not loaded again
                model = models.pop()
                load_model = UNet2DModel.from_pretrained(input_dir, subfolder="unet")
                model.register_to_config(**load_model.config)
                model.load_state_dict(load_model.state_dict())
                del load_model

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        datasets.utils.logging.set_verbosity_error()

    # handle the repo creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)
            if args.push_to_hub:
                repo_id = create_repo(
                    repo_id=args.hub_model_id or Path(args.output_dir).name,
                    exist_ok=True,
                    token=args.hub_token,
                ).repo_id

        if args.model_config_name_or_path is None:
            model = UNet2DModel(
                sample_size=args.resolution,
                in_channels=3,
                out_channels=3,
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
            )
        else:
            config = UNet2DModel.load_config(args.model_config_name_or_path)
            model = UNet2DModel.from_config(config)

        # Create EMA for the model
        if args.use_ema:
            ema_model = EMAModel(
                model.parameters(),
                decay=args.ema_max_decay,
                use_ema_warmup=True,
                inv_gamma=args.ema_inv_gamma,
                power=args.ema_power,
                model_config=model.config,
            )
        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
            args.mixed_precision = accelerator.mixed_precision
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16
            args.mixed_precision = accelerator.mixed_precision

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warning(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems during training, please update xFormers to at least 0.0.17. See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                    model.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available.")

        accepts_prediction_type = "prediction_type" in set(
            inspect.signature(DDPMScheduler.__init__).parameters.keys()
        )
        if accepts_prediction_type:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
                prediction_type=args.prediction_type,
            )
        else:
            noise_scheduler = DDPMScheduler(
                num_train_timesteps=args.ddpm_num_steps,
                beta_schedule=args.ddpm_beta_schedule,
            )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.adam_beta1, args.adam_beta2),
            weight_decay=args.adam_weight_decay,
            eps=args.adam_epsilon,
        )

        if args.dataset_name is not None:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                split="train",
            )
        else:
            dataset = load_dataset(
                "imagefolder",
                data_dir=args.train_data_dir,
                cache_dir=args.cache_dir,
                split="train",
            )

        augmentations = transforms.Compose(
            [
                transforms.Resize(
                    args.resolution, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                (
                    transforms.CenterCrop(args.resolution)
                    if args.center_crop
                    else transforms.RandomCrop(args.resolution)
                ),
                (
                    transforms.RandomHorizontalFlip()
                    if args.random_flip
                    else transforms.Lambda(lambda x: x)
                ),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        def transform_images(examples):
            images = [
                augmentations(image.convert("RGB")) for image in examples["image"]
            ]
            return {"input": images}

        logger.info(f"Dataset size: {len(dataset)}")

        dataset.set_transform(transform_images)
        train_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.train_batch_size,
            shuffle=True,
            num_workers=args.dataloader_num_workers,
        )
        # Initialize the learning rate scheduler
        lr_scheduler = get_scheduler(
            args.lr_scheduler,
            optimizer=optimizer,
            num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
            num_training_steps=(len(train_dataloader) * args.num_epochs),
        )
        model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, lr_scheduler
        )
        if args.use_ema:
            ema_model.to(accelerator.device)
        # We need to initialize the trackers we use, and also store our configuration.
        # The trackers initializes automatically on the main process.
        if accelerator.is_main_process:
            run = os.path.split(__file__)[-1].split(".")[0]
            accelerator.init_trackers(run)
        
        total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
        max_train_steps = args.num_epochs * num_update_steps_per_epoch    
        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {len(dataset)}")
        logger.info(f"  Num Epochs = {args.num_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_train_steps}")
        global_step = 0
        first_epoch = 0