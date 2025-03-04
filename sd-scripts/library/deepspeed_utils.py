def add_deepspeed_arguments(parser: argparse.ArgumentParser):
    # DeepSpeed Arguments. https://huggingface.co/docs/accelerate/usage_guides/deepspeed
    parser.add_argument("--deepspeed", action="store_true", help="enable deepspeed training")
    parser.add_argument("--zero_stage", type=int, default=2, choices=[0, 1, 2, 3], help="Possible options are 0,1,2,3.")
    parser.add_argument(
        "--offload_optimizer_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stages 2 and 3.",
    )
    parser.add_argument(
        "--offload_optimizer_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_device",
        type=str,
        default=None,
        choices=[None, "cpu", "nvme"],
        help="Possible options are none|cpu|nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--offload_param_nvme_path",
        type=str,
        default=None,
        help="Possible options are /nvme|/local_nvme. Only applicable with ZeRO Stage 3.",
    )
    parser.add_argument(
        "--zero3_init_flag",
        action="store_true",
        help="Flag to indicate whether to enable `deepspeed.zero.Init` for constructing massive models."
        "Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--zero3_save_16bit_model",
        action="store_true",
        help="Flag to indicate whether to save 16-bit model. Only applicable with ZeRO Stage-3.",
    )
    parser.add_argument(
        "--fp16_master_weights_and_gradients",
        action="store_true",
        help="fp16_master_and_gradients requires optimizer to support keeping fp16 master and gradients while keeping the optimizer states in fp32.",
    )