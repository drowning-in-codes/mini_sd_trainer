import argparse
def add_custom_train_arguments(parser: argparse.ArgumentParser, support_weighted_captions: bool = True):
    parser.add_argument(
        "--min_snr_gamma",
        type=float,
        default=None,
        help="gamma for reducing the weight of high loss timesteps. Lower numbers have stronger effect. 5 is recommended by paper. / 低いタイムステップでの高いlossに対して重みを減らすためのgamma値、低いほど効果が強く、論文では5が推奨",
    )
    parser.add_argument(
        "--scale_v_pred_loss_like_noise_pred",
        action="store_true",
        help="scale v-prediction loss like noise prediction loss / v-prediction lossをnoise prediction lossと同じようにスケーリングする",
    )
    parser.add_argument(
        "--v_pred_like_loss",
        type=float,
        default=None,
        help="add v-prediction like loss multiplied by this value / v-prediction lossをこの値をかけたものをlossに加算する",
    )
    parser.add_argument(
        "--debiased_estimation_loss",
        action="store_true",
        help="debiased estimation loss / debiased estimation loss",
    )
    if support_weighted_captions:
        parser.add_argument(
            "--weighted_captions",
            action="store_true",
            default=False,
            help="Enable weighted captions in the standard style (token:1.3). No commas inside parens, or shuffle/dropout may break the decoder. / 「[token]」、「(token)」「(token:1.3)」のような重み付きキャプションを有効にする。カンマを括弧内に入れるとシャッフルやdropoutで重みづけがおかしくなるので注意",
        )
