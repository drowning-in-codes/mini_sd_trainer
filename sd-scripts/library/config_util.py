import argparse
from pathlib import Path


def add_config_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--dataset_config",
        type=Path,
        default=None,
        help="config file for detail settings / 詳細な設定用の設定ファイル",
    )
