import argparse


def add_logging_arguments(parser: argparse.ArgumentParser):
    parser.add_argument(
        "--console_log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="set the logging level,default to INFO",
    )
    parser.add_argument(
        "--console_log_file", type=str, default=None, help="log to a file"
    )
    parser.add_argument(
        "--console_log_simple", action="store_true", help="Use simple console logging"
    )
