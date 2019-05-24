import argparse
import os


DES = ""


def check_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f"{path} doesnt exist, please check if your path exists")

    return path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", "-p", type=int, default="8001")
    parser.add_argument("--upload_loc", "-u", type=check_path,
                        default='../frontend/static/uploads')
    parser.add_argument("--save_loc", "-s", type=check_path,
                        default="../frontend/static/colored")
    parser.add_argument("--checkpoint", "-c", type=check_path,
                        default="../pre trained/gen.ckpt")

    args = parser.parse_args()
    return args
