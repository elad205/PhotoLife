import argparse
import os
import socket

DES = ""


def check_ip(ip):
    try:
        socket.inet_aton(ip)
    except socket.error:
        print("invalid ip")
        exit(-1)

    return ip


def check_port(port):
    try:
        port = int(port)
    except TypeError:
        print("invalid port")
        exit(-1)

    if port > 0 and port < 65536:
        return port
    else:
        print("invalid port")
        exit(-1)


def check_path(path):
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            f"{path} doesnt exist, please check if your path exists")

    return path


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--host", type=check_ip, default="127.0.0.1")
    parser.add_argument("--port", "-p", type=check_port, default="8001")
    parser.add_argument("--upload_loc", "-u", type=check_path,
                        default='../frontend/static/uploads')
    parser.add_argument("--save_loc", "-s", type=check_path,
                        default="../frontend/static/colored")
    parser.add_argument("--checkpoint", "-c", type=check_path,
                        default="../pre trained/gen.ckpt")

    parser.add_argument("--certificate", "-r", type=check_path,
                        default=(r"C:\Users\eladc\Documents\proj"
                                 r"\FinalProject_ML\backend\crt"
                                 r"\photolife-ml_me.crt"))

    parser.add_argument("--key", "-k", type=check_path,
                        default="c:\\private_key.key")

    args = parser.parse_args()
    return args
