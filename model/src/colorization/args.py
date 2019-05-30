import argparse
import os
import socket

DES = "this is a generative adversarial networks model.\n the " \
      "program provides some utilities such as training, testing, and " \
      "streaming input to the model\n. this program shows the learning " \
      "process that I have taken through this project and the " \
      "inspiration that I have taken from various articles and projects"


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


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


POSSIBLE_MODES = ("eval", "test", "train", "standby")


def get_mode(arg):
    if arg.lower() in POSSIBLE_MODES:
        return arg.lower()
    else:
        raise argparse.ArgumentTypeError(
            'invalid mode, see available modes in help')


def check_paths(paths):
    for path in paths:
        check_path(path)
    return paths


def check_path(path):
    if path == "":
        return ""

    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(
            "path doesnt exist, please check if your path exists")

    return path


def get_args():
    parser = argparse.ArgumentParser(description=DES)
    parser.add_argument('mode', type=get_mode, help="the mode of the network:"
                                                    "train, test, "
                                                    "eval - for single or few "
                                                    "images, standby- for "
                                                    "running as a subprocess")
    parser.add_argument('batch_size', type=int, help="the batch size for the "
                                                     "generator and "
                                                     "discrimnator")
    parser.add_argument("--test_dataset", type=check_path, default="",
                        help="a relative or absolute path to "
                             "the training folder")
    parser.add_argument("--train_dataset", type=check_path, default="",
                        help="a relative or absolute path to "
                             "the training folder")
    parser.add_argument('--epochs', type=int, default=10, help="the number "
                                                               "of epochs")
    parser.add_argument('--gen_lr', '-glr', type=float, default=3e-4,
                        help="the learning rate of the generator")
    parser.add_argument('--dis_lr', '-dlr', type=float, default=3e-5,
                        help="the learning rate of the discriminator")
    parser.add_argument('--decay_lr', type=str2bool, default=True,
                        help="whether or not to decay the learning rate")
    parser.add_argument('--visdom', '-v', type=str2bool, default=False,
                        help="whether or not to activate a visdom server")
    parser.add_argument('--host', type=check_ip, default='127.0.0.1',
                        help="visdom server ip")
    parser.add_argument('--port', type=check_port, default=8097,
                        help="visdom server port")
    parser.add_argument('--gen_steps_per_epoch', '-s', type=int, default=2,
                        help="the amount of steps to be done on the "
                             "generator for every discriminator step")
    parser.add_argument('--iter_per_epoch', '-i', type=int, default=5000,
                        help="the amount of iteration of each epoch")
    parser.add_argument('--checkpoint', '-c', type=check_path, default="",
                        help="a pre loaded checkpoint for the generator")
    parser.add_argument('--beta0', type=int, default=0,
                        help="beat 0 decay for ADAM optimizer")
    parser.add_argument('--save_weights', type=str2bool, default=True,
                        help="whether or not to save the weights "
                             "after training")
    parser.add_argument("--eval_images", nargs="+", default=None,
                        type=str, help="image pathes to evaluate")
    parser.add_argument("--save_loc", type=check_path, default="",
                        help="a path for saving images")

    args = parser.parse_args()
    return args
