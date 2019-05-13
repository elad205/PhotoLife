import argparse
import os

DES = ""


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


POSSIBLE_MODES = ("eval", "test", "train")


def get_mode(arg):
    if arg.lower() in POSSIBLE_MODES:
        return arg.lower()
    else:
        raise argparse.ArgumentTypeError(
            'invalid mode, see available modes in help')


def check_paths(paths):
    print(paths)
    for path in paths:
        if not os.path.exists(path):
            raise argparse.ArgumentTypeError(
                "path doesnt exist, please check if your path exists")

    return paths


def get_args():

    parser = argparse.ArgumentParser(description=DES)
    parser.add_argument('mode', type=get_mode)
    parser.add_argument('batch_size', type=int)
    parser.add_argument("--test_dataset", type=str, default="")
    parser.add_argument("--train_dataset", type=str, default="")
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--gen_lr', '-glr', type=float, default=3e-4)
    parser.add_argument('--dis_lr', '-dlr', type=float, default=3e-5)
    parser.add_argument('--decay_lr', type=str2bool, default=True)
    parser.add_argument('--visdom', '-v', type=str2bool, default=False)
    parser.add_argument('--host', type=str, default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--gen_steps_per_epoch', '-s', type=int, default=2)
    parser.add_argument('--iter_per_epoch', '-i', type=int, default=5000)
    parser.add_argument('--checkpoint', '-c', type=str, default="")
    parser.add_argument('--beta0', type=int, default=0)
    parser.add_argument('--save_weights', type=str2bool, default=True)
    parser.add_argument("--eval_images", nargs="+", default=None)

    args = parser.parse_args()
    return args




