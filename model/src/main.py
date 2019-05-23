import visdom
import torch
from colorization.DataParser import DataParser
from colorization.networks import GeneratorDecoder, Discriminator
from colorization.trainer import CombinedTraining
from colorization.args import get_args
import sys
import os
import threading
import queue


GEN_STRCUT = [('batchnorm', 0, 256),
              ('relu', None),
              ('decoderBlock', 512, 256, 3, 1),
              ('decoderBlock', 512, 128, 3, 1),
              ('decoderBlock', 256, 64, 3, 1),
              ('selfAtt', 128),
              ('decoderBlock', 128, 64, 3, 1),
              ('deconv', 128, 32, 3, 1),
              ('shuffle', None),
              ('relu', None),
              ('conv', 32, 3, 1, 1),
              ('tanh', None)]

DISCRIMINATOR_STRUCT = [('convBlock', 3, 128, 4, 2, 0.2, 0.2),
                        ('convBlock', 128, 128, 3, 1, 0.2, 0.5),
                        ('convBlock', 128, 256, 4, 2, 0.2, 0.5),
                        ('leaky', 0.2),
                        ('selfAtt', 256),
                        ('dropout', 0.5),
                        ('convBlock', 256, 512, 4, 2, 0.2, 0.5),
                        ('conv', 512, 1024, 4, 2),
                        ('leaky', 0.2),
                        ('conv', 1024, 1, 4, 1)
                        ]


def read_stdin(queue_):
    while True:
        msg = os.read(sys.stdin.fileno(), 1024)
        msg = msg.decode()
        msg = msg.replace("?", "")
        if os.path.exists(msg):
            queue_.put(msg)
        else:
            print("invalid")


def main(args):
    # connect to the visdom server
    if args.visdom:
        vis = visdom.Visdom(args.host, port=args.port)
    else:
        vis = None

    gen, dis = create_new_network(vis, args)
    if args.checkpoint != "":
        try:
            gen.load_state_dict(torch.load(args.checkpoint))
        except FileNotFoundError and KeyError:
            print(f"invalid checkpoint: {args.checkpoint}")
            exit(-1)

    if args.mode == "standby":
        queue_ = queue.Queue()
        input_files = []

        lisnter = threading.Thread(target=read_stdin, args=(queue_, ))
        lisnter.daemon = True
        lisnter.start()

        while True:
            while not queue_.empty() and len(input_files) < 5:
                input_files.append(queue_.get(block=False))

            if input_files:
                gen.eval_model(input_files, args.save_loc)
                input_files = []

    if args.mode == "train":
        # initialise data set
        train_loader = DataParser.load_places_train(
            batch_size=args.batch_size, train_dir=args.train_dataset)
        # create, train and test the network
        trainer_init(train_loader, dis, gen, args)

        if args.save_weights:
            torch.save(gen.state_dict(), 'gen.ckpt')
            torch.save(dis.state_dict(), 'dis.ckpt')

    elif args.mode == "eval":
        if args.eval_images is None:
            print("cant eval model without images", file=sys.stderr)
            exit(-1)
        gen.eval_model(args.eval_images)

    else:
        test_loader = DataParser.load_places_test(
            batch_size=args.batch_size, test_dir=args.test_dataset)

        gen.test_model(test_loader, args.iter_per_epoch, args.batch_size,
                       display_data=True)


def create_new_network(vis, args):
    """
    this function initialise the network, trains it on the train data and
    the evaluation data, tests it on the test data and saves the weights.
    :param vis: the visdom server to display graphs
    :return: the model created
    """
    generator = GeneratorDecoder(vis, (args.gen_lr, (args.beta0, 0.999)),
                                 GEN_STRCUT, torch.optim.Adam)
    discriminator = Discriminator(vis, (args.dis_lr, (args.beta0, 0.999)),
                                  DISCRIMINATOR_STRUCT, torch.optim.Adam)

    return generator, discriminator


def trainer_init(train_loader, discriminator, generator, args):

    trainer = CombinedTraining(discriminator, generator, args)

    trainer.super_train(train_loader, '1',
                        trainer.gan_train, decay_lr=args.decay_lr)


if __name__ == '__main__':
    # set train flag to false to load pre trained model
    main(get_args())
