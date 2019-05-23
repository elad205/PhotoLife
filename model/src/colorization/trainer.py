import tqdm
import numpy as np


class CombinedTraining(object):
    def __init__(self, discriminator, generator, args):
        self.discriminator = discriminator
        self.generator = generator
        self.batch_size = args.batch_size
        self.args = args

    def super_train(self, train_loader, serial,
                    training_func, decay_lr=False):
        """
        this function handles the whole training scheme, it gets a function
        pointer to run in training_func and runs it in the desired iterations
        and epochs.
        :param train_loader: the train dataset iterator
        :param serial: a serial string for the plots
        :param training_func: the function to be run every batch
        :param decay_lr: whether or not to decay the learning rate
        :return: None
        """

        # switch to train mode
        self.discriminator.train()
        self.generator.train()
        for epoch in range(self.args.epochs):

            pbar = tqdm.tqdm(total=self.args.iter_per_epoch)
            for i, (images, labels) in enumerate(train_loader):

                # if the desired iteration have been satisfied stop training
                if i * self.batch_size > self.args.iter_per_epoch:
                    break

                # convert the tensors to cuda tensors
                images = images.to(self.discriminator.device)
                labels = labels.to(self.discriminator.device)

                # run the training function
                training_func(images, labels, self.args.gen_steps_per_epoch)

                # update the progress bar
                pbar.update(self.batch_size)

                # decay the lr at the first 3 epochs
                if epoch < 3 and decay_lr:
                    self.discriminator.scheduler.step()
                    self.generator.scheduler.step()

            pbar.close()
            tqdm.tqdm.write(
                "epoch {0}\n avg loss of discriminator is {1}\n"
                " avg loss of generator {2}".format(
                    epoch, np.mean(
                        self.discriminator.train_logger["loss"]) if np.mean(
                        self.discriminator.train_logger[
                            "loss"]) is not None else 0,
                    np.mean(
                        self.generator.train_logger["loss"])))
            # every epoch calculate the average loss
            self.discriminator.train_logger["cost"].append(np.mean(
                self.discriminator.train_logger["loss"]))

            self.generator.train_logger["cost"].append(np.mean(
                self.generator.train_logger["loss"]))
            # zero out the losss
            self.discriminator.train_logger["loss"] = []
            self.generator.train_logger["loss"] = []

        self.discriminator.train_logger["epochs"] = list(range(
            self.args.epochs))
        self.generator.train_logger["epochs"] = \
            list(range(self.args.epochs))
        self.discriminator.test_logger["epochs"] = list(
            range(self.args.epochs))
        self.generator.test_logger["epochs"] = \
            list(range(self.args.epochs))
        # create a graph of the cost in respect to the epochs
        self.discriminator.cost_plot.draw_plot(
            self.discriminator.train_logger, "train" + serial)

        self.generator.cost_plot.draw_plot(
            self.generator.train_logger, "train" + serial)

        # zero the loggers
        self.discriminator.train_logger["cost"] = []
        self.discriminator.train_logger["cost"] = []
        self.discriminator.test_logger["cost"] = []
        self.discriminator.test_logger["cost"] = []

    def gan_train(self, images, labels, steps):
        self.discriminator.train_model(images, labels, self.generator)
        # train the generator

        for _ in range(steps):
            self.generator.train_model(images, labels, self.discriminator)

    def sgenerator_train(self, images, labels, _):
        self.generator.train_model(images, labels, None)

    def adjust_lr(self, new_lr_gen, new_lr_dis):
        for param_group in self.generator.optimizer.param_groups:
            param_group['lr'] = new_lr_gen

        for param_group in self.discriminator.optimizer.param_groups:
            param_group['lr'] = new_lr_dis
