from flask import Flask
import flask
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequestKeyError
import os
import subprocess
import imghdr
import time
import cv2
import uuid
from args import get_args


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).\
                __call__(*args, **kwargs)
        return cls._instances[cls]


class Generator(object, metaclass=Singleton):
    def __init__(self, args):
        super(Generator, self).__init__()
        self.generator = subprocess.Popen(["python",
                                           f"..{os.path.sep}model{os.path.sep}"
                                           f"src{os.path.sep}main.py",
                                           "standby", str(1), "--save_loc",
                                           args.save_loc, "--checkpoint",
                                           args.checkpoint],
                                          stdin=subprocess.PIPE,
                                          stderr=subprocess.PIPE)

# create one instance of the generator


gen = None


class PageHandler(object):

    # the only allowed types to be handled on host
    ALLOWED_TYPES = {'jpg', 'png', 'jpeg'}

    APP = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    def __init__(self, args):
        super(PageHandler, self).__init__()
        self.host = args.host
        self.port = args.port

        # set the upload folder
        PageHandler.APP.config['UPLOAD_FOLDER'] = args.upload_loc

    @staticmethod
    @APP.route("/")
    def home_page():
        return flask.render_template('index.html')

    @staticmethod
    def pad(msg, pad_sign="?", buffer_length=1024) -> bytes:
        """
        this function pads all filenames to a fixed size in order to use pipes
        :param msg: the massage to be padded
        :param pad_sign: the sign of the padding
        :param buffer_length: the length of the final buffer
        :return: the padded filename
        """
        return (msg + pad_sign * (buffer_length - len(msg))).encode()

    @staticmethod
    def unique_name(file_name) -> str:
        """
        this function creates a unique file name to each file uploaded
        with the uuid1 function which creates an identifier using the server
        timestamp, id and a number sequence
        :param file_name: the file name to be changed
        :return: a unique file name
        """
        filename = secure_filename(file_name)
        filename = filename.split(".")
        # change file name
        filename[0] = str(uuid.uuid1())
        filename = ".".join(filename)
        return filename

    @staticmethod
    @APP.route('/upload', methods=['GET', 'POST'])
    def upload_image():
        """
        this function is the main functionality of the serve, it handles the
        image upload process, the communication with the generator and sending
        the response back to the client.
        if there is an error anywhere during the process then the client will
        be redirected to an error page
        :return: None
        """
        global gen
        try:
            file = flask.request.files['image']

            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return flask.redirect("error.html")
            # check if the file is legal
            if file and PageHandler.allowed_file(file.filename):

                # get a unique name for the file
                filename = PageHandler.unique_name(file.filename)

                file_path = os.path.join(PageHandler.APP.
                                         config['UPLOAD_FOLDER'], filename)

                # save the file in a fixed size
                file.save(file_path)
                im = cv2.imread(file_path, cv2.IMREAD_COLOR)
                im = cv2.resize(im, (256, 256))
                cv2.imwrite(file_path, im)

                # check if the image is not corrupt
                if imghdr.what(file_path) is not None:

                    # write to the generator pipe the filename
                    os.write(gen.generator.stdin.fileno(),
                             PageHandler.pad(file_path))
                    # read back the result
                    res = os.read(gen.generator.stderr.fileno(), 4096).\
                        decode()
                    timeout = time.time() + 10
                    # if an error has accrued
                    if res == "0":
                        return flask.redirect("/")

                    # wait for an answer
                    while res not in file_path:
                        res = os.read(gen.generator.stderr.fileno(),
                                      4096).decode()
                        if time.time() > timeout:
                            return flask.redirect("error.html")

                    # if successful return the result
                    else:
                        return flask.render_template(
                            "result.html",
                            filename="colored/" + filename,
                            filename2="uploads/" + filename)

            return flask.render_template("error.html")
        except BadRequestKeyError:
            return flask.redirect(flask.request.url)

    @staticmethod
    def allowed_file(filename) -> bool:
        """
        this function checks if the file type is allowed
        :param filename: the file name to be checked
        :return: True whether the file type is legal else False
        """
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in PageHandler.ALLOWED_TYPES

    def run(self):
        # runs the website
        self.APP.run(host=self.host, port=self.port, threaded=True)


def main(args):
    global gen
    gen = Generator(args)
    web_app = PageHandler(args)
    web_app.run()


if __name__ == '__main__':
    main(get_args())
