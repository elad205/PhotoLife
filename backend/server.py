from flask import Flask
import flask
from werkzeug.utils import secure_filename
from werkzeug.exceptions import BadRequestKeyError
import os
import subprocess
import imghdr
import random
import time
import cv2

SAVE_LOC = "../frontend/static/colored"
MAX_SIZE = 50
CHECK_POINT = "../pre trained/gen.ckpt"


class Singleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).\
                __call__(*args, **kwargs)
        return cls._instances[cls]


class Generator(object, metaclass=Singleton):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator = subprocess.Popen(["python",
                                           f"..{os.path.sep}model{os.path.sep}"
                                           f"src{os.path.sep}main.py",
                                           "standby", str(1), "--save_loc",
                                           SAVE_LOC, "--checkpoint",
                                           CHECK_POINT], stdin=subprocess.PIPE,
                                          stderr=subprocess.PIPE)


gen = Generator()


class PageHandler(object):

    ALLOWED_TYPES = {'jpg', 'png'}

    APP = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    APP.config['UPLOAD_FOLDER'] = '../frontend/static/uploads'

    def __init__(self):
        super(PageHandler, self).__init__()

    @APP.route("/results")
    def render_result(self, file_name):
        pass

    @staticmethod
    @APP.route("/")
    def home_page():
        file = flask.request.args.get("filename", type=str)
        if file:
            pass
        return flask.render_template('index.html')

    @staticmethod
    @APP.route('/', methods=['GET', 'POST'])
    def upload_image():
        try:
            file = flask.request.files['image']
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return flask.redirect("error.html")
            if file and PageHandler.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = filename.split(".")
                filename[0] += str(random.randint(0, 100000))
                filename = ".".join(filename)
                file_path = os.path.join(PageHandler.APP.
                                         config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                im = cv2.imread(file_path, cv2.IMREAD_COLOR)
                im = cv2.resize(im, (256, 256))
                cv2.imwrite(file_path, im)
                if imghdr.what(file_path) is not None:
                    os.write(gen.generator.stdin.fileno(),
                             (file_path + "?" * (50-len(file_path))).encode())
                    res = os.read(gen.generator.stderr.fileno(), 4096).\
                        decode()
                    timeout = time.time() + 10
                    if res == "0":
                        return flask.redirect("/")
                    while res not in file_path:
                        res = os.read(gen.generator.stderr.fileno(),
                                      4096).decode()
                        if time.time() > timeout:
                            return flask.redirect("error.html")
                    else:
                        return flask.render_template(
                            "result.html",
                            filename="colored/" + filename,
                            filename2="uploads/" + filename)

            return flask.render_template("error.html")
        except BadRequestKeyError:
            return flask.redirect(flask.request.url)

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in PageHandler.ALLOWED_TYPES

    def run(self):
        self.APP.run(debug=True, port=5001, threaded=True)


def main():
    web_app = PageHandler()
    web_app.run()


if __name__ == '__main__':
    main()
