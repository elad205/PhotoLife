from flask import Flask
import flask
from werkzeug.utils import secure_filename
from werkzeug.exceptions import abort, BadRequestKeyError
import os
import subprocess
import imghdr
import threading


SAVE_LOC = "/home/elad/PhotoLife/FinalProject_ML/backend/colored"


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
        print(os.path.exists(SAVE_LOC))
        self.generator = subprocess.Popen(["python3", "../model/src/main.py",
                                           "standby", str(1), "--save_loc" ,SAVE_LOC],
                                          stdin=subprocess.PIPE)

        self.checker = threading.Thread(target=self.check_for_crashes)
        self.checker.daemon = True
        self.checker.start()

    def check_for_crashes(self):
        while True:
            if self.generator.poll() is not None:
                exit(1)


gen = Generator()


class PageHandler(object):

    ALLOWED_TYPES = {'jpg', 'png'}

    APP = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    APP.config['UPLOAD_FOLDER'] = 'uploads'

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
                return flask.redirect(flask.request.url)
            if file and PageHandler.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file_path = os.path.join(PageHandler.APP.config['UPLOAD_FOLDER'],
                                       filename)
                file.save(file_path)
                if imghdr.what(file_path) is not None:
                    res = gen.generator.communicate(file_path.encode())[0]
                    if res.returncode == -1:
                        return flask.redirect("/")
                    else:
                        return flask.send_file(SAVE_LOC + os.sep + filename)

                return flask.redirect(flask.url_for('upload_image',
                                                    filename=filename))
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
