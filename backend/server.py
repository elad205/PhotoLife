from flask import Flask
import flask
from werkzeug.utils import secure_filename
from werkzeug.exceptions import abort, BadRequestKeyError
import os


class PageHandler(object):

    ALLOWED_TYPES = {'jpg', 'png'}

    APP = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    APP.config['UPLOAD_FOLDER'] = \
        '/Users/elad/Desktop/final/FinalProject_ML/upload'

    def __init__(self):
        super(PageHandler, self).__init__()

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
            print(file)
            print(file.filename)
            # if user does not select file, browser also
            # submit an empty part without filename
            if file.filename == '':
                return flask.redirect(flask.request.url)
            if file and PageHandler.allowed_file(file.filename):
                filename = secure_filename(file.filename)
                file.save(os.path.join(PageHandler.APP.config['UPLOAD_FOLDER'],
                                       filename))
                return flask.redirect(flask.url_for('upload_image',
                                                    filename=filename))
        except BadRequestKeyError:
            return flask.redirect(flask.request.url)

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in PageHandler.ALLOWED_TYPES

    def run(self):
        self.APP.run(debug=True, port=5001)


def main():
    web_app = PageHandler()
    web_app.run()


if __name__ == '__main__':
    main()
