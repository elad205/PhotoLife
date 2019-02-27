from flask import Flask
import flask
from werkzeug.utils import secure_filename
import os


class PageHandler(object):

    ALLOWED_TYPES = {'jpg', 'png'}

    APP = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    APP.config['UPLOAD_FOLDER'] = ''

    def __init__(self):
        super(PageHandler, self).__init__()

    @APP.route("/")
    def home_page(self):
        return flask.render_template('index.html')

    @APP.route('/', methods=['GET', 'POST'])
    def upload_image(self):
        if 'file' not in flask.request.files:
            return flask.redirect(flask.request.url)

        file = flask.request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return flask.redirect(flask.request.url)
        if file and self.allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(self.APP.config['UPLOAD_FOLDER'], filename))
            return flask.redirect(flask.url_for('uploaded_file',
                                    filename=filename))

    @staticmethod
    def allowed_file(filename):
        return '.' in filename and \
               filename.rsplit('.', 1)[1].lower() in PageHandler.ALLOWED_TYPES

    def run(self):
        self.APP.run(debug=True)


def main():
    web_app = PageHandler()
    web_app.run()


if __name__ == '__main__':
    main()
