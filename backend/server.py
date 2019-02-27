from flask import Flask
import flask


def main():
    app = Flask(__name__, template_folder='../frontend/templates',
                static_folder='../frontend/static')

    @app.route("/")
    def home_page():
        return flask.render_template('index.html')

    app.run(debug=True)


if __name__ == '__main__':
    main()
