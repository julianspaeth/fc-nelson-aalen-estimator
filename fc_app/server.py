from flask import Flask
import logging
from fc_app.api import api_bp
from fc_app.web import web_bp


class FCServer:
    """
    Encapsulates the flask app and user-defined logic.
    """
    flask_app: Flask

    def __init__(self):
        self.flask_app = Flask(__name__)
        # This line is necessary to show error messages
        self.flask_app.secret_key = "something you know"
        if __name__ != '__main__':
            gunicorn_logger = logging.getLogger('gunicorn.error')
            self.flask_app.logger.handlers = gunicorn_logger.handlers
            self.flask_app.logger.setLevel(gunicorn_logger.level)
        self.flask_app.register_blueprint(api_bp, url_prefix='/api/')
        self.flask_app.register_blueprint(web_bp, url_prefix='/web/')

    def start(self, port=5000):
        """
        Starts a local debug server.
        """
        self.flask_app.run(debug=True, host='0.0.0.0', port=port)

    def wsgi(self, environ, start_response):
        """
        Offers a wsgi callable for production environments.
        """
        return self.flask_app.wsgi_app(environ, start_response)
