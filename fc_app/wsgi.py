from fc_app.server import FCServer


def application(environ, start_response):
    return FCServer().wsgi(environ, start_response)
