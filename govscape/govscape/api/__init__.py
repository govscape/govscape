from flask_restx import Api

from .endpoints.health import ns as health_ns
from .endpoints.pages import ns as pages_ns
from .endpoints.search import ns as search_ns


def init_api(app):
    """Initialize the Flask-RESTX API"""
    api = Api(
        version="1.0",
        title="GovScape API",
        description="A RESTful API for searching government PDF documents",
        doc="/docs",
        prefix="/api",
    )
    api.init_app(app)

    # Add namespaces
    api.add_namespace(health_ns)
    api.add_namespace(search_ns)
    api.add_namespace(pages_ns)

    return api
