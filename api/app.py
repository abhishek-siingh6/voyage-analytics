"""Flask API application for Voyage Analytics."""

from flask import Flask
from logging.config import dictConfig

from api.routes import register_routes
from api.gender_routes import register_gender_routes
from api.recommend_routes import register_recommend_routes


def create_app() -> Flask:
    """
    Create and configure the Flask application.

    Returns:
        Configured Flask application instance.
    """
    # Configure logging
    # Simple logging configuration – avoid Flask-specific WSGI handler which may not exist in this environment
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="[%(asctime)s] %(levelname)s in %(module)s: %(message)s")

    app = Flask(__name__)

    # Load configuration
    app.config["JSON_SORT_KEYS"] = False

    # Register routes
    register_routes(app)
    register_gender_routes(app)
    register_recommend_routes(app)

    return app


# Create app instance for WSGI
app = create_app()


if __name__ == "__main__":
    from config.settings import settings

    app.run(
        host=settings.api_host,
        port=settings.api_port,
        debug=False,
    )
