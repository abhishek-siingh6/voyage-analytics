"""Utility script to run local development server."""

import os
import sys
import argparse

from api.app import app
from config.settings import settings


def create_dummy_model_if_needed():
    """Create dummy model if it doesn't exist."""
    import joblib
    import os

    dummy_model_path = "tests/fixtures/dummy_model.joblib"
    if not os.path.exists(dummy_model_path):
        print("Creating dummy model for development...")
        from tests.fixtures.dummy_model import create_dummy_model
        create_dummy_model(dummy_model_path)

        # Also copy to artifacts for convenience
        if not os.path.exists("artifacts/flight_price_pipeline.joblib"):
            os.makedirs("artifacts", exist_ok=True)
            with open(dummy_model_path, "rb") as src:
                with open("artifacts/flight_price_pipeline.joblib", "wb") as dst:
                    dst.write(src.read())
            print(f"Dummy model copied to: artifacts/flight_price_pipeline.joblib")


def main():
    """Main entry point for running the Flask app."""
    parser = argparse.ArgumentParser(description="Run Voyage Analytics API")
    parser.add_argument(
        "--host",
        default=settings.api_host,
        help=f"Host to bind to (default: {settings.api_host})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=settings.api_port,
        help=f"Port to bind to (default: {settings.api_port})",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (not recommended for production)",
    )
    parser.add_argument(
        "--create-dummy",
        action="store_true",
        help="Create dummy model if it doesn't exist",
    )

    args = parser.parse_args()

    if args.create_dummy:
        create_dummy_model_if_needed()

    print(f"Starting Voyage Analytics API...")
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  Model: {settings.model_path}")
    print(f"  Debug: {args.debug}")

    # Check if model exists (warning but not error for development)
    if not os.path.exists(settings.model_path):
        print(f"\nWarning: Model not found at {settings.model_path}")
        print("  For local testing, a dummy model will be created.")
        print("  Run with --create-dummy to generate a dummy model.")
        print("  Or place your trained model at the path above.")
        create_dummy_model_if_needed()

    app.run(
        host=args.host,
        port=args.port,
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
