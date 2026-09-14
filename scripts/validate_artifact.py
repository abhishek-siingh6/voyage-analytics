"""Utility script to validate model artifact."""

import os
import sys
import joblib
import json
import argparse


def validate_artifact(artifact_path: str) -> bool:
    """
    Validate that the model artifact exists and can be loaded.

    Args:
        artifact_path: Path to the model artifact

    Returns:
        True if artifact is valid, False otherwise
    """
    print(f"Validating model artifact: {artifact_path}")

    # Check file exists
    if not os.path.exists(artifact_path):
        print(f"  ❌ File not found: {artifact_path}")
        return False
    print(f"  ✓ File exists")

    # Check file size
    file_size = os.path.getsize(artifact_path)
    if file_size < 1000:  # Less than 1KB
        print(f"  ⚠ Warning: File size is very small ({file_size} bytes)")
    else:
        print(f"  ✓ File size: {file_size / 1024:.2f} KB")

    # Try to load the model
    try:
        print(f"  Loading model...")
        model = joblib.load(artifact_path)
        print(f"  ✓ Model loaded successfully")

        # Check for predict method
        if hasattr(model, "predict"):
            print(f"  ✓ Model has 'predict' method")
        else:
            print(f"  ⚠ Warning: Model doesn't have 'predict' method")

        # Check for expected attributes
        for attr in ["model_name", "model_version"]:
            if hasattr(model, attr):
                print(f"  ✓ Model has '{attr}': {getattr(model, attr)}")
            else:
                print(f"  ⚠ Warning: Model missing '{attr}' attribute")

        return True

    except Exception as e:
        print(f"  ❌ Failed to load model: {e}")
        return False


def validate_metadata(metadata_path: str) -> bool:
    """Validate model metadata file if it exists."""
    if not os.path.exists(metadata_path):
        print(f"  ⚠ Metadata file not found: {metadata_path}")
        return False

    print(f"Validating metadata: {metadata_path}")

    try:
        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        print(f"  ✓ Metadata file is valid JSON")

        # Check required fields
        if "model_name" in metadata:
            print(f"  ✓ model_name: {metadata['model_name']}")
        if "model_version" in metadata:
            print(f"  ✓ model_version: {metadata['model_version']}")

        return True

    except json.JSONDecodeError as e:
        print(f"  ❌ Invalid JSON: {e}")
        return False
    except Exception as e:
        print(f"  ❌ Failed to read metadata: {e}")
        return False


def main():
    """Main entry point for artifact validation."""
    parser = argparse.ArgumentParser(
        description="Validate model artifacts for Voyage Analytics"
    )
    parser.add_argument(
        "--artifact",
        default="artifacts/flight_price_pipeline.joblib",
        help="Path to model artifact",
    )
    parser.add_argument(
        "--metadata",
        default="artifacts/model_metadata.json",
        help="Path to model metadata",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Voyage Analytics Model Artifact Validator")
    print("=" * 60)
    print()

    artifact_valid = validate_artifact(args.artifact)
    print()

    metadata_valid = validate_metadata(args.metadata)
    print()

    print("=" * 60)
    print("Summary:")
    print(f"  Artifact: {'✓ Valid' if artifact_valid else '❌ Invalid'}")
    print(f"  Metadata: {'✓ Valid' if metadata_valid else '❌ Invalid'}")

    if artifact_valid:
        print()
        print("✓ Artifacts are ready for use with the API")
        return 0
    else:
        print()
        print("❌ Artifact validation failed")
        print("  See artifacts/README.md for instructions on obtaining artifacts")
        return 1


if __name__ == "__main__":
    sys.exit(main())
