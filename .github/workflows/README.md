# GitHub Actions CI/CD Pipeline

This workflow runs automated tests and builds the Docker image for Voyage Analytics.

## Workflow Triggers

- Runs on every push to main and feature branches
- Runs on pull requests
- Can be manually triggered

## Steps

1. **Checkout code** - Gets the repository
2. **Setup Python** - Installs Python 3.11
3. **Install dependencies** - Installs project dependencies
4. **Run tests** - Executes pytest with coverage
5. **Build Docker image** - Builds the application container

## Configuration

The workflow uses environment variables from GitHub Secrets:

- `DOCKER_USERNAME` - Docker Hub username (for pushing images)
- `DOCKER_PASSWORD` - Docker Hub password/token

## Running Locally

```bash
# Run tests locally
pytest

# Run tests with coverage
pytest --cov=src --cov=api --cov-report=term-missing
```

## Customization

Modify `.github/workflows/ci.yml` to add:

- Security scanning
- Code quality checks (bandit, sonarqube)
- Automatic deployment to staging
- Slack notifications
