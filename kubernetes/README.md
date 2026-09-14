# Kubernetes configuration for Voyage Analytics

This directory contains Kubernetes manifests for deploying the Voyage Analytics API.

## Prerequisites

- Kubernetes cluster (local or cloud)
- `kubectl` configured to access the cluster
- Docker registry access (for pushing images)

## Resources

### Namespace

Creates the `voyage-analytics` namespace.

### ConfigMap

Stores environment configuration (model path, MLflow URI, etc.).

### Deployment

Runs the Flask API with:
- 2 replicas (configurable)
- Liveness and readiness probes
- Environment variables from ConfigMap

### Service

Exposes the API internally via ClusterIP on port 5000.

## Usage

### Apply all resources

```bash
kubectl apply -f kubernetes/
```

### Apply individually

```bash
kubectl apply -f kubernetes/namespace.yaml
kubectl apply -f kubernetes/configmap.yaml
kubectl apply -f kubernetes/deployment.yaml
kubectl apply -f kubernetes/service.yaml
```

### Verify deployment

```bash
kubectl get pods -n voyage-analytics
kubectl get services -n voyage-analytics
kubectl get deployments -n voyage-analytics
```

### View logs

```bash
kubectl logs -n voyage-analytics -l app=voyage-analytics-api -f
```

### Scale deployment

```bash
# Scale to 3 replicas
kubectl scale deployment voyage-analytics-api -n voyage-analytics --replicas=3
```

### Delete resources

```bash
kubectl delete -f kubernetes/
```

## Configuration

Edit `kubernetes/configmap.yaml` to customize:

- `MODEL_PATH`: Path to the model artifact (should be mounted as volume)
- `MLFLOW_TRACKING_URI`: MLflow server URI
- `API_HOST` and `API_PORT`: API binding settings

## Persistent Storage

The model artifact should be mounted as a volume. Example:

```yaml
volumeMounts:
- name: model-volume
  mountPath: /app/artifacts
  readOnly: true

volumes:
- name: model-volume
  configMap:
    name: voyage-analytics-model
```

## Ingress (Optional)

To expose externally, configure an Ingress controller:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: voyage-analytics-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: voyage-analytics.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: voyage-analytics-api
            port:
              number: 5000
```
