# Model Loading

Loaders that deserialize and cache model artifacts for inference. Each loader
caches its model in memory after the first load to avoid repeated file I/O.

| File | Loads | Artifact |
|------|-------|----------|
| `loader.py` | Flight-price pipeline (XGBoost) | `MODEL_PATH` |
| `gender_loader.py` | Gender classifier | `GENDER_MODEL_PATH` |

## Flight-price loader (`loader.py`)

```python
from src.model.loader import get_model, load_model, ModelLoader

model = get_model()              # get loaded model or load it
load_model()                     # load (raises if file missing)
ModelLoader.unload_model()       # clear the cache
```

A compatibility shim patches `_RemainderColsList` so pickles serialized by an
older scikit-learn still load on newer versions.

## Gender loader (`gender_loader.py`)

```python
from src.model.gender_loader import get_gender_model, GenderModelLoader

model = get_gender_model()
GenderModelLoader.unload_model()
```

## Error handling

- `FileNotFoundError` — artifact missing at the configured path.
- `RuntimeError` — deserialization failed for another reason.

## Tests

```bash
pytest tests/test_model_loader.py
```
