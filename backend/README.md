# Neuroflow Backend

The backend service is a Flask application that provides machine learning capabilities to the Neuroflow IDE. It has been modularized for better maintainability and uses `uv` for dependency management.

## Directory Structure

```text
backend/
├── app.py              # Main entry point (Registers Blueprints & Config)
├── routes/             # API Endpoints (Flask Blueprints)
│   ├── regression.py
│   ├── classification.py
│   ├── ml_others.py    # PCA, SVD
│   ├── data.py         # Stats & Conversion
│   └── database.py     # DB connections
├── models/             # Pure Logic (ML algorithms from scratch)
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   └── decomposition.py
└── utils/              # Shared Helpers
    ├── preprocessing.py
    ├── metrics.py
    └── distances.py
```

## How to Add a New Model

To add a new model (e.g., `MyModel` for Classification):

1.  **Add Logic**: Implement the class in `backend/models/classification.py`. Ensure it has `fit` and `predict` methods.
2.  **Add Route**: Update `backend/routes/classification.py`.
    ```python
    from backend.models.classification import MyModel

    @classification_bp.route("/api/my-model", methods=["POST"])
    def api_my_model():
        # Handle request data, call MyModel, return JSON
        pass
    ```
3.  **Automatic Registration**: Since `classification_bp` is already registered in `app.py`, your new route will be available immediately at `/api/my-model`.

## Environment Variables

Copy `.env.example` to `.env` in the root directory.
- `PORT`: The port the Flask server will listen on (default: 3000).

## Running
```bash
# From project root
uv run backend/app.py
```
