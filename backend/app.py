import sys
import os
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Ensure project root is in sys.path for absolute imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import Blueprints
from backend.routes.regression import regression_bp
from backend.routes.classification import classification_bp
from backend.routes.ml_others import ml_others_bp
from backend.routes.data import data_bp
from backend.routes.database import database_bp
from backend.routes.knn import knn_bp
from backend.routes.clustering import clustering_bp

app = Flask(__name__)
CORS(app)

# Register Blueprints
app.register_blueprint(regression_bp)
app.register_blueprint(classification_bp)
app.register_blueprint(ml_others_bp)
app.register_blueprint(data_bp)
app.register_blueprint(database_bp)
app.register_blueprint(knn_bp)
app.register_blueprint(clustering_bp)

@app.route("/api/health")
def health():
    return jsonify({"status": "ok", "modularized": True})

if __name__ == "__main__":
    # Note: Run from project root with: uv run backend/app.py
    port = int(os.environ.get("PORT", 3000))
    app.run(debug=True, host="0.0.0.0", port=port)