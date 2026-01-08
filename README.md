# Neuroflow IDE

A robust IDE for building and experimenting with machine learning models using a flow-based interface.

## Prerequisites

- **Node.js** (v18 or higher recommended)
- **Python** (v3.10 or higher)
- **uv** (Python package manager)

## Setup and Installation

### 1. Clone the repository
```bash
git clone <repository-url>
cd neuroflow-ide
```

### 2. Backend Setup (using uv)
```bash
# Install dependencies and create virtual environment
uv sync

# Create environment file
cp .env.example .env
```

### 3. Frontend Setup (using npm)
```bash
# Install dependencies
npm install

# Fix potential permissions issues with binaries
chmod +x node_modules/.bin/*
```

## Running the Application

### Start the Backend
```bash
# From project root
uv run backend/app.py
```

### Start the Frontend (Electron)
```bash
# From project root
npm run electron:dev
```

## Project Structure

- `backend/`: Flask-based ML service (modularized).
- `src/`: React-based frontend components.
- `electron/`: Electron configuration and main process.
- `models/`: Custom ML model implementations.
- `routes/`: API endpoint definitions using Flask Blueprints.
- `utils/`: Preprocessing and metric calculation helpers.
