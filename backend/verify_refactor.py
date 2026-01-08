
import sys
import os
import numpy as np

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    print("Testing imports...")
    from backend.app import app
    from backend.models.regression import LinearRegression
    from backend.models.classification import LogisticRegression
    from backend.utils.preprocessing import train_test_split
    
    print("Imports successful!")

    print("Testing Utility...")
    X = np.random.rand(10, 2)
    y = np.random.randint(0, 2, 10)
    X_train, X_test, y_train, y_test = train_test_split(X, y)
    print("train_test_split successful!")

    print("Testing LinearRegression...")
    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)
    pred_LR = model_LR.predict(X_test)
    print("LinearRegression fit/predict successful!")

    print("Testing LogisticRegression...")
    model_LogR = LogisticRegression()
    model_LogR.fit(X_train, y_train)
    pred_LogR = model_LogR.predict(X_test)
    print("LogisticRegression fit/predict successful!")

    print("ALL CHECKS PASSED.")

except ImportError as e:
    print(f"IMPORT ERROR: {e}")
    sys.exit(1)
except Exception as e:
    print(f"RUNTIME ERROR: {e}")
    sys.exit(1)
