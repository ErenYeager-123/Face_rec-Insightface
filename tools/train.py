import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.engine import FaceEngine

if __name__ == "__main__":
    engine = FaceEngine()
    engine.rebuild_database()