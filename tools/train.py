import sys
import os

# --- HACK PATH: Thêm thư mục gốc dự án vào path để import được src ---
# Lấy đường dẫn hiện tại (tools/) -> đi ra ngoài 1 cấp (FaceRecognitionProject/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.engine import FaceEngine

if __name__ == "__main__":
    # Tận dụng luôn hàm rebuild_database đã viết trong engine.py
    # Giúp code gọn gàng, không bị lặp lại logic
    engine = FaceEngine()
    engine.rebuild_database()