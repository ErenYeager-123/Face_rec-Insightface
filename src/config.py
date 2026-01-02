import os

# --- ĐƯỜNG DẪN --- 
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SRC_DIR) # Thư mục FaceRecognitionProject

DATA_DIR = os.path.join(BASE_DIR, "data")
IMAGES_DIR = os.path.join(DATA_DIR, "images")
DB_FILE = os.path.join(DATA_DIR, "embeddings.pkl")

# --- CAMERA ---
CAM_ID = 0  
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720

# --- GIAO DIỆN (UI) ---
UI_WIDTH = 640
UI_HEIGHT = 480 

# --- AI & MODEL ---
THRESHOLD = 0.5
MODEL_NAME = 'buffalo_l'
CTX_ID = 0 
PADDING = 50