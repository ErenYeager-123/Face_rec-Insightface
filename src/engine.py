import cv2
import numpy as np
import os
import pickle 
from insightface.app import FaceAnalysis
from datetime import datetime
from src import config  # <--- Sửa import

class FaceEngine:
    def __init__(self):
        print(f"🔄 Đang khởi tạo InsightFace ({config.MODEL_NAME})...")
        self.app = FaceAnalysis(name=config.MODEL_NAME, providers=['CPUExecutionProvider'])
        self.app.prepare(ctx_id=config.CTX_ID, det_size=(640, 640))
        
        self.known_names = []
        self.known_embeddings = []
        
        if not os.path.exists(config.IMAGES_DIR):
            os.makedirs(config.IMAGES_DIR)
        self.load_database() 

    def load_database(self):
        try:
            with open(config.DB_FILE, "rb") as f:
                data = pickle.load(f)
            self.known_embeddings = data["embeddings"]
            self.known_names = data["names"]
            print(f"🚀 Hoàn tất! Đã nạp {len(self.known_names)} khuôn mặt.")
        except FileNotFoundError:
            print("⚠️ Chưa có file dữ liệu. Hệ thống sẽ bắt đầu trống.")
            self.known_embeddings = np.empty((0, 512))
            self.known_names = []

    def save_database(self):
        data = {"embeddings": self.known_embeddings, "names": self.known_names}
        with open(config.DB_FILE, "wb") as f:
            pickle.dump(data, f)
        print("💾 Đã cập nhật database.")

    def recognize_face(self, face_embedding):
        if len(self.known_embeddings) == 0: return "Unknown", 0.0
        norm_embedding = face_embedding / np.linalg.norm(face_embedding)
        scores = np.dot(self.known_embeddings, norm_embedding)
        best_idx = np.argmax(scores)
        best_score = scores[best_idx]
        if best_score > config.THRESHOLD:
            return self.known_names[best_idx], best_score
        return "Unknown", best_score

    def register_user(self, frame, face_bbox, face_embedding, name):
        user_folder = os.path.join(config.IMAGES_DIR, name)
        if not os.path.exists(user_folder): os.makedirs(user_folder)
        
        x1, y1, x2, y2 = face_bbox.astype(int)
        h, w, _ = frame.shape
        pad = config.PADDING
        x1 = max(0, x1 - pad); y1 = max(0, y1 - pad)
        x2 = min(w, x2 + pad); y2 = min(h, y2 + pad)
        face_img = frame[y1:y2, x1:x2]

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = os.path.join(user_folder, f"{name}_{timestamp}.jpg")
        
        if face_img.size > 0:
            cv2.imwrite(save_path, face_img)
            print(f"📸 Đã lưu ảnh: {save_path}")
            
            norm_embedding = face_embedding / np.linalg.norm(face_embedding)
            if len(self.known_embeddings) == 0:
                self.known_embeddings = np.array([norm_embedding])
            else:
                self.known_embeddings = np.vstack([self.known_embeddings, norm_embedding])
            self.known_names.append(name)
            self.save_database()
            return True
        return False
    
    # Hàm này dùng để tái tạo data (gọi từ tools/train.py)
    def rebuild_database(self):
        print(f"\n🔄 BẮT ĐẦU HUẤN LUYỆN TỪ: {config.IMAGES_DIR}")
        import glob
        known_names = []
        known_embeddings = []
        files = glob.glob(os.path.join(config.IMAGES_DIR, "*", "*.*"))
        
        for img_path in files:
            if not img_path.lower().endswith(('.jpg', '.png', '.jpeg')): continue
            name = os.path.basename(os.path.dirname(img_path))
            img = cv2.imread(img_path)
            if img is None: continue
            
            faces = self.app.get(img)
            if faces:
                # Lấy mặt to nhất
                faces = sorted(faces, key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]), reverse=True)
                emb = faces[0].embedding
                emb = emb / np.linalg.norm(emb)
                known_embeddings.append(emb)
                known_names.append(name)
                print(f"✅ Đã học: {name}")

        self.known_names = known_names
        self.known_embeddings = np.array(known_embeddings)
        self.save_database()