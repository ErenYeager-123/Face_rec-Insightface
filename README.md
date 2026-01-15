# 🛡️ HỆ THỐNG NHẬN DIỆN KHUÔN MẶT - HƯỚNG DẪN CÀI ĐẶT

# --- BƯỚC 1: CÀI ĐẶT THƯ VIỆN ---
# Mở Terminal và chạy lệnh sau:
pip install -r requirements.txt

# --- BƯỚC 2: CẤU HÌNH HỆ THỐNG (src/config.py) ---
# Mở file src/config.py và kiểm tra các thông số sau:
# [cite_start]CAM_ID: 0 (webcam laptop), 1 hoặc 2 (iVCam/DroidCam).
# [cite_start]UI_WIDTH & UI_HEIGHT: Nên để 640x360 (tỷ lệ 16:9) để tránh bị bóp méo hình[cite: 3].
# [cite_start]FRAME_WIDTH & FRAME_HEIGHT: Độ phân giải gốc camera (thường là 1280x720)[cite: 4].

# --- BƯỚC 3: QUY TRÌNH VẬN HÀNH LẦN ĐẦU ---

# 1. Chuẩn bị dữ liệu ảnh:
# [cite_start]- Tạo thư mục theo tên trong data/images/ (VD: data/images/NguyenVanA/)[cite: 5].
# [cite_start]- Copy ảnh khuôn mặt vào thư mục đó[cite: 6].

# 2. Huấn luyện (Training):
# Quét ảnh để tạo file vector embeddings.pkl:
[cite_start]python tools/train.py [cite: 7]

# 3. Khởi chạy ứng dụng:
[cite_start]python main.py [cite: 7]