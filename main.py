import cv2
from src.engine import FaceEngine  # <--- Import từ src
from src.ui import FaceAppUI       # <--- Import từ src
from src import config             # <--- Import từ src

class MainController:
    def __init__(self):
        self.engine = FaceEngine()
        self.ui = FaceAppUI(self)

        print(f"📷 Đang mở camera ID: {config.CAM_ID}...")
        self.cap = cv2.VideoCapture(config.CAM_ID)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.current_frame = None
        self.current_face = None
        self.current_name = "Unknown"
        self.video_loop()

    def video_loop(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv2.flip(frame, 1)
            self.current_frame = frame.copy()
            
            faces = self.engine.app.get(frame)
            if not faces:
                self.current_face = None
                self.ui.update_status("Scanning...", False)
            else:
                detected_names = []
                for face in faces:
                    box = face.bbox.astype(int)
                    name, score = self.engine.recognize_face(face.embedding)
                    detected_names.append(name)
                    color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                    x1, y1, x2, y2 = box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{name} ({score:.2f})", (x1, y1 - 10), cv2.FONT_HERSHEY_DUPLEX, 0.6, color, 2)

                if len(faces) == 1:
                    self.current_face = faces[0]
                    self.current_name = detected_names[0]
                    self.ui.update_status(self.current_name, self.current_name == "Unknown")
                else:
                    self.current_face = None
                    names_str = ", ".join(detected_names)
                    self.ui.name_label.config(text=f"Đông người: {names_str}", fg="orange")
                    self.ui.btn_add_photo.config(state="disabled", bg="gray")
                    self.ui.btn_register.config(state="disabled", bg="gray")

            self.ui.update_video_feed(frame)
        self.ui.after(30, self.video_loop)

    def handle_add_photo(self):
        if self.current_face and self.current_name != "Unknown":
            success = self.engine.register_user(self.current_frame, self.current_face.bbox, self.current_face.embedding, self.current_name)
            if success:
                from tkinter import messagebox
                messagebox.showinfo("Thành công", f"Đã thêm ảnh cho {self.current_name}!")

    def handle_register_new(self, new_name):
        if self.current_face:
            clean_name = new_name.strip().replace(" ", "_")
            if clean_name:
                success = self.engine.register_user(self.current_frame, self.current_face.bbox, self.current_face.embedding, clean_name)
                if success:
                    from tkinter import messagebox
                    messagebox.showinfo("Thành công", f"Đã đăng ký: {clean_name}!")

    def start(self):
        self.ui.mainloop()
        self.cap.release()

if __name__ == "__main__":
    app = MainController()
    app.start()