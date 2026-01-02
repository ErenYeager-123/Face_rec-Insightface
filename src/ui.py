import tkinter as tk
from tkinter import simpledialog
from PIL import Image, ImageTk
import cv2
from src import config   

class FaceAppUI(tk.Tk):
    def __init__(self, main_controller):
        super().__init__()
        self.controller = main_controller
        self.title("Face Recognition System")
        self.geometry("1000x520")
        self.config(bg="#211241") 
        self.create_layout()

    def create_layout(self):
        # KHUNG VIDEO
        self.video_frame = tk.Label(self, bg="black")
        self.video_frame.place(x=20, y=20, width=config.UI_WIDTH, height=config.UI_HEIGHT)

        # KHUNG ĐIỀU KHIỂN
        control_panel = tk.Frame(self, bg="light blue")
        control_panel.place(x=680, y=20, width=300, height=config.UI_HEIGHT)

        tk.Label(control_panel, text="BẢNG ĐIỀU KHIỂN", font=("Calibri", 16, "bold"), bg="light blue", fg="gray20").pack(pady=15)

        self.btn_add_photo = tk.Button(control_panel, text="📸 THÊM ẢNH (Người cũ)", 
                                       font=("Calibri", 12, "bold"), bg="turquoise3", fg="gray15",
                                       state="disabled", command=self.on_add_photo, height=2)
        self.btn_add_photo.pack(fill="x", padx=20, pady=10)

        self.btn_register = tk.Button(control_panel, text="🆕 ĐĂNG KÝ MỚI", 
                                      font=("Calibri", 12, "bold"), bg="forest green", fg="white",
                                      state="disabled", command=self.on_register_new, height=2)
        self.btn_register.pack(fill="x", padx=20, pady=10)

        self.name_label = tk.Label(control_panel, text="Đang quét...", font=("Calibri", 14), bg="light blue", fg="gray")
        self.name_label.pack(pady=15)

        tk.Button(control_panel, text="THOÁT", command=self.quit, bg="red2", fg="white").pack(side="bottom", pady=20)

    def update_video_feed(self, cv2_image):
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        rgb_image = cv2.resize(rgb_image, (config.UI_WIDTH, config.UI_HEIGHT))
        pil_image = Image.fromarray(rgb_image)
        imgtk = ImageTk.PhotoImage(image=pil_image)
        self.video_frame.imgtk = imgtk
        self.video_frame.configure(image=imgtk)

    def update_status(self, name, is_unknown):
        if name == "Scanning...":
            self.name_label.config(text="Đang quét...", fg="gray")
            self.btn_add_photo.config(state="disabled", bg="gray")
            self.btn_register.config(state="disabled", bg="gray")
            return

        self.name_label.config(text=f"Người: {name}", fg="black" if not is_unknown else "red")
        if is_unknown:
            self.btn_register.config(state="normal", bg="forest green")
            self.btn_add_photo.config(state="disabled", bg="gray")
        else:
            self.btn_register.config(state="disabled", bg="gray")
            self.btn_add_photo.config(state="normal", bg="turquoise3")

    def on_add_photo(self):
        self.controller.handle_add_photo()

    def on_register_new(self):
        name = simpledialog.askstring("Đăng ký", "Nhập tên người dùng mới:")
        if name:
            self.controller.handle_register_new(name)