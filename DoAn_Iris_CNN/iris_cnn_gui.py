import os
import json
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf

# =========================
# 1. LOAD MODEL VA CLASS
# =========================
MODEL_PATH = "flower_model_best.keras"
CLASS_PATH = "class_names.json"

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Khong tim thay file flower_model_best.keras")

if not os.path.exists(CLASS_PATH):
    raise FileNotFoundError("Khong tim thay file class_names.json")

model = tf.keras.models.load_model(MODEL_PATH)

with open(CLASS_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)

IMG_SIZE = (224, 224)
selected_image_path = None


# =========================
# 2. HAM CHON ANH
# =========================
def choose_image():
    global selected_image_path

    file_path = filedialog.askopenfilename(
        title="Chon anh hoa",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    selected_image_path = file_path

    img = Image.open(file_path).convert("RGB")
    img.thumbnail((350, 350))

    img_tk = ImageTk.PhotoImage(img)
    image_label.config(image=img_tk)
    image_label.image = img_tk

    result_label.config(text="Ket qua: Chua du doan")
    confidence_label.config(text="Do tin cay: ")
    detail_text.delete("1.0", tk.END)


# =========================
# 3. HAM DU DOAN
# =========================
def predict_image():
    if selected_image_path is None:
        messagebox.showwarning("Thong bao", "Vui long chon anh truoc!")
        return

    try:
        img = Image.open(selected_image_path).convert("RGB")
        img = img.resize(IMG_SIZE)

        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # KHONG chia /255 vi model moi da co layer Rescaling ben trong
        predictions = model.predict(img_array)

        probabilities = predictions[0]
        predicted_index = np.argmax(probabilities)
        predicted_class = class_names[predicted_index]
        confidence = probabilities[predicted_index] * 100

        result_label.config(
            text=f"Ket qua: {predicted_class}",
            fg="#1b5e20"
        )

        confidence_label.config(
            text=f"Do tin cay: {confidence:.2f}%",
            fg="#0d47a1"
        )

        detail_text.delete("1.0", tk.END)
        detail_text.insert(tk.END, "Xac suat tung lop:\n\n")

        sorted_indices = np.argsort(probabilities)[::-1]

        for idx in sorted_indices:
            class_name = class_names[idx]
            prob = probabilities[idx] * 100
            detail_text.insert(tk.END, f"{class_name}: {prob:.2f}%\n")

    except Exception as e:
        messagebox.showerror("Loi", f"Khong the du doan anh!\n{e}")


# =========================
# 4. GIAO DIEN TKINTER
# =========================
root = tk.Tk()
root.title("Flower Classification - CNN")
root.geometry("850x600")
root.resizable(False, False)
root.configure(bg="#f5f7fa")

title_label = tk.Label(
    root,
    text="HE THONG PHAN LOAI HOA BANG CNN",
    font=("Arial", 20, "bold"),
    bg="#f5f7fa",
    fg="#1a237e"
)
title_label.pack(pady=15)

main_frame = tk.Frame(root, bg="#f5f7fa")
main_frame.pack(fill="both", expand=True, padx=20, pady=10)

left_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="groove")
left_frame.place(x=20, y=10, width=400, height=470)

right_frame = tk.Frame(main_frame, bg="#ffffff", bd=2, relief="groove")
right_frame.place(x=440, y=10, width=370, height=470)

image_label = tk.Label(
    left_frame,
    text="Chua chon anh",
    bg="#eeeeee",
    fg="#555555",
    font=("Arial", 13)
)
image_label.place(x=25, y=20, width=350, height=350)

choose_btn = tk.Button(
    left_frame,
    text="Chon anh",
    font=("Arial", 12, "bold"),
    bg="#1976d2",
    fg="white",
    command=choose_image
)
choose_btn.place(x=65, y=390, width=120, height=40)

predict_btn = tk.Button(
    left_frame,
    text="Du doan",
    font=("Arial", 12, "bold"),
    bg="#388e3c",
    fg="white",
    command=predict_image
)
predict_btn.place(x=215, y=390, width=120, height=40)

info_title = tk.Label(
    right_frame,
    text="KET QUA DU DOAN",
    font=("Arial", 16, "bold"),
    bg="#ffffff",
    fg="#b71c1c"
)
info_title.pack(pady=20)

result_label = tk.Label(
    right_frame,
    text="Ket qua: Chua du doan",
    font=("Arial", 14, "bold"),
    bg="#ffffff",
    fg="#333333"
)
result_label.pack(pady=10)

confidence_label = tk.Label(
    right_frame,
    text="Do tin cay:",
    font=("Arial", 13),
    bg="#ffffff",
    fg="#333333"
)
confidence_label.pack(pady=5)

detail_text = tk.Text(
    right_frame,
    font=("Consolas", 11),
    bg="#f9f9f9",
    fg="#222222"
)
detail_text.place(x=25, y=160, width=320, height=280)

footer_label = tk.Label(
    root,
    text="Model: CNN Thuan (Custom) | Dataset: 10 loai hoa",
    font=("Arial", 10),
    bg="#f5f7fa",
    fg="#555555"
)
footer_label.pack(pady=5)

root.mainloop()