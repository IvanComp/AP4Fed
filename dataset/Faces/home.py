import os
import random
import threading
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox

# Percorso base (stesso livello di home.py)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def generate_realistic_face(img_size):
    """
    Genera un volto "realistico" in maniera schematica ma con maggiori dettagli.
    Inquadratura fissa (dal petto in su) con contorni, sopracciglia, capelli e lineamenti definiti.
    """
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    # Scegli tonalità cutanea
    skin_tones = [(255, 224, 189), (241, 194, 125), (224, 172, 105), (198, 134, 66)]
    skin_color = random.choice(skin_tones)

    # Testa (forma ovale)
    head_left = img_size * 0.3
    head_top = img_size * 0.1
    head_right = img_size * 0.7
    head_bottom = img_size * 0.55
    draw.ellipse([head_left, head_top, head_right, head_bottom], fill=skin_color, outline="black")

    # Capelli (forma ad arco sopra la testa)
    hair_color = random.choice(["brown", "black", "darkred"])
    hair_top = head_top - (img_size * 0.05)
    hair_bottom = head_top + (img_size * 0.1)
    draw.arc([head_left, hair_top, head_right, hair_bottom], start=0, end=180, fill=hair_color, width=8)

    # Occhi (ellissi bianche con pupille nere)
    eye_radius = (head_right - head_left) * 0.05
    eye_y = head_top + (head_bottom - head_top) * 0.4
    left_eye_center = (head_left + (head_right - head_left) * 0.35, eye_y)
    right_eye_center = (head_left + (head_right - head_left) * 0.65, eye_y)
    draw.ellipse([left_eye_center[0]-eye_radius, left_eye_center[1]-eye_radius,
                  left_eye_center[0]+eye_radius, left_eye_center[1]+eye_radius],
                 fill="white", outline="black")
    draw.ellipse([right_eye_center[0]-eye_radius, right_eye_center[1]-eye_radius,
                  right_eye_center[0]+eye_radius, right_eye_center[1]+eye_radius],
                 fill="white", outline="black")
    pupil_radius = eye_radius * 0.5
    draw.ellipse([left_eye_center[0]-pupil_radius, left_eye_center[1]-pupil_radius,
                  left_eye_center[0]+pupil_radius, left_eye_center[1]+pupil_radius],
                 fill="black")
    draw.ellipse([right_eye_center[0]-pupil_radius, right_eye_center[1]-pupil_radius,
                  right_eye_center[0]+pupil_radius, right_eye_center[1]+pupil_radius],
                 fill="black")

    # Sopracciglia (linee sottili sopra gli occhi)
    brow_offset_y = eye_radius * 1.5
    draw.line([ (left_eye_center[0]-eye_radius, left_eye_center[1]-brow_offset_y),
                (left_eye_center[0]+eye_radius, left_eye_center[1]-brow_offset_y) ],
              fill="black", width=2)
    draw.line([ (right_eye_center[0]-eye_radius, right_eye_center[1]-brow_offset_y),
                (right_eye_center[0]+eye_radius, right_eye_center[1]-brow_offset_y) ],
              fill="black", width=2)

    # Naso (linea verticale con lieve curva)
    nose_start = (img_size * 0.5, head_top + (head_bottom - head_top) * 0.45)
    nose_end = (img_size * 0.5, head_top + (head_bottom - head_top) * 0.65)
    draw.line([nose_start, nose_end], fill="black", width=1)

    # Bocca (arco sorridente)
    mouth_left = (head_left + (head_right - head_left)*0.4, head_top + (head_bottom - head_top)*0.75)
    mouth_right = (head_left + (head_right - head_left)*0.6, head_top + (head_bottom - head_top)*0.75)
    draw.arc([mouth_left[0], mouth_left[1]-8, mouth_right[0], mouth_right[1]+8],
             start=0, end=180, fill="black", width=2)

    # Collo (rettangolo sotto la testa)
    neck_left = img_size * 0.45
    neck_top = head_bottom
    neck_right = img_size * 0.55
    neck_bottom = head_bottom + (img_size * 0.15)
    draw.rectangle([neck_left, neck_top, neck_right, neck_bottom], fill=skin_color, outline="black")

    return img

def generate_stylized_face(img_size):
    """
    Genera un volto "stilizzato" con forme geometriche semplificate ma arricchite.
    Vengono disegnati anche capelli (a strati), sopracciglia e lineamenti stilizzati.
    """
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    skin_tones = [(255, 224, 189), (241, 194, 125), (224, 172, 105), (198, 134, 66)]
    skin_color = random.choice(skin_tones)

    # Volto a forma di poligono
    top_center = (img_size/2, img_size*0.15)
    left_point = (img_size*0.32, img_size*0.5)
    right_point = (img_size*0.68, img_size*0.5)
    chin = (img_size/2, img_size*0.75)
    draw.polygon([top_center, left_point, chin, right_point], fill=skin_color, outline="black")

    # Capelli: linee orizzontali sopra il volto
    hair_color = random.choice(["brown", "black", "darkred"])
    for i in range(3):
        y = img_size*0.1 + i* (img_size*0.02)
        draw.line([(img_size*0.3, y), (img_size*0.7, y)], fill=hair_color, width=3)

    # Occhi: cerchietti neri
    left_eye = (img_size*0.42, img_size*0.42)
    right_eye = (img_size*0.58, img_size*0.42)
    eye_radius = 4
    draw.ellipse([left_eye[0]-eye_radius, left_eye[1]-eye_radius,
                  left_eye[0]+eye_radius, left_eye[1]+eye_radius],
                 fill="black")
    draw.ellipse([right_eye[0]-eye_radius, right_eye[1]-eye_radius,
                  right_eye[0]+eye_radius, right_eye[1]+eye_radius],
                 fill="black")

    # Sopracciglia: archi sopra gli occhi
    draw.arc([left_eye[0]-eye_radius*1.5, left_eye[1]-eye_radius*3,
              left_eye[0]+eye_radius*1.5, left_eye[1]-eye_radius],
             start=0, end=180, fill="black", width=2)
    draw.arc([right_eye[0]-eye_radius*1.5, right_eye[1]-eye_radius*3,
              right_eye[0]+eye_radius*1.5, right_eye[1]-eye_radius],
             start=0, end=180, fill="black", width=2)

    # Naso: triangolino
    nose_top = (img_size*0.5, img_size*0.45)
    nose_left = (img_size*0.48, img_size*0.55)
    nose_right = (img_size*0.52, img_size*0.55)
    draw.polygon([nose_top, nose_left, nose_right], fill="black")

    # Bocca: linea dritta con un tocco di colore
    mouth_y = img_size*0.62
    draw.line([(img_size*0.45, mouth_y), (img_size*0.55, mouth_y)], fill="red", width=2)

    # Collo: rettangolo stilizzato
    neck_left = img_size * 0.44
    neck_top = img_size*0.75
    neck_right = img_size * 0.56
    neck_bottom = img_size * 0.9
    draw.rectangle([neck_left, neck_top, neck_right, neck_bottom], fill=skin_color, outline="black")

    return img

def generate_cartoon_face(img_size):
    """
    Genera un volto in stile "cartoon" con caratteristiche esagerate e dettagli marcati.
    Vengono aggiunti occhi grandi, sopracciglia spesse, capelli voluminosi e un sorriso accentuato.
    """
    img = Image.new("RGB", (img_size, img_size), "white")
    draw = ImageDraw.Draw(img)

    skin_tones = [(255, 224, 189), (241, 194, 125), (224, 172, 105), (198, 134, 66)]
    skin_color = random.choice(skin_tones)

    # Testa: grande cerchio
    head_center = (img_size/2, img_size*0.38)
    head_radius = img_size*0.3
    draw.ellipse([head_center[0]-head_radius, head_center[1]-head_radius,
                  head_center[0]+head_radius, head_center[1]+head_radius],
                 fill=skin_color, outline="black")

    # Capelli: ciuffo voluminoso (poligono irregolare)
    hair_color = random.choice(["brown", "black", "darkred"])
    hair_points = [
        (head_center[0]-head_radius, head_center[1]-head_radius*0.8),
        (head_center[0]-head_radius*0.5, head_center[1]-head_radius*1.2),
        (head_center[0], head_center[1]-head_radius*1.3),
        (head_center[0]+head_radius*0.5, head_center[1]-head_radius*1.2),
        (head_center[0]+head_radius, head_center[1]-head_radius*0.8)
    ]
    draw.polygon(hair_points, fill=hair_color)

    # Occhi: molto grandi con contorno spesso
    eye_radius = head_radius*0.35
    left_eye_center = (head_center[0] - head_radius*0.5, head_center[1] - head_radius*0.1)
    right_eye_center = (head_center[0] + head_radius*0.5, head_center[1] - head_radius*0.1)
    draw.ellipse([left_eye_center[0]-eye_radius, left_eye_center[1]-eye_radius,
                  left_eye_center[0]+eye_radius, left_eye_center[1]+eye_radius],
                 fill="white", outline="black", width=3)
    draw.ellipse([right_eye_center[0]-eye_radius, right_eye_center[1]-eye_radius,
                  right_eye_center[0]+eye_radius, right_eye_center[1]+eye_radius],
                 fill="white", outline="black", width=3)
    pupil_radius = eye_radius * 0.5
    draw.ellipse([left_eye_center[0]-pupil_radius, left_eye_center[1]-pupil_radius,
                  left_eye_center[0]+pupil_radius, left_eye_center[1]+pupil_radius],
                 fill="black")
    draw.ellipse([right_eye_center[0]-pupil_radius, right_eye_center[1]-pupil_radius,
                  right_eye_center[0]+pupil_radius, right_eye_center[1]+pupil_radius],
                 fill="black")

    # Sopracciglia spesse
    brow_y = left_eye_center[1] - eye_radius*1.1
    draw.rectangle([left_eye_center[0]-eye_radius, brow_y,
                    left_eye_center[0]+eye_radius, brow_y+4], fill="black")
    brow_y = right_eye_center[1] - eye_radius*1.1
    draw.rectangle([right_eye_center[0]-eye_radius, brow_y,
                    right_eye_center[0]+eye_radius, brow_y+4], fill="black")

    # Naso: piccolo e rotondo
    nose_center = (head_center[0], head_center[1]+head_radius*0.1)
    nose_radius = head_radius*0.1
    draw.ellipse([nose_center[0]-nose_radius, nose_center[1]-nose_radius,
                  nose_center[0]+nose_radius, nose_center[1]+nose_radius],
                 fill="black")

    # Bocca: sorriso ampio e marcato
    mouth_left = (head_center[0] - head_radius*0.5, head_center[1] + head_radius*0.5)
    mouth_right = (head_center[0] + head_radius*0.5, head_center[1] + head_radius*0.5)
    draw.arc([mouth_left[0], mouth_left[1]-10, mouth_right[0], mouth_right[1]+10],
             start=0, end=180, fill="red", width=4)

    # Collo: rettangolo sotto la testa
    neck_left = img_size * 0.45
    neck_top = head_center[1] + head_radius
    neck_right = img_size * 0.55
    neck_bottom = neck_top + img_size*0.2
    draw.rectangle([neck_left, neck_top, neck_right, neck_bottom], fill=skin_color, outline="black")

    return img

def generate_dataset(num_images, img_size, face_style, progress_callback):
    """
    Genera il dataset di immagini RGB di volti con lo stile scelto,
    salvandole in data/rgb_images accanto a home.py.
    """
    data_dir = os.path.join(BASE_DIR, 'data')
    output_dir = os.path.join(data_dir, 'rgb_images')
    os.makedirs(output_dir, exist_ok=True)

    for i in range(num_images):
        if face_style == "Realistic":
            img = generate_realistic_face(img_size)
        elif face_style == "Stylized":
            img = generate_stylized_face(img_size)
        elif face_style == "Cartoon":
            img = generate_cartoon_face(img_size)
        else:
            img = generate_realistic_face(img_size)
        filename = os.path.join(output_dir, f"face_{i}.png")
        img.save(filename)
        progress_callback(i + 1, num_images)

    dataset_path = os.path.abspath(output_dir)
    return dataset_path

def start_generation():
    try:
        num_images = int(num_images_scale.get())
        img_size = int(img_size_scale.get())
        face_style = style_var.get()
    except ValueError:
        messagebox.showerror("Error", "Please verify the input values.")
        return

    generate_button.config(state=tk.DISABLED)
    status_label.config(text="Generation in progress, please wait...")
    progress_bar['value'] = 0

    def progress_callback(current, total):
        progress = (current / total) * 100
        progress_bar['value'] = progress

    def generation_thread():
        dataset_path = generate_dataset(num_images, img_size, face_style, progress_callback)
        root.after(0, generation_complete, dataset_path)

    def generation_complete(dataset_path):
        status_label.config(text=f"Dataset generated: {num_images} images in {dataset_path}")
        generate_button.config(state=tk.NORMAL)

    t = threading.Thread(target=generation_thread)
    t.start()

# Interfaccia grafica
root = tk.Tk()
root.title("Face Dataset Generator")
root.geometry("400x400")
root.configure(bg="white")
root.resizable(False, False)

frame = tk.Frame(root, padx=10, pady=10, bg="white")
frame.pack(fill=tk.BOTH, expand=True)

tk.Label(frame, text="Number of images", bg="white", fg="black").grid(row=0, column=0, sticky="w", pady=2)
num_images_scale = tk.Scale(frame, from_=1, to=500, orient=tk.HORIZONTAL, bg="white", fg="black")
num_images_scale.set(100)
num_images_scale.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame, text="Image size (pixels)", bg="white", fg="black").grid(row=1, column=0, sticky="w", pady=2)
img_size_scale = tk.Scale(frame, from_=128, to=512, orient=tk.HORIZONTAL, bg="white", fg="black")
img_size_scale.set(256)
img_size_scale.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame, text="Face style", bg="white", fg="black").grid(row=2, column=0, sticky="w", pady=2)
style_var = tk.StringVar(frame)
style_var.set("Realistic")
option_menu = tk.OptionMenu(frame, style_var, "Realistic", "Stylized", "Cartoon")
option_menu.config(bg="white", fg="black")
option_menu.grid(row=2, column=1, padx=5, pady=2)

generate_button = tk.Button(frame, text="Generate Faces", command=start_generation, bg="white", fg="black")
generate_button.grid(row=3, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.grid(row=4, column=0, columnspan=2, pady=5)

status_label = tk.Label(frame, text="Set parameters and click 'Generate Faces'", bg="white", fg="black")
status_label.grid(row=5, column=0, columnspan=2, pady=5)

root.mainloop()
