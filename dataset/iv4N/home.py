import os
import random
import threading
import numpy as np
from PIL import Image, ImageDraw
import tkinter as tk
from tkinter import ttk, messagebox

# Calcola il percorso della directory in cui si trova home.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def draw_random_circle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size):
    center = (random.randint(0, img_size - 1), random.randint(0, img_size - 1))
    radius = random.randint(3, img_size // 3)
    left = center[0] - radius
    top = center[1] - radius
    right = center[0] + radius
    bottom = center[1] + radius
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw_rgb.ellipse([left, top, right, bottom], fill=color)
    draw_label.ellipse([left, top, right, bottom], fill=label_value)
    draw_depth.ellipse([left, top, right, bottom], fill=depth_value)

def draw_random_rectangle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size):
    x1 = random.randint(0, img_size - 10)
    y1 = random.randint(0, img_size - 10)
    x2 = min(x1 + random.randint(5, img_size // 2), img_size - 1)
    y2 = min(y1 + random.randint(5, img_size // 2), img_size - 1)
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw_rgb.rectangle([x1, y1, x2, y2], fill=color)
    draw_label.rectangle([x1, y1, x2, y2], fill=label_value)
    draw_depth.rectangle([x1, y1, x2, y2], fill=depth_value)

def draw_random_triangle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size):
    pts = [(random.randint(0, img_size - 1), random.randint(0, img_size - 1)) for _ in range(3)]
    color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
    draw_rgb.polygon(pts, fill=color)
    draw_label.polygon(pts, fill=label_value)
    draw_depth.polygon(pts, fill=depth_value)

def generate_dataset(num_images, img_size, min_shapes, max_shapes, num_classes, progress_callback):
    # Salva la cartella "data" nello stesso livello del file home.py
    data_dir = os.path.join(BASE_DIR, 'data')
    output_dirs = {
        'rgb': os.path.join(data_dir, 'rgb_images'),
        'label': os.path.join(data_dir, 'labels'),
        'depth': os.path.join(data_dir, 'depth_maps'),
        'labels_npy': os.path.join(data_dir, 'labels_npy')
    }
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)

    for i in range(num_images):
        rgb_img = Image.new("RGB", (img_size, img_size), (255, 255, 255))
        label_img = Image.new("L", (img_size, img_size), 0)
        depth_img = Image.new("L", (img_size, img_size), 0)

        draw_rgb = ImageDraw.Draw(rgb_img)
        draw_label = ImageDraw.Draw(label_img)
        draw_depth = ImageDraw.Draw(depth_img)

        num_shapes = random.randint(min_shapes, max_shapes)
        for _ in range(num_shapes):
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            label_value = random.randint(1, num_classes)
            depth_value = random.randint(50, 255)
            if shape_type == 'circle':
                draw_random_circle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size)
            elif shape_type == 'rectangle':
                draw_random_rectangle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size)
            elif shape_type == 'triangle':
                draw_random_triangle(draw_rgb, draw_label, draw_depth, label_value, depth_value, img_size)

        rgb_filename = os.path.join(output_dirs['rgb'], f"rgb_{i}.png")
        label_filename = os.path.join(output_dirs['label'], f"label_{i}.png")
        depth_filename = os.path.join(output_dirs['depth'], f"depth_{i}.png")
        rgb_img.save(rgb_filename)
        label_img.save(label_filename)
        depth_img.save(depth_filename)

        # Salva l'array NumPy delle etichette nella cartella labels_npy
        npy_filename = os.path.join(output_dirs['labels_npy'], f"label_{i}.npy")
        # Convertiamo la label in array NumPy
        label_array = np.array(label_img)
        np.save(npy_filename, label_array)

        progress_callback(i + 1, num_images)

    dataset_path = os.path.abspath(data_dir)
    return dataset_path

def start_generation():
    try:
        num_images = int(num_images_scale.get())
        img_size = int(img_size_scale.get())
        min_shapes = int(min_shapes_scale.get())
        max_shapes = int(max_shapes_scale.get())
        num_classes = int(num_classes_scale.get())
    except ValueError:
        messagebox.showerror("Error", "Please verify the input values.")
        return

    if min_shapes > max_shapes:
        messagebox.showerror("Error", "Minimum number of shapes cannot be greater than maximum number of shapes.")
        return

    generate_button.config(state=tk.DISABLED)
    status_label.config(text="Generation in progress, please wait...")
    progress_bar['value'] = 0

    def progress_callback(current, total):
        progress = (current / total) * 100
        progress_bar['value'] = progress

    def generation_thread():
        dataset_path = generate_dataset(num_images, img_size, min_shapes, max_shapes, num_classes, progress_callback)
        root.after(0, generation_complete, dataset_path)

    def generation_complete(dataset_path):
        status_label.config(text=f"Dataset generated: {num_images} images in {dataset_path}")
        generate_button.config(state=tk.NORMAL)

    t = threading.Thread(target=generation_thread)
    t.start()

root = tk.Tk()
root.title("Dataset Generator")
root.geometry("400x400")
root.configure(bg="white")
root.resizable(False, False)  # La finestra non è ridimensionabile

frame = tk.Frame(root, padx=10, pady=10, bg="white")
frame.pack(fill=tk.BOTH, expand=True)

tk.Label(frame, text="Number of images", bg="white", fg="black").grid(row=0, column=0, sticky="w", pady=2)
num_images_scale = tk.Scale(frame, from_=100, to=100000, orient=tk.HORIZONTAL, bg="white", fg="black")
num_images_scale.set(50000)
num_images_scale.grid(row=0, column=1, padx=5, pady=2)

tk.Label(frame, text="Image size (NxN)", bg="white", fg="black").grid(row=1, column=0, sticky="w", pady=2)
img_size_scale = tk.Scale(frame, from_=16, to=256, orient=tk.HORIZONTAL, bg="white", fg="black")
img_size_scale.set(32)
img_size_scale.grid(row=1, column=1, padx=5, pady=2)

tk.Label(frame, text="Minimum number of shapes", bg="white", fg="black").grid(row=2, column=0, sticky="w", pady=2)
min_shapes_scale = tk.Scale(frame, from_=1, to=10, orient=tk.HORIZONTAL, bg="white", fg="black")
min_shapes_scale.set(1)
min_shapes_scale.grid(row=2, column=1, padx=5, pady=2)

tk.Label(frame, text="Maximum number of shapes", bg="white", fg="black").grid(row=3, column=0, sticky="w", pady=2)
max_shapes_scale = tk.Scale(frame, from_=1, to=10, orient=tk.HORIZONTAL, bg="white", fg="black")
max_shapes_scale.set(4)
max_shapes_scale.grid(row=3, column=1, padx=5, pady=2)

tk.Label(frame, text="Number of classes", bg="white", fg="black").grid(row=4, column=0, sticky="w", pady=2)
num_classes_scale = tk.Scale(frame, from_=2, to=10, orient=tk.HORIZONTAL, bg="white", fg="black")
num_classes_scale.set(5)
num_classes_scale.grid(row=4, column=1, padx=5, pady=2)

generate_button = tk.Button(frame, text="Generate Dataset", command=start_generation, bg="white", fg="black")
generate_button.grid(row=5, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.grid(row=6, column=0, columnspan=2, pady=5)

status_label = tk.Label(frame, text="Set the parameters and click 'Generate Dataset'", bg="white", fg="black")
status_label.grid(row=7, column=0, columnspan=2, pady=5)

root.mainloop()
