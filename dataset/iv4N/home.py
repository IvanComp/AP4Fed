import os
import random
import threading
import numpy as np
import cv2
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import ttk, messagebox

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def random_cell_widths(num_classes, total_width, alpha):
    proportions = np.random.dirichlet([alpha] * num_classes)
    widths = (proportions * total_width).astype(int)
    diff = total_width - np.sum(widths)
    if diff != 0:
        idx = np.random.choice(num_classes)
        widths[idx] += diff
    for i in range(num_classes):
        if widths[i] < 1:
            diff = 1 - widths[i]
            widths[i] = 1
            for j in range(num_classes):
                if j != i and widths[j] > 1:
                    subtract = min(diff, widths[j] - 1)
                    widths[j] -= subtract
                    diff -= subtract
                    if diff == 0:
                        break
    return widths

def draw_large_circle_cell_rgb(img, x_min, x_max, y_min, y_max):
    cell_width = x_max - x_min
    cell_height = y_max - y_min
    cx = (x_min + x_max) // 2 + random.randint(-1, 1)
    cy = (y_min + y_max) // 2 + random.randint(-1, 1)
    margin = int(min(cell_width, cell_height) * 0.1)
    radius = max(1, min(cell_width, cell_height) // 2 - margin)
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.circle(img, (cx, cy), radius, color, -1)

def draw_large_rectangle_cell_rgb(img, x_min, x_max, y_min, y_max):
    cell_width = x_max - x_min
    cell_height = y_max - y_min
    margin_x = max(1, int(cell_width * 0.1))
    margin_y = max(1, int(cell_height * 0.1))
    pt1 = (x_min + random.randint(0, margin_x), y_min + random.randint(0, margin_y))
    pt2 = (x_max - random.randint(0, margin_x), y_max - random.randint(0, margin_y))
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.rectangle(img, pt1, pt2, color, -1)

def draw_large_triangle_cell_rgb(img, x_min, x_max, y_min, y_max):
    cell_width = x_max - x_min
    cell_height = y_max - y_min
    margin_x = max(1, int(cell_width * 0.1))
    margin_y = max(1, int(cell_height * 0.1))
    pt1 = [x_min + random.randint(0, margin_x), y_min + random.randint(0, margin_y)]
    pt2 = [x_max - random.randint(0, margin_x), y_min + random.randint(0, margin_y)]
    pt3 = [(x_min + x_max) // 2 + random.randint(-margin_x, margin_x), y_max - random.randint(0, margin_y)]
    pts = np.array([pt1, pt2, pt3])
    color = [random.randint(0, 255) for _ in range(3)]
    cv2.fillPoly(img, [pts], color)

def generate_dataset(num_images, img_size, num_classes, fg_ratio, alpha, progress_callback):
    data_dir = os.path.join(BASE_DIR, 'data')
    output_dirs = {
        'rgb': os.path.join(data_dir, 'rgb_images'),
        'label': os.path.join(data_dir, 'labels'),
        'depth': os.path.join(data_dir, 'depth_maps'),
        'labels_npy': os.path.join(data_dir, 'labels_npy')
    }
    for d in output_dirs.values():
        os.makedirs(d, exist_ok=True)
    total_fg_width = int(round(img_size * fg_ratio))
    left_margin = (img_size - total_fg_width) // 2
    for i in range(num_images):
        rgb_img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
        label_img = np.zeros((img_size, img_size), dtype=np.uint8)
        depth_img = np.zeros((img_size, img_size), dtype=np.uint8)
        cell_widths = random_cell_widths(num_classes, total_fg_width, alpha)
        x_bounds = [left_margin]
        for w in cell_widths:
            x_bounds.append(x_bounds[-1] + w)
        classes = list(range(1, num_classes + 1))
        random.shuffle(classes)
        for idx, cls in enumerate(classes):
            x_min = x_bounds[idx]
            x_max = x_bounds[idx + 1]
            y_min = 0
            y_max = img_size
            depth_value = random.randint(50, 255)
            shape_type = random.choice(['circle', 'rectangle', 'triangle'])
            if shape_type == 'circle':
                draw_large_circle_cell_rgb(rgb_img, x_min, x_max, y_min, y_max)
            elif shape_type == 'rectangle':
                draw_large_rectangle_cell_rgb(rgb_img, x_min, x_max, y_min, y_max)
            elif shape_type == 'triangle':
                draw_large_triangle_cell_rgb(rgb_img, x_min, x_max, y_min, y_max)
            label_img[y_min:y_max, x_min:x_max] = cls
            depth_img[y_min:y_max, x_min:x_max] = depth_value
        cv2.imwrite(os.path.join(output_dirs['rgb'], f"rgb_{i}.png"), rgb_img)
        cv2.imwrite(os.path.join(output_dirs['label'], f"label_{i}.png"), label_img)
        cv2.imwrite(os.path.join(output_dirs['depth'], f"depth_{i}.png"), depth_img)
        np.save(os.path.join(output_dirs['labels_npy'], f"label_{i}.npy"), label_img)
        progress_callback(i + 1, num_images)
    return os.path.abspath(data_dir)

def show_results(dataset_path, num_images, img_size, num_classes):
    sample_indices = [0, num_images // 2, num_images - 1] if num_images >= 3 else list(range(num_images))
    samples = []
    dirs = {
        'rgb': os.path.join(dataset_path, 'rgb_images'),
        'label': os.path.join(dataset_path, 'labels'),
        'depth': os.path.join(dataset_path, 'depth_maps')
    }
    for idx in sample_indices:
        rgb = Image.open(os.path.join(dirs['rgb'], f"rgb_{idx}.png"))
        label = Image.open(os.path.join(dirs['label'], f"label_{idx}.png"))
        depth = Image.open(os.path.join(dirs['depth'], f"depth_{idx}.png"))
        samples.append((rgb, label, depth))
    stats = {i: 0 for i in range(num_classes + 1)}
    labels_npy_dir = os.path.join(dataset_path, 'labels_npy')
    npy_files = [os.path.join(labels_npy_dir, f) for f in os.listdir(labels_npy_dir) if f.endswith('.npy')]
    for file in npy_files:
        arr = np.load(file)
        unique, counts = np.unique(arr, return_counts=True)
        for u, c in zip(unique, counts):
            stats[int(u)] += c
    total_pixels = num_images * (img_size * img_size)
    result_window = tk.Toplevel()
    result_window.title("Dataset Samples and Statistics")
    result_window.configure(bg="white")
    
    sample_frame = tk.Frame(result_window, bg="white")
    sample_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    tk.Label(sample_frame, text="Sample Images", font=("Arial", 14), bg="white", fg="black").grid(row=0, column=0, columnspan=3)
    image_refs = []
    for col, (rgb, label, depth) in enumerate(samples):
        rgb_thumb = rgb.resize((img_size * 4, img_size * 4))
        label_thumb = label.resize((img_size * 4, img_size * 4))
        depth_thumb = depth.resize((img_size * 4, img_size * 4))
        rgb_photo = ImageTk.PhotoImage(rgb_thumb)
        label_photo = ImageTk.PhotoImage(label_thumb)
        depth_photo = ImageTk.PhotoImage(depth_thumb)
        tk.Label(sample_frame, text="RGB", bg="white", fg="black").grid(row=1, column=col)
        tk.Label(sample_frame, image=rgb_photo, bg="white").grid(row=2, column=col)
        tk.Label(sample_frame, text="Label", bg="white", fg="black").grid(row=3, column=col)
        tk.Label(sample_frame, image=label_photo, bg="white").grid(row=4, column=col)
        tk.Label(sample_frame, text="Depth", bg="white", fg="black").grid(row=5, column=col)
        tk.Label(sample_frame, image=depth_photo, bg="white").grid(row=6, column=col)
        image_refs.append((rgb_photo, label_photo, depth_photo))
    result_window.image_refs = image_refs

    stats_frame = tk.Frame(result_window, bg="white")
    stats_frame.pack(padx=10, pady=10, fill=tk.BOTH, expand=True)
    tk.Label(stats_frame, text="Dataset Statistics", font=("Arial", 14), bg="white", fg="black").pack(anchor="w")
    stats_text = ""
    for lbl in range(num_classes + 1):
        stats_text += f"Label {lbl}: {stats[lbl]} pixels ({(stats[lbl] / total_pixels * 100):.2f}%)\n"
    tk.Label(stats_frame, text=stats_text, justify="left", bg="white", fg="black").pack(anchor="w")

def start_generation():
    try:
        num_images_val = int(num_images_scale.get())
        img_size_val = int(img_size_scale.get())
        num_classes_val = int(num_classes_scale.get())
        fg_ratio_val = float(fg_ratio_scale.get())
        alpha_val = float(alpha_scale.get())
    except ValueError:
        messagebox.showerror("Error", "Please verify the input values.")
        return
    generate_button.config(state=tk.DISABLED)
    status_label.config(text="Generation in progress, please wait...", bg="white", fg="black")
    progress_bar['value'] = 0
    def progress_callback(current, total):
        progress_bar['value'] = (current / total) * 100
    def generation_thread():
        dataset_path = generate_dataset(num_images_val, img_size_val, num_classes_val, fg_ratio_val, alpha_val, progress_callback)
        root.after(0, generation_complete, dataset_path, num_images_val, img_size_val, num_classes_val)
    def generation_complete(dataset_path, num_images_val, img_size_val, num_classes_val):
        status_label.config(text=f"Dataset generated: {num_images_val} images in {dataset_path}", bg="white", fg="black")
        generate_button.config(state=tk.NORMAL)
        show_results(dataset_path, num_images_val, img_size_val, num_classes_val)
    t = threading.Thread(target=generation_thread)
    t.start()

root = tk.Tk()
root.title("Dataset Generator")
root.geometry("400x450")
root.configure(bg="white")
root.resizable(False, False)

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

tk.Label(frame, text="Number of classes", bg="white", fg="black").grid(row=2, column=0, sticky="w", pady=2)
num_classes_scale = tk.Scale(frame, from_=2, to=10, orient=tk.HORIZONTAL, bg="white", fg="black")
num_classes_scale.set(5)
num_classes_scale.grid(row=2, column=1, padx=5, pady=2)

tk.Label(frame, text="Foreground Ratio", bg="white", fg="black").grid(row=3, column=0, sticky="w", pady=2)
fg_ratio_scale = tk.Scale(frame, from_=0.5, to=1.0, resolution=0.01, orient=tk.HORIZONTAL, bg="white", fg="black")
fg_ratio_scale.set(0.77)
fg_ratio_scale.grid(row=3, column=1, padx=5, pady=2)

tk.Label(frame, text="Alpha (Dirichlet)", bg="white", fg="black").grid(row=4, column=0, sticky="w", pady=2)
alpha_scale = tk.Scale(frame, from_=0.1, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, bg="white", fg="black")
alpha_scale.set(0.5)
alpha_scale.grid(row=4, column=1, padx=5, pady=2)

generate_button = tk.Button(frame, text="Generate Dataset", command=start_generation, bg="white", fg="black")
generate_button.grid(row=5, column=0, columnspan=2, pady=10)

progress_bar = ttk.Progressbar(frame, orient=tk.HORIZONTAL, length=300, mode='determinate')
progress_bar.grid(row=6, column=0, columnspan=2, pady=5)

status_label = tk.Label(frame, text="Set the parameters and click 'Generate Dataset'", bg="white", fg="black")
status_label.grid(row=7, column=0, columnspan=2, pady=5)

root.mainloop()
