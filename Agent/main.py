import tkinter as tk
import subprocess, threading, time, os, platform

def start_server():
    flags = dict(stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if platform.system() == "Windows":
        flags["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **flags)

def list_models():
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, check=True).stdout
        return [line.split()[0] for line in out.strip().splitlines()] or ["llama3"]
    except:
        return ["llama3"]

def query(model, prompt):
    try:
        out = subprocess.run(["ollama", "run", model], input=prompt, text=True, capture_output=True, check=True)
        return out.stdout.strip()
    except subprocess.CalledProcessError as e:
        return f"Error: {e.stderr.strip()}"

def submit():
    prompt = input_box.get("1.0", tk.END).strip()
    model = model_var.get()
    if not prompt: return
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, "🧠 Generating a response...\n")
    output_box.config(state=tk.DISABLED)
    threading.Thread(target=lambda: run(model, prompt)).start()

def run(model, prompt):
    resp = query(model, prompt)
    output_box.config(state=tk.NORMAL)
    output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, resp)
    output_box.config(state=tk.DISABLED)

start_server()
root = tk.Tk()
root.title("Ollama Local Agent")
root.geometry("800x600")
root.configure(bg="white")
font_label = ("Arial", 11)
font_text = ("Arial", 11)

model_frame = tk.Frame(root, bg="white")
model_frame.pack(padx=15, pady=(10, 0), fill="x")
tk.Label(model_frame, text="Model:", font=font_label, bg="white", fg="black").pack(side="left")
model_var = tk.StringVar(value="llama3")
model_menu = tk.OptionMenu(model_frame, model_var, *list_models())
model_menu.config(font=font_label, bg="white", fg="black", highlightthickness=0)
model_menu.pack(side="left", padx=(10, 0))

input_frame = tk.Frame(root, bg="white")
input_frame.pack(padx=15, pady=10, fill="x")
tk.Label(input_frame, text="Enter your prompt:", font=font_label, bg="white", fg="black").pack(anchor="w")
input_box = tk.Text(input_frame, height=6, wrap="word", font=font_text, bg="white", fg="black", insertbackground="black")
input_box.pack(fill="x", pady=5)
tk.Button(input_frame, text="Send", command=submit, font=font_label, bg="white", fg="black", relief="solid", borderwidth=1).pack(anchor="e", pady=5)

output_frame = tk.Frame(root, bg="white")
output_frame.pack(padx=15, pady=(0, 15), fill="both", expand=True)
tk.Label(output_frame, text="Agent response:", font=font_label, bg="white", fg="black").pack(anchor="w")
output_box = tk.Text(output_frame, height=15, wrap="word", font=("Courier", 11), bg="white", fg="black", insertbackground="black")
output_box.pack(fill="both", expand=True)
output_box.config(state=tk.DISABLED)

root.mainloop()
