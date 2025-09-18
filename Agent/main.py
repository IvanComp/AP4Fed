import tkinter as tk, subprocess, threading, time, os, platform

def start_server():
    flags = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        flags["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **flags)
    time.sleep(1)

def list_models():
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True)
        return [line.split()[0] for line in out.stdout.strip().splitlines()] or ["llama3"]
    except:
        return ["llama3"]

def query(model, prompt):
    return subprocess.run(["ollama", "run", model], input=prompt, text=True, capture_output=True).stdout.strip()

def submit():
    prompt = input_box.get("1.0", tk.END).strip(); model = model_var.get()
    if not prompt: return
    output_box.config(state=tk.NORMAL); output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, "🧠 Generating a response...\n"); output_box.config(state=tk.DISABLED)
    threading.Thread(target=lambda: run(model, prompt)).start()

def run(model, prompt):
    resp = query(model, prompt)
    output_box.config(state=tk.NORMAL); output_box.delete("1.0", tk.END)
    output_box.insert(tk.END, resp); output_box.config(state=tk.DISABLED)

def on_keypress(event):
    if platform.system() == "Darwin" and (event.state & 0x10):
        input_box.insert(tk.INSERT, "\n")
    elif platform.system() != "Darwin" and (event.state & 0x04):
        input_box.insert(tk.INSERT, "\n")
    else:
        submit()
        return "break"
    
start_server()
root = tk.Tk(); root.title("AI Agent"); root.geometry("800x600"); root.configure(bg="white")
font_label = ("Arial", 14, "bold"); font_text = ("Courier", 16)
top_frame = tk.Frame(root, bg="white")
top_frame.pack(fill="x", padx=15, pady=10)
top_frame.columnconfigure(0, weight=1)
top_frame.columnconfigure(1, weight=0)
input_box = tk.Text(top_frame, height=5, wrap="word", font=font_text,
    bg="white", fg="black", insertbackground="black",
    padx=10, pady=8, spacing1=3, spacing3=3)
input_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
input_box.bind("<Return>", on_keypress)
right_panel = tk.Frame(top_frame, bg="white")
right_panel.grid(row=0, column=1, sticky="ne")
send_btn = tk.Button(right_panel, text="➤ Send", font=("Courier", 14),
    bg="#e0e0e0", fg="black", relief="flat", padx=12, pady=5,
    borderwidth=1, cursor="hand1", anchor="center")
send_btn.pack(pady=(0, 10))
send_btn.config(command=submit)
logo_path = os.path.join(os.path.dirname(__file__), "aiagentlogo.png")
logo = tk.PhotoImage(file=logo_path)
tk.Label(right_panel, image=logo, bg="white").pack()
model_frame = tk.Frame(right_panel, bg="white"); model_frame.pack(pady=(10, 0))
tk.Label(model_frame, text="Model:", font=font_label, bg="white", fg="black").pack(side="left")
model_var = tk.StringVar(value="llama3")
m = tk.OptionMenu(model_frame, model_var, *list_models())
m.config(font=font_text, bg="white", fg="black", highlightthickness=0)
m.pack(side="left", padx=(5, 0))
output_frame = tk.Frame(root, bg="white"); output_frame.pack(padx=15, pady=(0,15), fill="both", expand=True)
tk.Label(output_frame, text="Agent Response:", font=font_label, bg="white", fg="black").pack(anchor="w")
output_box = tk.Text(output_frame, height=15, wrap="word", font=("Courier", 16),
    bg="black", fg="#00FF00", insertbackground="#00FF00",
    relief="flat", padx=15, pady=10, spacing1=5, spacing3=5)
output_box.pack(fill="both", expand=True); output_box.config(state=tk.DISABLED)

root.mainloop()
