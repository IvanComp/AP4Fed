import tkinter as tk, subprocess, threading, time, os, platform, shutil, sys, tkinter.messagebox as msg
def install_ollama():
    s=platform.system()
    if s=="Darwin": subprocess.run(["brew","install","ollama"])
    elif s=="Linux": subprocess.run("curl -fsSL https://ollama.com/install.sh | sh",shell=True)
    elif s=="Windows":
        u="https://ollama.com/download/OllamaSetup.exe"
        p=os.path.join(os.getenv("TEMP"),"OllamaSetup.exe")
        subprocess.run(["curl","-L",u,"-o",p],shell=True); subprocess.run([p],shell=True)
    else: sys.exit(1)
def ensure_ollama():
    if not shutil.which("ollama"):
        if msg.askyesno("Ollama not found","Ollama is not installed. Install now?"):
            install_ollama(); time.sleep(5)
            if not shutil.which("ollama"): msg.showerror("Install failed","❌ Ollama installation failed."); sys.exit(1)
        else: sys.exit(0)
def start_server():
    f={"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL}
    if platform.system()=="Windows": f["creationflags"]=subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama","serve"],**f); time.sleep(1)
def list_models():
    try: return [l.split()[0] for l in subprocess.run(["ollama","list"],capture_output=True,text=True).stdout.strip().splitlines()] or ["llama3"]
    except: return ["llama3"]
def query(m,p): return subprocess.run(["ollama","run",m],input=p,text=True,capture_output=True).stdout.strip()
def submit():
    p=input_box.get("1.0",tk.END).strip(); m=model_var.get()
    if not p: return
    output_box.config(state=tk.NORMAL); output_box.delete("1.0",tk.END)
    output_box.insert(tk.END,"🧠 Generating a response...\n"); output_box.config(state=tk.DISABLED)
    threading.Thread(target=lambda: run(m,p)).start()
def run(m,p):
    r=query(m,p)
    output_box.config(state=tk.NORMAL); output_box.delete("1.0",tk.END)
    output_box.insert(tk.END,r); output_box.config(state=tk.DISABLED)
def on_keypress(e):
    if platform.system()=="Darwin" and (e.state&0x10): input_box.insert(tk.INSERT,"\n")
    elif platform.system()!="Darwin" and (e.state&0x04): input_box.insert(tk.INSERT,"\n")
    else: submit(); return "break"
ensure_ollama(); start_server()
root=tk.Tk(); root.title("AI Agent"); root.geometry("1100x600"); root.configure(bg="white")
fl=("Courier",16,"bold"); ft=("Courier",16)
top=tk.Frame(root,bg="white"); top.pack(fill="x",padx=15,pady=10); top.columnconfigure(0,weight=1); top.columnconfigure(1,weight=0)
input_box=tk.Text(top,height=5,wrap="word",font=ft,bg="white",fg="black",insertbackground="black",padx=10,pady=8,spacing1=3,spacing3=3)
input_box.grid(row=0,column=0,sticky="nsew",padx=(0,10)); input_box.bind("<Return>",on_keypress)
right=tk.Frame(top,bg="white"); right.grid(row=0,column=1,sticky="ne")
send_btn=tk.Button(right,text="➤ Send",font=("Courier",14),bg="#e0e0e0",fg="black",relief="flat",padx=12,pady=5,borderwidth=1,cursor="hand1",anchor="center")
send_btn.pack(pady=(0,10)); send_btn.config(command=submit)
lp=os.path.join(os.path.dirname(__file__),"aiagentlogo.png")
logo=tk.PhotoImage(file=lp) if os.path.exists(lp) else None
tk.Label(right,image=logo if logo else "",text="🤖" if not logo else "",font=("Arial",36),bg="white").pack()
mf=tk.Frame(right,bg="white"); mf.pack(pady=(10,0))
tk.Label(mf,text="Model:",font=ft,bg="white",fg="black").pack(side="left")
model_var=tk.StringVar(value="llama3")
m=tk.OptionMenu(mf,model_var,*list_models()); m.config(font=ft,bg="white",fg="black",highlightthickness=0); m.pack(side="left",padx=(5,0))
of=tk.Frame(root,bg="white"); of.pack(padx=15,pady=(0,15),fill="both",expand=True)
tk.Label(of,text="Agent Response:",font=("Courier",14),bg="white",fg="black").pack(anchor="w")
output_box=tk.Text(of,height=15,wrap="word",font=fl,bg="black",fg="#00FF00",insertbackground="#00FF00",relief="flat",padx=15,pady=10,spacing1=5,spacing3=5)
output_box.pack(fill="both",expand=True); output_box.config(state=tk.DISABLED)
root.mainloop()
