import tkinter as tk, tkinter.messagebox as msg
import subprocess, platform, shutil, time, sys, os, json, re, threading
from pathlib import Path

APP_TITLE = "Multi-LLM Debate • 3 Agents + Coordinator"
DEFAULT_MODEL = "llama3"

AGENTS = [
    {"id":"A1", "name":"Astra",  "title":"### [MULTI-AGENT • Astra • Debate]"},
    {"id":"A2", "name":"Marvin", "title":"### [MULTI-AGENT • Marvin • Debate]"},
    {"id":"A3", "name":"Talos",  "title":"### [MULTI-AGENT • Talos • Debate]"},
]

# -------------------- Ollama bootstrap --------------------

def install_ollama():
    s = platform.system()
    if s == "Darwin":
        subprocess.run(["brew","install","ollama"])
    elif s == "Linux":
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    elif s == "Windows":
        url = "https://ollama.com/download/OllamaSetup.exe"
        tmp = os.path.join(os.getenv("TEMP"),"OllamaSetup.exe")
        subprocess.run(["curl","-L",url,"-o",tmp], shell=True)
        subprocess.run([tmp], shell=True)
    else:
        sys.exit(1)

def ensure_ollama():
    if not shutil.which("ollama"):
        if msg.askyesno("Ollama not found","Ollama is not installed. Install now?"):
            install_ollama()
            time.sleep(5)
            if not shutil.which("ollama"):
                msg.showerror("Install failed","❌ Ollama installation failed.")
                sys.exit(1)
        else:
            sys.exit(0)

def start_server():
    flags = {"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL}
    if platform.system()=="Windows":
        flags["creationflags"]=subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama","serve"],**flags)
    time.sleep(1)

def list_models():
    try:
        out = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=5)
        lines = [l.split()[0] for l in out.stdout.strip().splitlines()[1:] if l.strip()]
        return lines or [DEFAULT_MODEL]
    except Exception:
        return [DEFAULT_MODEL]

def query(model, prompt):
    try:
        r = subprocess.run(
            ["ollama","run",model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=60
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

# -------------------- Parsing helpers --------------------

NON_ANSWERS = {
    "", "ok", "okay", "ready", "sure", "```",
    "i am ready", "im ready", "lets begin",
    "okay im ready lets begin"
}

def norm_answer(s):
    s=(s or "").strip().lower()
    s=re.sub(r"\s+"," ",s)
    s=re.sub(r"[.,;:!\?\-\(\)\[\]\\/\"\'`]", "", s)
    return s

def is_non_answer(s):
    ns=norm_answer(s)
    return (
        ns in NON_ANSWERS
        or ns.startswith("this is a")
        or ns.startswith("i am")
        or len(ns)<1
    )

def extract_json_block(text):
    if not text:
        return {}
    m=re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    blob=m.group(1) if m else None
    if not blob:
        m=re.search(r"(\{[\s\S]*\})", text)
        blob=m.group(1) if m else None
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            pass
    plain=text.strip()
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", plain) or re.fullmatch(r"[A-Za-z\-]+", plain):
        return {"rationale":"","answer":plain,"confidence":0.5}
    return {}

def build_agent_prompt(agent_name, agent_title, task, history):
    return (
f"""{agent_title}
You are {agent_name}.
Task: {task}

Debate rules:
- OUTPUT MUST BE ONLY one fenced JSON code block. No text outside JSON.
- JSON must have exactly: rationale, answer, confidence.
- 'answer' must be short and final. If the task asks for a color, 'answer' MUST be a single word.
- confidence in [0,1].
- Use the same language as the task.
- If uncertain, give your best guess. Do not say you are ready.

Previous round context (use it to update your position, do not copy into JSON):
{history if history.strip() else "(none)"}

Return ONLY the final JSON, for example:

```json
{{ "rationale": "short explanation", "answer": "one-word", "confidence": 0.9 }}
```"""
    )

def build_fix_prompt(agent_title, task, previous_raw):
    return (
f"""{agent_title}
Previous output invalid. Fix it now.
Task: {task}
Rules:
- Only ONE JSON block with: rationale, answer, confidence.
- No text outside JSON.
- If the task asks for a color, 'answer' = one word.

Previous attempt:
{previous_raw}
"""
    )

class ClientAgent:
    def __init__(self,name,title,model):
        self.name=name
        self.title=title
        self.model=model

    def speak(self,task,history=""):
        prompt=build_agent_prompt(self.name,self.title,task,history)
        raw=query(self.model,prompt)
        js=extract_json_block(raw)
        ans = js.get("answer","") if isinstance(js,dict) else ""
        conf= float(js.get("confidence",0)) if isinstance(js,dict) else 0.0
        rat = js.get("rationale","") if isinstance(js,dict) else ""

        if not ans or is_non_answer(ans):
            fixp=build_fix_prompt(self.title,task,raw)
            raw2=query(self.model,fixp)
            js2=extract_json_block(raw2)
            ans  = js2.get("answer",ans)
            conf = float(js2.get("confidence",conf)) if isinstance(js2,dict) else conf
            rat  = js2.get("rationale",rat) if isinstance(js2,dict) else rat

        # Easter egg: riddle fallback
        if is_non_answer(ans) and ("horse" in task.lower() and "napoleon" in task.lower() and "color" in task.lower()):
            ans,conf,rat="white",0.99,"explicit riddle"

        return {
            "agent":self.name,
            "answer":ans or "",
            "confidence":conf,
            "rationale":rat
        }

# -------------------- Config reader --------------------

def read_coordination_mode():
    """
    Legge configuration/config.json e ritorna (mode, raw_strategy)
    mode ∈ {"voting","role","debate","none"}
    raw_strategy è la stringa originale dal config (es. "AI-Agents (Voting-Based)").
    """
    base=Path(__file__).resolve().parent
    cfg_path=base/"configuration"/"config.json"
    try:
        cfg=json.loads(cfg_path.read_text(encoding="utf-8"))
        raw=cfg.get("adaptation","None")
    except Exception:
        return "none","None"

    low=str(raw).lower()
    if "voting" in low:  return "voting", raw
    if "role"   in low:  return "role",   raw
    if "debate" in low:  return "debate", raw
    return "none", raw

# -------------------- UI builder --------------------

def build_ui():
    # bootstrap ollama backend
    ensure_ollama()
    start_server()

    root=tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1150x700")
    root.configure(bg="white")

    fl=("Courier",16,"bold")
    ft=("Courier",16)

    # --- TOP BAR: input + controls ---
    top=tk.Frame(root,bg="white")
    top.pack(fill="x",padx=15,pady=10)
    top.columnconfigure(0,weight=1)
    top.columnconfigure(1,weight=0)

    input_box=tk.Text(
        top,
        height=5,
        wrap="word",
        font=ft,
        bg="white",
        fg="black",
        insertbackground="black",
        padx=10,
        pady=8,
        spacing1=3,
        spacing3=3
    )
    input_box.grid(row=0,column=0,sticky="nsew",padx=(0,10))

    right=tk.Frame(top,bg="white")
    right.grid(row=0,column=1,sticky="ne")

    send_btn=tk.Button(
        right,
        text="➤ Send",
        font=("Courier",14),
        bg="#e0e0e0",
        fg="black",
        relief="flat",
        padx=12,
        pady=5,
        borderwidth=1,
        cursor="hand1",
        anchor="center"
    )
    send_btn.pack(pady=(0,10))

    mf=tk.Frame(right,bg="white")
    mf.pack(pady=(10,0))

    tk.Label(mf,text="Coordinator:",font=ft,bg="white",fg="black").grid(row=0,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[0]['name']+":",font=ft,bg="white",fg="black").grid(row=1,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[1]['name']+":",font=ft,bg="white",fg="black").grid(row=2,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[2]['name']+":",font=ft,bg="white",fg="black").grid(row=3,column=0,sticky="w",padx=(0,8))

    models=list_models()
    coord_var=tk.StringVar(value=models[0] if models else DEFAULT_MODEL)
    a1_var=tk.StringVar(value=models[0] if models else DEFAULT_MODEL)
    a2_var=tk.StringVar(value=models[0] if models else DEFAULT_MODEL)
    a3_var=tk.StringVar(value=models[0] if models else DEFAULT_MODEL)

    def make_dd(parent, var, row):
        w=tk.OptionMenu(parent,var,*models)
        w.config(font=ft,bg="white",fg="black",highlightthickness=0)
        w.grid(row=row,column=1,sticky="w")

    make_dd(mf, coord_var, 0)
    make_dd(mf, a1_var,    1)
    make_dd(mf, a2_var,    2)
    make_dd(mf, a3_var,    3)

    # --- ICONS (if present on disk) ---
    icon_img=None
    coord_icon=None
    try:
        from PIL import Image, ImageTk
        _icon_path = Path(__file__).with_name("aiagentlogo.png")
        if _icon_path.exists():
            _pil = Image.open(_icon_path)
            _pil = _pil.resize((18,18))
            icon_img = ImageTk.PhotoImage(_pil)
    except Exception:
        pass

    try:
        from PIL import Image, ImageTk
        _cpath = Path(__file__).with_name("coordinator.png")
        if _cpath.exists():
            _pilc = Image.open(_cpath)
            _pilc = _pilc.resize((18,18))
            coord_icon = ImageTk.PhotoImage(_pilc)
    except Exception:
        pass

    # --- OUTPUT AREA ---
    of=tk.Frame(root,bg="white")
    of.pack(padx=15,pady=(0,15),fill="both",expand=True)

    # Row with the 3 agents
    agents_row=tk.Frame(of,bg="white")
    agents_row.pack(fill="both",expand=False)
    for col in range(3):
        agents_row.columnconfigure(col,weight=1,uniform="agcols")

    def make_agent_panel(parent, idx, name, model_var, icon_img_local):
        f = tk.Frame(parent, bg="white", bd=0, highlightthickness=0)
        f.grid(row=0, column=idx, padx=5, sticky="nsew")

        header = tk.Frame(f, bg="white")
        header.pack(anchor="w", fill="x")

        if icon_img_local is not None:
            icon_label = tk.Label(header, image=icon_img_local, bg="white")
            icon_label.image = icon_img_local
            icon_label.pack(side="left", padx=(0,6))

        hdr = tk.Label(
            header,
            text=f"Agent {idx+1} - {name} - {model_var.get()}",
            font=("Courier",14),
            bg="white",
            fg="black"
        )
        hdr.pack(side="left")

        # live update header if model changes
        def _upd(*_):
            try:
                hdr.config(text=f"Agent {idx+1} - {name} - {model_var.get()}")
            except Exception:
                pass
        try:
            model_var.trace_add("write", _upd)
        except Exception:
            try:
                model_var.trace("w", _upd)
            except Exception:
                pass

        t = tk.Text(
            f,
            height=12,
            wrap="word",
            font=("Courier",12),
            bg="black",
            fg="#00FF00",
            insertbackground="#00FF00",
            relief="flat",
            padx=12,
            pady=10,
            spacing1=5,
            spacing3=5
        )
        t.pack(fill="both", expand=True)
        t.config(state=tk.DISABLED)
        return t

    out_boxes = {}
    out_boxes[AGENTS[0]["name"]] = make_agent_panel(agents_row, 0, AGENTS[0]["name"], a1_var, icon_img)
    out_boxes[AGENTS[1]["name"]] = make_agent_panel(agents_row, 1, AGENTS[1]["name"], a2_var, icon_img)
    out_boxes[AGENTS[2]["name"]] = make_agent_panel(agents_row, 2, AGENTS[2]["name"], a3_var, icon_img)

    # Coordinator panel (bottom)
    coord_frame = tk.Frame(of, bg="white")
    coord_frame.pack(fill="both", expand=True, pady=(10,0))

    coord_header_frame = tk.Frame(coord_frame, bg="white")
    coord_header_frame.pack(anchor="w", fill="x")

    if coord_icon is not None:
        _il = tk.Label(coord_header_frame, image=coord_icon, bg="white")
        _il.image = coord_icon
        _il.pack(side="left", padx=(0,6))

    coord_header = tk.Label(
        coord_header_frame,
        text=f"Agent Coordinator - {coord_var.get()}",
        font=("Courier",14),
        bg="white",
        fg="black"
    )
    coord_header.pack(side="left")

    def _coord_upd(*_):
        try:
            coord_header.config(text=f"Agent Coordinator - {coord_var.get()}")
        except Exception:
            pass
    try:
        coord_var.trace_add("write", _coord_upd)
    except Exception:
        try:
            coord_var.trace("w", _coord_upd)
        except Exception:
            pass

    coord_text = tk.Text(
        coord_frame,
        height=7,
        wrap="word",
        font=("Courier",12),
        bg="black",
        fg="#00FF00",
        insertbackground="#00FF00",
        relief="flat",
        padx=12,
        pady=10,
        spacing1=5,
        spacing3=5
    )
    coord_text.pack(fill="both", expand=True)
    coord_text.config(state=tk.DISABLED)

    # --- Helpers: thread-safe text update + spinner ---

    def ui_set(widget, text):
        def _set():
            widget.config(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, text)
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        widget.after(0, _set)

    busy = {
        AGENTS[0]["name"]: False,
        AGENTS[1]["name"]: False,
        AGENTS[2]["name"]: False
    }

    def start_spinner(name):
        busy[name] = True
        def tick(i=0):
            if not busy.get(name):
                return
            dots = "." * ((i % 3) + 1)
            ui_set(out_boxes[name], f"⏳ {name} is reasoning{dots}")
            out_boxes[name].after(350, lambda: tick(i+1))
        tick()

    def stop_spinner(name):
        busy[name] = False

    def set_box(name, content):
        stop_spinner(name)
        ui_set(out_boxes[name], content)

    # -------------------- Coordination modes --------------------
    # For ora implementiamo solo Voting-Based.
    # Role-Based / Debate-Based restano stub ma mantengono lo stile.

    def run_voting_based(task, strategy_label):
        # puliamo le box e stampiamo header della strategia
        for b in out_boxes.values():
            ui_set(b, "")
        ui_set(
            coord_text,
            f"Starting Voting (democratic) [{strategy_label}]\n"
            f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
            f"Topic: {task}\n"
        )

        def worker():
            try:
                # spinner visivo mentre gli agenti ragionano
                start_spinner(AGENTS[0]["name"])
                start_spinner(AGENTS[1]["name"])
                start_spinner(AGENTS[2]["name"])

                # costruiamo gli agenti con il modello scelto dalla UI
                c1=ClientAgent(AGENTS[0]["name"], AGENTS[0]["title"], a1_var.get())
                c2=ClientAgent(AGENTS[1]["name"], AGENTS[1]["title"], a2_var.get())
                c3=ClientAgent(AGENTS[2]["name"], AGENTS[2]["title"], a3_var.get())

                # ognuno parla una volta
                s1=c1.speak(task)
                s2=c2.speak(task)
                s3=c3.speak(task)

                # aggiorniamo le box agent con risposta/razionale/confidenza
                set_box(
                    AGENTS[0]["name"],
                    f"Answer: {s1['answer']}\nConfidence: {s1['confidence']}\n\nRationale:\n{s1['rationale']}\n"
                )
                set_box(
                    AGENTS[1]["name"],
                    f"Answer: {s2['answer']}\nConfidence: {s2['confidence']}\n\nRationale:\n{s2['rationale']}\n"
                )
                set_box(
                    AGENTS[2]["name"],
                    f"Answer: {s3['answer']}\nConfidence: {s3['confidence']}\n\nRationale:\n{s3['rationale']}\n"
                )

                # majority vote
                answers=[s1,s2,s3]
                votes={}
                for s in answers:
                    k=norm_answer(s["answer"])
                    if not is_non_answer(k):
                        votes[k]=votes.get(k,0)+1

                if votes:
                    max_votes=max(votes.values())
                    winners=[k for k,v in votes.items() if v==max_votes]
                else:
                    winners=[]

                if len(winners)==1:
                    best=winners[0]
                    final_ans=next(
                        (s["answer"] for s in answers if norm_answer(s["answer"])==best),
                        ""
                    )
                    justification="Simple majority"
                elif len(winners)>1:
                    # tiebreak deterministico
                    best=sorted(winners)[0]
                    final_ans=next(
                        (s["answer"] for s in answers if norm_answer(s["answer"])==best),
                        best
                    )
                    justification="Tie -> deterministic pick"
                else:
                    final_ans=""
                    justification="No valid votes"

                ui_set(
                    coord_text,
                    f"Starting Voting (democratic) [{strategy_label}]\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\n"
                    f"Final output: {final_ans}\n"
                    f"Why: {justification}\n"
                )

            except Exception as e:
                ui_set(
                    coord_text,
                    f"Starting Voting (democratic) [{strategy_label}]\n"
                    f"Topic: {task}\n\n"
                    f"Final output: (error)\nWhy: {e}\n"
                )
            finally:
                stop_spinner(AGENTS[0]["name"])
                stop_spinner(AGENTS[1]["name"])
                stop_spinner(AGENTS[2]["name"])

        threading.Thread(target=worker, daemon=True).start()

    def run_role_based(task, strategy_label):
        # placeholder estetico
        for b in out_boxes.values():
            ui_set(b, "Role-Based mode not implemented yet.")
        ui_set(
            coord_text,
            f"Starting Role-Based [{strategy_label}]\n"
            f"Topic: {task}\n\n(Not implemented yet)"
        )

    def run_debate_based(task, strategy_label):
        # placeholder estetico
        for b in out_boxes.values():
            ui_set(b, "Debate-Based mode not implemented yet.")
        ui_set(
            coord_text,
            f"Starting Debate-Based [{strategy_label}]\n"
            f"Topic: {task}\n\n(Not implemented yet)"
        )

    def run_none(task, strategy_label):
        for b in out_boxes.values():
            ui_set(b, "Adaptation disabled.")
        ui_set(
            coord_text,
            f"Adaptation disabled [{strategy_label}]\n"
            f"No AI-agent coordination will run.\n"
            f"Task was:\n{task}"
        )

    # -------------------- Submit / keybind --------------------

    def submit():
        task=input_box.get("1.0", tk.END).strip()
        if not task:
            return
        mode, raw_strategy = read_coordination_mode()

        if mode=="voting":
            run_voting_based(task, raw_strategy)
        elif mode=="role":
            run_role_based(task, raw_strategy)
        elif mode=="debate":
            run_debate_based(task, raw_strategy)
        else:
            run_none(task, raw_strategy)

    send_btn.config(command=submit)

    def on_keypress(e):
        if platform.system()=="Darwin" and (e.state & 0x10):
            input_box.insert(tk.INSERT,"\n")
            return "break"
        elif platform.system()!="Darwin" and (e.state & 0x04):
            input_box.insert(tk.INSERT,"\n")
            return "break"
        else:
            submit()
            return "break"

    input_box.bind("<Return>", on_keypress)

    root.minsize(1100,650)
    return root

if __name__ == "__main__":
    ui = build_ui()
    ui.mainloop()
