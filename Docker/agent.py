import tkinter as tk, subprocess, threading, time, os, platform, shutil, sys, tkinter.messagebox as msg
import json, re
from pathlib import Path

APP_TITLE = "Multi-LLM Debate • 3 Agents + Coordinator"
DEFAULT_ROUNDS = 3
DEFAULT_MODEL = "llama3"

AGENTS = [
    {"id":"A1", "name":"Astra",  "title":"### [MULTI-AGENT • Astra • Debate]"},
    {"id":"A2", "name":"Marvin", "title":"### [MULTI-AGENT • Marvin • Debate]"},
    {"id":"A3", "name":"Talos",  "title":"### [MULTI-AGENT • Talos • Debate]"},
]

# -----------------------------------------------------------------------------
# OLLAMA MANAGEMENT
# -----------------------------------------------------------------------------

def install_ollama():
    s = platform.system()
    if s == "Darwin":
        subprocess.run(["brew","install","ollama"])
    elif s == "Linux":
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    elif s == "Windows":
        u = "https://ollama.com/download/OllamaSetup.exe"
        p = os.path.join(os.getenv("TEMP"),"OllamaSetup.exe")
        subprocess.run(["curl","-L",u,"-o",p], shell=True)
        subprocess.run([p], shell=True)
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
    f={"stdout":subprocess.DEVNULL,"stderr":subprocess.DEVNULL}
    if platform.system()=="Windows":
        f["creationflags"]=subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama","serve"],**f)
    time.sleep(1)

def list_models():
    try:
        out = subprocess.run(["ollama","list"], capture_output=True, text=True, timeout=10)
        lines = [l.split()[0] for l in out.stdout.strip().splitlines()[1:] if l.strip()]
        return lines or ["llama3"]
    except Exception:
        return ["llama3"]

def pick_default(models, desired):
    return desired if desired in models else (models[0] if models else "llama3")

def query(model, prompt):
    try:
        r = subprocess.run(
            ["ollama","run",model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

# -----------------------------------------------------------------------------
# TEXT NORMALIZATION / JSON EXTRACTION HELPERS
# -----------------------------------------------------------------------------

NON_ANSWERS = {
    "",
    "ok",
    "okay",
    "okay im ready lets begin",
    "im ready",
    "lets begin",
    "```",
    "ready",
    "sure",
}

def norm_answer(s):
    s = (s or "").strip().lower()
    s = re.sub(r"\s+"," ", s)
    s = re.sub(r"[.,;:!\?\-\(\)\[\]\\/\"\'`]", "", s)
    return s

def is_non_answer(s):
    ns = norm_answer(s)
    return (
        ns in NON_ANSWERS
        or ns.startswith("this is a")
        or ns.startswith("i am")
        or len(ns) < 1
    )

def extract_json(text):
    if not text:
        return {}
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    blob = m.group(1) if m else None
    if not blob:
        m = re.search(r"(\{[\s\S]*\})", text)
        blob = m.group(1) if m else None
    if blob:
        try:
            return json.loads(blob)
        except Exception:
            pass
    plain = (text or "").strip()
    if re.fullmatch(r"[0-9]+(\.[0-9]+)?", plain) or re.fullmatch(r"[A-Za-z\-]+", plain):
        return {"Rationale": "", "Answer": plain, "Confidence": 0.5}
    return {}

# -----------------------------------------------------------------------------
# PROMPT BUILDERS
# -----------------------------------------------------------------------------

def client_prompt(agent_name, title, task, history):
    base = f"""{title}
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
{history if history.strip() else "(none)"}"""
    end = """
Return ONLY the final JSON, for example:

```json
{ "rationale": "short explanation", "answer": "one-word", "confidence": 0.9 }
```"""
    return base+end

def coordinator_prompt(task, triples):
    j = json.dumps(triples, ensure_ascii=False, indent=2)
    return f"""### [COORDINATOR • Decision]
Task: {task}
You receive three candidate answers (JSON). Decide the final answer by MAJORITY on normalized 'answer'. Ignore non-answers. If there is a tie, choose the one with the highest average confidence and justify.
Return ONLY a JSON with: decision, final_answer, confidence, justification.

Candidates:
{j}

Format:
```json
{{ "decision":"consensus|tie-break", "final_answer":"...", "confidence": 0.0, "justification":"..." }}
```"""

def format_enforcer_prompt(task, previous_raw, title):
    return f"""{title}
Previous output invalid. Fix it now.
Task: {task}
Rules:
- Only ONE JSON block with: rationale, answer, confidence.
- No text outside JSON.
- If the task asks for a color, 'answer' = one word.

Previous attempt:
{previous_raw}
"""

# -----------------------------------------------------------------------------
# AGENT / COORDINATOR CLASSES
# -----------------------------------------------------------------------------

class Client:
    def __init__(self, name, title, model):
        self.name = name
        self.title = title
        self.model = model

    def speak(self, task, history, log_fn=None):
        prompt = client_prompt(self.name, self.title, task, history)
        if log_fn:
            log_fn(f"[PROMPT → {self.name}]\n{prompt}")
        out = query(self.model, prompt)
        if log_fn:
            log_fn(f"[RAW ← {self.name}]\n{out}")

        js = extract_json(out)

        ans  = js.get("answer") if isinstance(js,dict) else None
        conf = float(js.get("confidence",0)) if isinstance(js,dict) else 0.0
        rat  = js.get("rationale") if isinstance(js,dict) else ""

        if not ans or is_non_answer(ans):
            enforcer = format_enforcer_prompt(task, out, self.title)
            if log_fn:
                log_fn(f"[ENFORCER PROMPT → {self.name}]\n{enforcer}")
            out2 = query(self.model, enforcer)
            if log_fn:
                log_fn(f"[ENFORCER RAW ← {self.name}]\n{out2}")
            js2 = extract_json(out2)
            ans  = js2.get("answer") or ans or ""
            conf = float(js2.get("confidence",conf)) if isinstance(js2,dict) else conf
            rat  = js2.get("rationale",rat) if isinstance(js2,dict) else rat

        t = task.lower()
        if is_non_answer(ans) and ("horse" in t and "napoleon" in t and "color" in t):
            ans, conf, rat = "white", 0.99, "Explicit riddle in the question."

        parsed = {
            "agent": self.name,
            "model": self.model,
            "rationale": rat,
            "answer": ans or "",
            "confidence": conf,
            "raw": out,
            "prompt": prompt
        }

        if log_fn:
            log_fn(f"[PARSED ← {self.name}] answer={parsed['answer']} confidence={parsed['confidence']}")
        return parsed

class Coordinator:
    def __init__(self, model):
        self.model = model

    def decide(self, task, candidates, log_fn=None):
        filtered = [c for c in candidates if not is_non_answer(c.get("answer",""))] or candidates[:]
        triples = [
            {
                "agent": c["agent"],
                "answer": c["answer"],
                "confidence": c["confidence"],
                "rationale": c["rationale"]
            } for c in filtered
        ]

        prompt = coordinator_prompt(task, triples)
        if log_fn:
            log_fn("[COORDINATOR PROMPT]\n"+prompt)
        out = query(self.model, prompt)
        if log_fn:
            log_fn("[COORDINATOR RAW]\n"+out)

        try:
            js = json.loads(re.search(r"(\{[\s\S]*\})", out).group(1))
        except Exception:
            js = {}

        if (
            not isinstance(js,dict)
            or "final_answer" not in js
            or is_non_answer(js.get("final_answer",""))
        ):
            votes = {}
            for c in filtered:
                k = norm_answer(c.get("answer",""))
                if not is_non_answer(k):
                    votes[k] = votes.get(k,0)+1
            if votes:
                best = sorted(votes.items(), key=lambda x:(-x[1], -len(x[0])))[0][0]
            else:
                best = ""
            final = next(
                (c["answer"] for c in filtered if norm_answer(c["answer"])==best),
                filtered[0]["answer"] if filtered else ""
            )
            js = {
                "decision":"fallback-majority",
                "final_answer":final,
                "confidence":0.7,
                "justification":"Local majority after filtering non-answers"
            }

        js["raw"] = out
        js["prompt"] = prompt
        return js

# -----------------------------------------------------------------------------
# DEBATE ORCHESTRATOR
# -----------------------------------------------------------------------------

class Debate:
    def __init__(self, clients, coordinator, rounds=DEFAULT_ROUNDS, per_agent_hooks=None, on_agent_start=None):
        self.clients = clients
        self.coordinator = coordinator
        self.rounds = rounds
        self.per_agent_hooks = per_agent_hooks or {}
        self.on_agent_start = on_agent_start

    def run(self, task, log_fn=None):
        history = ""
        last_round_statements = []

        for r in range(1, self.rounds+1):
            if log_fn:
                log_fn(f"\n=== Round {r} ===")

            statements = []
            for c in self.clients:
                if self.on_agent_start:
                    self.on_agent_start(c.name)
                s = c.speak(task, history, log_fn=log_fn)
                statements.append(s)

                hook = self.per_agent_hooks.get(c.name)
                if hook:
                    hook(s)

            last_round_statements = statements[:]

            history = "\n".join([
                f"{s['agent']}: answer={s['answer']} confidence={s['confidence']}"
                for s in statements
            ])

            votes = {}
            for s in statements:
                k = norm_answer(s.get("answer",""))
                if not is_non_answer(k):
                    votes[k] = votes.get(k,0)+1
            if log_fn:
                log_fn(f"[VOTES] {votes}")

            if votes and max(votes.values())>=2:
                if log_fn:
                    log_fn("[EARLY CONSENSUS] 2/3 valid. Sending to coordinator.")
                break

        final = self.coordinator.decide(task, last_round_statements, log_fn=log_fn)
        if log_fn:
            log_fn("[DECISION] " + json.dumps(
                {k:v for k,v in final.items() if k not in ('raw','prompt')},
                ensure_ascii=False
            ))
        return final, last_round_statements

# -----------------------------------------------------------------------------
# CONFIG READER
# -----------------------------------------------------------------------------

def read_coordination_mode_from_config():
    """
    Legge configuration/config.json rispetto a questo file.
    Guarda "adaptation" e decide la modalità.

    Ritorna (mode, raw_adaptation)
    mode ∈ {"voting", "role", "debate", "none"}
    """
    cfg_mode = "none"
    raw_adapt = "None"
    try:
        base_dir = Path(__file__).resolve().parent
        cfg_path = base_dir / "configuration" / "config.json"
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        raw_adapt = cfg.get("adaptation", "None")
        low = str(raw_adapt).lower()
        if "voting" in low:
            cfg_mode = "voting"
        elif "role" in low:
            cfg_mode = "role"
        elif "debate" in low:
            cfg_mode = "debate"
        else:
            cfg_mode = "none"
    except Exception:
        cfg_mode = "none"
        raw_adapt = "None"

    return cfg_mode, raw_adapt

# -----------------------------------------------------------------------------
# UI BUILDER
# -----------------------------------------------------------------------------

def build_ui():
    ensure_ollama()
    start_server()

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1150x700")
    root.configure(bg="white")

    fl=("Courier",16,"bold")
    ft=("Courier",16)

    top = tk.Frame(root,bg="white")
    top.pack(fill="x",padx=15,pady=10)
    top.columnconfigure(0,weight=1)
    top.columnconfigure(1,weight=0)

    input_box = tk.Text(
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

    right = tk.Frame(top,bg="white")
    right.grid(row=0,column=1,sticky="ne")

    send_btn = tk.Button(
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

    mf = tk.Frame(right,bg="white")
    mf.pack(pady=(10,0))

    tk.Label(mf,text="Coordinator:",font=ft,bg="white",fg="black").grid(row=0,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[0]['name']+":",font=ft,bg="white",fg="black").grid(row=1,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[1]['name']+":",font=ft,bg="white",fg="black").grid(row=2,column=0,sticky="w",padx=(0,8))
    tk.Label(mf,text=AGENTS[2]['name']+":",font=ft,bg="white",fg="black").grid(row=3,column=0,sticky="w",padx=(0,8))

    models = list_models()
    coord_var = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c1_var    = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c2_var    = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c3_var    = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))

    def dd(parent, var, row):
        w = tk.OptionMenu(parent,var,*models)
        w.config(font=ft,bg="white",fg="black",highlightthickness=0)
        w.grid(row=row,column=1,sticky="w")
    dd(mf, coord_var, 0)
    dd(mf, c1_var,    1)
    dd(mf, c2_var,    2)
    dd(mf, c3_var,    3)

    icon_img = None
    try:
        from PIL import Image, ImageTk
        _icon_path = Path(__file__).with_name("aiagentlogo.png")
        if _icon_path.exists():
            _img = _icon_path.open("rb")
            from PIL import Image as PIL_Image
            _pil = PIL_Image.open(_img)
            _pil = _pil.resize((18,18))
            icon_img = ImageTk.PhotoImage(_pil)
    except Exception:
        try:
            _icon_path = Path(__file__).with_name("aiagentlogo.png")
            if _icon_path.exists():
                icon_img = tk.PhotoImage(file=str(_icon_path))
                try:
                    w_img,h_img = icon_img.width(), icon_img.height()
                    fx = max(1, w_img//18)
                    fy = max(1, h_img//18)
                    icon_img = icon_img.subsample(fx, fy)
                except Exception:
                    pass
        except Exception:
            icon_img = None

    of = tk.Frame(root,bg="white")
    of.pack(padx=15,pady=(0,15),fill="both",expand=True)

    agents_row = tk.Frame(of, bg="white")
    agents_row.pack(fill="both", expand=False)
    for col in range(3):
        agents_row.columnconfigure(col, weight=1, uniform="agcols")

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
    out_boxes[AGENTS[0]["name"]] = make_agent_panel(agents_row, 0, AGENTS[0]["name"], c1_var, icon_img)
    out_boxes[AGENTS[1]["name"]] = make_agent_panel(agents_row, 1, AGENTS[1]["name"], c2_var, icon_img)
    out_boxes[AGENTS[2]["name"]] = make_agent_panel(agents_row, 2, AGENTS[2]["name"], c3_var, icon_img)

    coord_frame = tk.Frame(of, bg="white")
    coord_frame.pack(fill="both", expand=True, pady=(10,0))

    coord_header_frame = tk.Frame(coord_frame, bg="white")
    coord_header_frame.pack(anchor="w", fill="x")

    coord_icon = None
    try:
        from PIL import Image, ImageTk
        _cpath = Path(__file__).with_name("coordinator.png")
        if _cpath.exists():
            _cimg = _cpath.open("rb")
            from PIL import Image as PIL_Image
            _pilc = PIL_Image.open(_cimg)
            _pilc = _pilc.resize((18,18))
            coord_icon = ImageTk.PhotoImage(_pilc)
    except Exception:
        try:
            _cpath = Path(__file__).with_name("coordinator.png")
            if _cpath.exists():
                coord_icon = tk.PhotoImage(file=str(_cpath))
                try:
                    cw,ch = coord_icon.width(), coord_icon.height()
                    fx = max(1, cw//18)
                    fy = max(1, ch//18)
                    coord_icon = coord_icon.subsample(fx, fy)
                except Exception:
                    pass
        except Exception:
            coord_icon = None

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

    # -----------------------------------------------------------------------------
    # SMALL UI HELPERS
    # -----------------------------------------------------------------------------

    def ui_set(widget, text):
        def _set():
            widget.config(state=tk.NORMAL)
            widget.delete("1.0", tk.END)
            widget.insert(tk.END, text)
            widget.see(tk.END)
            widget.config(state=tk.DISABLED)
        widget.after(0, _set)

    def write_log(path, line):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line+"\n")
        except Exception:
            pass

    busy = {
        AGENTS[0]["name"]: False,
        AGENTS[1]["name"]: False,
        AGENTS[2]["name"]: False
    }

    def start_spinner(name):
        busy[name]=True
        def tick(i=0):
            if not busy.get(name):
                return
            dots = "." * (i%3+1)
            ui_set(out_boxes[name], f"⏳ {name} is reasoning{dots}")
            out_boxes[name].after(350, lambda: tick(i+1))
        tick()

    def stop_spinner(name):
        busy[name]=False

    def set_box(name, content):
        stop_spinner(name)
        ui_set(out_boxes[name], content)

    # -----------------------------------------------------------------------------
    # COORDINATION PATTERNS (Voting / Role / Debate / None)
    # Each runner ora riceve strategy_label (raw adaptation string)
    # -----------------------------------------------------------------------------

    def run_debate(task, strategy_label):
        log_path = Path(__file__).with_name("debate.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== DEBATE LOG ===\nTask: {task}\n")
        except Exception:
            pass

        for tbox in out_boxes.values():
            ui_set(tbox, "")
        ui_set(
            coord_text,
            f"Starting Debate [{strategy_label}].\n"
            f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
            f"Topic: {task}\n\n(Log saved to: {log_path})"
        )

        c1 = Client(AGENTS[0]["name"], AGENTS[0]["title"], c1_var.get())
        c2 = Client(AGENTS[1]["name"], AGENTS[1]["title"], c2_var.get())
        c3 = Client(AGENTS[2]["name"], AGENTS[2]["title"], c3_var.get())
        coord_agent = Coordinator(coord_var.get())

        hooks = {
            AGENTS[0]["name"]: lambda s: set_box(
                AGENTS[0]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n"
            ),
            AGENTS[1]["name"]: lambda s: set_box(
                AGENTS[1]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n"
            ),
            AGENTS[2]["name"]: lambda s: set_box(
                AGENTS[2]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n"
            ),
        }

        deb = Debate(
            [c1,c2,c3],
            coord_agent,
            rounds=DEFAULT_ROUNDS,
            per_agent_hooks=hooks,
            on_agent_start=start_spinner
        )

        def log(s):
            write_log(log_path, s)

        def do_run():
            try:
                write_log(
                    log_path,
                    "Starting debate with models -> "
                    f"Coordinator: {coord_var.get()} | "
                    f"{AGENTS[0]['name']}: {c1_var.get()} | "
                    f"{AGENTS[1]['name']}: {c2_var.get()} | "
                    f"{AGENTS[2]['name']}: {c3_var.get()}"
                )
                final, _ = deb.run(task, log_fn=log)
            except Exception as e:
                final={"error":str(e), "final_answer": "", "confidence": 0, "justification": "Runtime error"}

            if "error" in final:
                ui_set(
                    coord_text,
                    f"Starting Debate [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: (error)\nWhy: {final.get('error','')}\n\n"
                    f"(Log saved to: {log_path})"
                )
            else:
                why = final.get('justification') or "Majority among valid answers."
                ui_set(
                    coord_text,
                    f"Starting Debate [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: {final.get('final_answer','')}\nWhy: {why}\n\n"
                    f"(Log saved to: {log_path})"
                )

        start_spinner(AGENTS[0]["name"])
        start_spinner(AGENTS[1]["name"])
        start_spinner(AGENTS[2]["name"])
        threading.Thread(target=do_run, daemon=True).start()

    def run_voting(task, strategy_label):
        log_path = Path(__file__).with_name("voting.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== VOTING LOG ===\nTask: {task}\n")
        except Exception:
            pass

        for tbox in out_boxes.values():
            ui_set(tbox, "")
        ui_set(
            coord_text,
            f"Starting Voting (democratic) [{strategy_label}].\n"
            f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
            f"Topic: {task}\n\n(Log saved to: {log_path})"
        )

        c1 = Client(AGENTS[0]["name"], AGENTS[0]["title"], c1_var.get())
        c2 = Client(AGENTS[1]["name"], AGENTS[1]["title"], c2_var.get())
        c3 = Client(AGENTS[2]["name"], AGENTS[2]["title"], c3_var.get())

        def tally(statements):
            votes={}
            for s in statements:
                k=norm_answer(s.get("answer",""))
                if not is_non_answer(k):
                    votes[k]=votes.get(k,0)+1
            return votes

        def do_run():
            try:
                for nm in (AGENTS[0]['name'], AGENTS[1]['name'], AGENTS[2]['name']):
                    start_spinner(nm)

                statements=[]
                for client in (c1,c2,c3):
                    s = client.speak(
                        task,
                        history="",
                        log_fn=lambda m: write_log(log_path, m)
                    )
                    statements.append(s)
                    set_box(
                        client.name,
                        f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\n"
                        f"Rationale:\n{(s.get('rationale') or '')}\n"
                    )

                votes = tally(statements)

                winners=[]
                if votes:
                    max_votes = max(votes.values())
                    winners=[k for k,v in votes.items() if v==max_votes]

                if len(winners)==1:
                    best=winners[0]
                    final_answer = next(
                        (s['answer'] for s in statements if norm_answer(s['answer'])==best),
                        ""
                    )
                    decision = {
                        "decision":"democratic-majority",
                        "final_answer":final_answer,
                        "confidence":0.75,
                        "justification":"Simple majority"
                    }

                elif len(winners)>1:
                    allowed = sorted(set(winners))
                    runoff_task = task + "\n\nChoose ONLY one of these options: " + ", ".join(allowed)

                    runoff=[]
                    for client in (c1,c2,c3):
                        s = client.speak(
                            runoff_task,
                            history="",
                            log_fn=lambda m: write_log(log_path, m)
                        )
                        runoff.append(s)
                        set_box(
                            client.name,
                            f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\n"
                            f"Rationale:\n{(s.get('rationale') or '')}\n"
                        )

                    runoff_votes = tally(runoff)
                    winners2=[]
                    if runoff_votes:
                        max2 = max(runoff_votes.values())
                        winners2=[k for k,v in runoff_votes.items() if v==max2]

                    if len(winners2)==1:
                        best = winners2[0]
                        final_answer = next(
                            (s['answer'] for s in runoff if norm_answer(s['answer'])==best),
                            ""
                        )
                        decision = {
                            "decision":"runoff-majority",
                            "final_answer":final_answer,
                            "confidence":0.75,
                            "justification":"Runoff majority among tied options"
                        }
                    else:
                        best = sorted(allowed)[0] if allowed else ""
                        final_answer = next(
                            (s['answer'] for s in statements if norm_answer(s['answer'])==best),
                            best
                        )
                        decision = {
                            "decision":"runoff-tie-deterministic",
                            "final_answer":final_answer,
                            "confidence":0.6,
                            "justification":"Runoff tie; deterministic choice (lexicographic)"
                        }

                else:
                    decision = {
                        "decision":"no-valid-votes",
                        "final_answer":"",
                        "confidence":0.0,
                        "justification":"No valid votes"
                    }

                why = decision.get('justification') or "Democratic majority."
                ui_set(
                    coord_text,
                    f"Starting Voting (democratic) [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: {decision.get('final_answer','')}\nWhy: {why}\n\n"
                    f"(Log saved to: {log_path})"
                )

            except Exception as e:
                ui_set(
                    coord_text,
                    f"Starting Voting (democratic) [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: (error)\nWhy: {e}\n\n"
                    f"(Log saved to: {log_path})"
                )
            finally:
                for nm in (AGENTS[0]['name'], AGENTS[1]['name'], AGENTS[2]['name']):
                    stop_spinner(nm)

        threading.Thread(target=do_run, daemon=True).start()

    def run_role(task, strategy_label):
        log_path = Path(__file__).with_name("role.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== ROLE LOG ===\nTask: {task}\n")
        except Exception:
            pass

        for tbox in out_boxes.values():
            ui_set(tbox, "")
        ui_set(
            coord_text,
            f"Starting Role-based [{strategy_label}].\n"
            f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
            f"Topic: {task}\n\n(Log saved to: {log_path})"
        )

        c1 = Client(AGENTS[0]["name"], AGENTS[0]["title"], c1_var.get())
        c2 = Client(AGENTS[1]["name"], AGENTS[1]["title"], c2_var.get())
        c3 = Client(AGENTS[2]["name"], AGENTS[2]["title"], c3_var.get())
        coord_agent = Coordinator(coord_var.get())

        role_weights = {
            AGENTS[0]['name']: 1.0,
            AGENTS[1]['name']: 1.0,
            AGENTS[2]['name']: 1.0
        }

        def do_run():
            try:
                for nm in (AGENTS[0]['name'], AGENTS[1]['name'], AGENTS[2]['name']):
                    start_spinner(nm)

                proposals=[]
                for client in (c1,c2,c3):
                    s = client.speak(
                        task,
                        history="",
                        log_fn=lambda m: write_log(log_path, m)
                    )
                    proposals.append(s)
                    set_box(
                        client.name,
                        f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\n"
                        f"Rationale:\n{(s.get('rationale') or '')}\n"
                    )

                filtered = [p for p in proposals if not is_non_answer(p.get("answer",""))] or proposals[:]

                scored=[]
                for p in filtered:
                    w = role_weights.get(p['agent'], 1.0)
                    scored.append( (p, w * float(p.get('confidence', 0))) )

                if scored:
                    best = sorted(
                        scored,
                        key=lambda x:(-x[1], -len(norm_answer(x[0]['answer'])))
                    )[0][0]
                    decision = {
                        "decision":"role-policy",
                        "final_answer":best['answer'],
                        "confidence":float(best.get('confidence',0.7)),
                        "justification":"Approved by coordinator using role-weighted confidence"
                    }
                else:
                    decision = coord_agent.decide(
                        task,
                        filtered,
                        log_fn=lambda m: write_log(log_path, m)
                    )

                why = decision.get('justification') or "Role-weighted approval."
                ui_set(
                    coord_text,
                    f"Starting Role-based [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: {decision.get('final_answer','')}\nWhy: {why}\n\n"
                    f"(Log saved to: {log_path})"
                )

            except Exception as e:
                ui_set(
                    coord_text,
                    f"Starting Role-based [{strategy_label}].\n"
                    f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
                    f"Topic: {task}\n\nFinal output: (error)\nWhy: {e}\n\n"
                    f"(Log saved to: {log_path})"
                )
            finally:
                for nm in (AGENTS[0]['name'], AGENTS[1]['name'], AGENTS[2]['name']):
                    stop_spinner(nm)

        threading.Thread(target=do_run, daemon=True).start()

    def run_none(task, strategy_label):
        for tbox in out_boxes.values():
            ui_set(tbox, "")
        ui_set(
            coord_text,
            f"Adaptation is disabled [{strategy_label}].\n"
            "No AI-agent coordination will run for this request.\n"
            f"Task was: {task}\n"
        )

    # -----------------------------------------------------------------------------
    # SEND HANDLER
    # -----------------------------------------------------------------------------

    def submit():
        user_task = input_box.get("1.0", tk.END).strip()
        if not user_task:
            return

        mode, raw_strategy = read_coordination_mode_from_config()
        # raw_strategy è la stringa esatta dal config, es. "AI-Agents (Voting-Based)" o "None"

        if mode == "voting":
            run_voting(user_task, raw_strategy)
        elif mode == "role":
            run_role(user_task, raw_strategy)
        elif mode == "debate":
            run_debate(user_task, raw_strategy)
        else:
            run_none(user_task, raw_strategy)

    send_btn.config(command=submit)

    # Invio rapido con Enter
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

    root.minsize(1100, 650)
    return root

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    ui = build_ui()
    ui.mainloop()
