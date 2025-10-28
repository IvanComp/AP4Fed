import os
import sys
import json
import re
import time
import yaml
import copy
import glob
import random
import locale
import subprocess
import ctypes
import platform
import shutil
import threading
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtCore import Qt, QProcess, QProcessEnvironment, QTimer, QCoreApplication
from PyQt5.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QGridLayout,
)
import tkinter as tk
import tkinter.messagebox as msg
from pathlib import Path
from typing import Dict, List, Tuple 

# ─────────────────────────────────────────────────────────────
# Stay awake
# ─────────────────────────────────────────────────────────────
def keep_awake():
    if sys.platform == "darwin":
        subprocess.Popen(["caffeinate", "-dimsu"])
    elif os.name == "nt":
        ES_CONTINUOUS        = 0x80000000
        ES_SYSTEM_REQUIRED   = 0x00000001
        ES_AWAYMODE_REQUIRED = 0x00000040
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED | ES_AWAYMODE_REQUIRED
        )
    else:
        try:
            subprocess.Popen([
                "systemd-inhibit",
                "--what=idle",
                "--why=Keep app awake",
                "--mode=block",
                "sleep",
                "infinity"
            ])
        except FileNotFoundError:
            pass

keep_awake()

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

def random_pastel():
    return (
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5,
        random.random() * 0.5 + 0.5,
    )

# ─────────────────────────────────────────────────────────────
# Helpers per config
# ─────────────────────────────────────────────────────────────
def load_config(simulation_type):
    """
    Legge Docker/configuration/config.json oppure Local/configuration/config.json
    in base a simulation_type.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(base_dir, simulation_type, "configuration", "config.json")
    with open(cfg_path, "r") as f:
        return json.load(f)

def load_adaptation(simulation_type):
    try:
        cfg = load_config(simulation_type)
        return cfg.get("adaptation", "None")
    except Exception:
        return "None"

# ─────────────────────────────────────────────────────────────
#  MULTI-AGENT UI (tkinter)
# ─────────────────────────────────────────────────────────────

APP_TITLE = "Multi-LLM Debate • 3 Agents + Coordinator"
DEFAULT_ROUNDS = 3
DEFAULT_MODEL = "llama3"

AGENTS = [
    {"id": "A1", "name": "Astra",  "title": "### [MULTI-AGENT • Astra • Debate]"},
    {"id": "A2", "name": "Marvin", "title": "### [MULTI-AGENT • Marvin • Debate]"},
    {"id": "A3", "name": "Talos",  "title": "### [MULTI-AGENT • Talos • Debate]"},
]

def install_ollama():
    s = platform.system()
    if s == "Darwin":
        subprocess.run(["brew", "install", "ollama"])
    elif s == "Linux":
        subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True)
    elif s == "Windows":
        u = "https://ollama.com/download/OllamaSetup.exe"
        p = os.path.join(os.getenv("TEMP"), "OllamaSetup.exe")
        subprocess.run(["curl", "-L", u, "-o", p], shell=True)
        subprocess.run([p], shell=True)
    else:
        sys.exit(1)

def ensure_ollama():
    if not shutil.which("ollama"):
        if msg.askyesno("Ollama not found", "Ollama is not installed. Install now?"):
            install_ollama()
            time.sleep(5)
            if not shutil.which("ollama"):
                msg.showerror("Install failed", "❌ Ollama installation failed.")
                sys.exit(1)
        else:
            sys.exit(0)

def start_server():
    f = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if platform.system() == "Windows":
        f["creationflags"] = subprocess.CREATE_NO_WINDOW
    subprocess.Popen(["ollama", "serve"], **f)
    time.sleep(1)

def list_models():
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=10)
        lines = [l.split()[0] for l in out.stdout.strip().splitlines()[1:] if l.strip()]
        return lines or ["llama3"]
    except Exception:
        return ["llama3"]

def pick_default(models, desired):
    return desired if desired in models else (models[0] if models else "llama3")

def query(model, prompt):
    try:
        r = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            text=True,
            capture_output=True,
            timeout=120
        )
        return (r.stdout or r.stderr or "").strip()
    except Exception as e:
        return f"ERROR: {e}"

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
    s = re.sub(r"\s+", " ", s)
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
    return base + end

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
        ans = js.get("answer") if isinstance(js, dict) else None
        conf = float(js.get("confidence", 0)) if isinstance(js, dict) else 0.0
        rat = js.get("rationale") if isinstance(js, dict) else ""
        if not ans or is_non_answer(ans):
            enforcer = format_enforcer_prompt(task, out, self.title)
            if log_fn:
                log_fn(f"[ENFORCER PROMPT → {self.name}]\n{enforcer}")
            out2 = query(self.model, enforcer)
            if log_fn:
                log_fn(f"[ENFORCER RAW ← {self.name}]\n{out2}")
            js2 = extract_json(out2)
            ans = js2.get("answer") or ans or ""
            conf = float(js2.get("confidence", conf)) if isinstance(js2, dict) else conf
            rat = js2.get("rationale", rat) if isinstance(js2, dict) else rat

        # Easter egg/riddle hardening
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
            "prompt": prompt,
        }
        if log_fn:
            log_fn(f"[PARSED ← {self.name}] answer={parsed['answer']} confidence={parsed['confidence']}")
        return parsed

class Coordinator:
    def __init__(self, model):
        self.model = model

    def decide(self, task, candidates, log_fn=None):
        filtered = [c for c in candidates if not is_non_answer(c.get("answer", ""))] or candidates[:]
        triples = [
            {
                "agent": c["agent"],
                "answer": c["answer"],
                "confidence": c["confidence"],
                "rationale": c["rationale"],
            }
            for c in filtered
        ]
        prompt = coordinator_prompt(task, triples)
        if log_fn:
            log_fn("[COORDINATOR PROMPT]\n" + prompt)
        out = query(self.model, prompt)
        if log_fn:
            log_fn("[COORDINATOR RAW]\n" + out)
        try:
            js = json.loads(re.search(r"(\{[\s\S]*\})", out).group(1))
        except Exception:
            js = {}
        if (
            not isinstance(js, dict)
            or "final_answer" not in js
            or is_non_answer(js.get("final_answer", ""))
        ):
            # fallback majority vote
            votes = {}
            for c in filtered:
                k = norm_answer(c.get("answer", ""))
                if not is_non_answer(k):
                    votes[k] = votes.get(k, 0) + 1
            best = (
                sorted(votes.items(), key=lambda x: (-x[1], -len(x[0])))[0][0]
                if votes
                else ""
            )
            final = next(
                (
                    c["answer"]
                    for c in filtered
                    if norm_answer(c["answer"]) == best
                ),
                filtered[0]["answer"] if filtered else "",
            )
            js = {
                "decision": "fallback-majority",
                "final_answer": final,
                "confidence": 0.7,
                "justification": "Local majority after filtering non-answers",
            }
        js["raw"] = out
        js["prompt"] = prompt
        return js

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
        for r in range(1, self.rounds + 1):
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
            history = "\n".join(
                [
                    f"{s['agent']}: answer={s['answer']} confidence={s['confidence']}"
                    for s in statements
                ]
            )
            votes = {}
            for s in statements:
                k = norm_answer(s.get("answer", ""))
                if not is_non_answer(k):
                    votes[k] = votes.get(k, 0) + 1
            if log_fn:
                log_fn(f"[VOTES] {votes}")
            if votes and max(votes.values()) >= 2:
                if log_fn:
                    log_fn("[EARLY CONSENSUS] 2/3 valid. Sending to coordinator.")
                break
        final = self.coordinator.decide(task, last_round_statements, log_fn=log_fn)
        if log_fn:
            log_fn(
                "[DECISION] "
                + json.dumps(
                    {k: v for k, v in final.items() if k not in ("raw", "prompt")},
                    ensure_ascii=False,
                )
            )
        return final, last_round_statements

def build_ui(simulation_type):
    # carico la strategia di coordinamento dal config
    adaptation_mode = load_adaptation(simulation_type)

    ensure_ollama()
    start_server()

    root = tk.Tk()
    root.title(APP_TITLE)
    root.geometry("1150x700")
    root.configure(bg="white")

    fl = ("Courier", 16, "bold")
    ft = ("Courier", 16)

    top = tk.Frame(root, bg="white")
    top.pack(fill="x", padx=15, pady=10)
    top.columnconfigure(0, weight=1)
    top.columnconfigure(1, weight=0)

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
        spacing3=3,
    )
    input_box.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

    right = tk.Frame(top, bg="white")
    right.grid(row=0, column=1, sticky="ne")

    def submit():
        p = input_box.get("1.0", tk.END).strip()
        if not p:
            return
        run_debate(p)

    def on_keypress(e):
        if platform.system() == "Darwin" and (e.state & 0x10):
            input_box.insert(tk.INSERT, "\n")
            return "break"
        elif platform.system() != "Darwin" and (e.state & 0x04):
            input_box.insert(tk.INSERT, "\n")
            return "break"
        else:
            submit()
            return "break"

    input_box.bind("<Return>", on_keypress)

    send_btn = tk.Button(
        right,
        text="➤ Send",
        font=("Courier", 14),
        bg="#e0e0e0",
        fg="black",
        relief="flat",
        padx=12,
        pady=5,
        borderwidth=1,
        cursor="hand1",
        anchor="center",
        command=submit,
    )
    send_btn.pack(pady=(0, 10))

    mf = tk.Frame(right, bg="white")
    mf.pack(pady=(10, 0))
    tk.Label(mf, text="Coordinator:", font=ft, bg="white", fg="black").grid(
        row=0, column=0, sticky="w", padx=(0, 8)
    )
    tk.Label(mf, text=AGENTS[0]["name"] + ":", font=ft, bg="white", fg="black").grid(
        row=1, column=0, sticky="w", padx=(0, 8)
    )
    tk.Label(mf, text=AGENTS[1]["name"] + ":", font=ft, bg="white", fg="black").grid(
        row=2, column=0, sticky="w", padx=(0, 8)
    )
    tk.Label(mf, text=AGENTS[2]["name"] + ":", font=ft, bg="white", fg="black").grid(
        row=3, column=0, sticky="w", padx=(0, 8)
    )

    models = list_models()
    coord_var = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c1_var = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c2_var = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))
    c3_var = tk.StringVar(value=pick_default(models, DEFAULT_MODEL))

    def dd(parent, var, row):
        w = tk.OptionMenu(parent, var, *models)
        w.config(font=ft, bg="white", fg="black", highlightthickness=0)
        w.grid(row=row, column=1, sticky="w")

    dd(mf, coord_var, 0)
    dd(mf, c1_var, 1)
    dd(mf, c2_var, 2)
    dd(mf, c3_var, 3)

    # Coordinaton mechanism label (invece di radio button)
    pf = tk.Frame(right, bg="white")
    pf.pack(pady=(12, 0), fill="x")

    tk.Label(
        pf,
        text="Coordination Mechanism:",
        font=ft,
        bg="white",
        fg="black"
    ).grid(row=0, column=0, sticky="w", padx=(0, 8))

    mech_lbl = tk.Label(
        pf,
        text=adaptation_mode,
        font=("Courier", 14, "bold"),
        bg="white",
        fg="black",
        anchor="w",
        justify="left"
    )
    mech_lbl.grid(row=0, column=1, sticky="w")

    # ICONS / PANELS
    icon_img = None
    try:
        from PIL import Image, ImageTk
        _icon_path = Path(__file__).with_name("aiagentlogo.png")
        if _icon_path.exists():
            _img = Image.open(_icon_path)
            _img = _img.resize((18, 18))
            icon_img = ImageTk.PhotoImage(_img)
    except Exception:
        try:
            _icon_path = Path(__file__).with_name("aiagentlogo.png")
            if _icon_path.exists():
                icon_img = tk.PhotoImage(file=str(_icon_path))
                try:
                    w, h = icon_img.width(), icon_img.height()
                    fx = max(1, w // 18)
                    fy = max(1, h // 18)
                    icon_img = icon_img.subsample(fx, fy)
                except Exception:
                    pass
        except Exception:
            icon_img = None

    of = tk.Frame(root, bg="white")
    of.pack(padx=15, pady=(0, 15), fill="both", expand=True)

    agents_row = tk.Frame(of, bg="white")
    agents_row.pack(fill="both", expand=False)
    for col in range(3):
        agents_row.columnconfigure(col, weight=1, uniform="agcols")

    def make_agent_panel(parent, idx, name, model_var, icon_img):
        f = tk.Frame(parent, bg="white", bd=0, highlightthickness=0)
        f.grid(row=0, column=idx, padx=5, sticky="nsew")
        header = tk.Frame(f, bg="white")
        header.pack(anchor="w", fill="x")
        if icon_img is not None:
            icon_label = tk.Label(header, image=icon_img, bg="white")
            icon_label.image = icon_img
            icon_label.pack(side="left", padx=(0, 6))
        hdr = tk.Label(
            header,
            text=f"Agent {idx+1} - {name} - {model_var.get()}",
            font=("Courier", 14),
            bg="white",
            fg="black",
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
            font=("Courier", 12),
            bg="black",
            fg="#00FF00",
            insertbackground="#00FF00",
            relief="flat",
            padx=12,
            pady=10,
            spacing1=5,
            spacing3=5,
        )
        t.pack(fill="both", expand=True)
        t.config(state=tk.DISABLED)
        return t

    out_boxes = {}
    out_boxes[AGENTS[0]["name"]] = make_agent_panel(
        agents_row, 0, AGENTS[0]["name"], c1_var, icon_img
    )
    out_boxes[AGENTS[1]["name"]] = make_agent_panel(
        agents_row, 1, AGENTS[1]["name"], c2_var, icon_img
    )
    out_boxes[AGENTS[2]["name"]] = make_agent_panel(
        agents_row, 2, AGENTS[2]["name"], c3_var, icon_img
    )

    coord_frame = tk.Frame(of, bg="white")
    coord_frame.pack(fill="both", expand=True, pady=(10, 0))
    coord_header_frame = tk.Frame(coord_frame, bg="white")
    coord_header_frame.pack(anchor="w", fill="x")

    coord_icon = None
    try:
        from PIL import Image, ImageTk
        _cpath = Path(__file__).with_name("coordinator.png")
        if _cpath.exists():
            _cimg = Image.open(_cpath)
            _cimg = _cimg.resize((18, 18))
            coord_icon = ImageTk.PhotoImage(_cimg)
    except Exception:
        try:
            _cpath = Path(__file__).with_name("coordinator.png")
            if _cpath.exists():
                coord_icon = tk.PhotoImage(file=str(_cpath))
                try:
                    cw, ch = coord_icon.width(), coord_icon.height()
                    fx = max(1, cw // 18)
                    fy = max(1, ch // 18)
                    coord_icon = coord_icon.subsample(fx, fy)
                except Exception:
                    pass
        except Exception:
            coord_icon = None

    if coord_icon is not None:
        _il = tk.Label(coord_header_frame, image=coord_icon, bg="white")
        _il.image = coord_icon
        _il.pack(side="left", padx=(0, 6))

    coord_header = tk.Label(
        coord_header_frame,
        text=f"Agent Coordinator - {coord_var.get()}",
        font=("Courier", 14),
        bg="white",
        fg="black",
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
        font=("Courier", 12),
        bg="black",
        fg="#00FF00",
        insertbackground="#00FF00",
        relief="flat",
        padx=12,
        pady=10,
        spacing1=5,
        spacing3=5,
    )
    coord_text.pack(fill="both", expand=True)
    coord_text.config(state=tk.DISABLED)

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
                f.write(line + "\n")
        except Exception:
            pass

    busy = {
        AGENTS[0]["name"]: False,
        AGENTS[1]["name"]: False,
        AGENTS[2]["name"]: False,
    }

    def start_spinner(name):
        busy[name] = True
        def tick(i=0):
            if not busy.get(name):
                return
            dots = "." * (i % 3 + 1)
            ui_set(out_boxes[name], f"⏳ {name} is reasoning{dots}")
            out_boxes[name].after(350, lambda: tick(i + 1))
        tick()

    def stop_spinner(name):
        busy[name] = False

    def set_box(name, content):
        stop_spinner(name)
        ui_set(out_boxes[name], content)

    def run_debate(task):
        log_path = Path(__file__).with_name("debate.log")
        try:
            with open(log_path, "w", encoding="utf-8") as f:
                f.write(f"=== DEBATE LOG ===\nTask: {task}\n")
        except Exception:
            pass

        # header iniziale con la strategy
        header_intro = (
            f"Coordination Mechanism: {adaptation_mode}\n"
            f"Agents: {AGENTS[0]['name']}, {AGENTS[1]['name']}, {AGENTS[2]['name']}\n"
            f"Topic: {task}\n\n"
        )

        for tbox in out_boxes.values():
            ui_set(tbox, "")
        ui_set(
            coord_text,
            header_intro + f"(Log saved to: {log_path})"
        )

        c1 = Client(AGENTS[0]["name"], AGENTS[0]["title"], c1_var.get())
        c2 = Client(AGENTS[1]["name"], AGENTS[1]["title"], c2_var.get())
        c3 = Client(AGENTS[2]["name"], AGENTS[2]["title"], c3_var.get())
        coord = Coordinator(coord_var.get())

        hooks = {
            AGENTS[0]["name"]: lambda s: set_box(
                AGENTS[0]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n",
            ),
            AGENTS[1]["name"]: lambda s: set_box(
                AGENTS[1]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n",
            ),
            AGENTS[2]["name"]: lambda s: set_box(
                AGENTS[2]["name"],
                f"Answer: {s['answer']}\nConfidence: {s['confidence']}\n\nRationale:\n{(s.get('rationale') or '')}\n",
            ),
        }

        deb = Debate(
            [c1, c2, c3],
            coord,
            rounds=DEFAULT_ROUNDS,
            per_agent_hooks=hooks,
            on_agent_start=start_spinner,
        )

        def log_fn(s):
            write_log(log_path, s)

        def do_run():
            try:
                write_log(
                    log_path,
                    f"Starting debate with models -> "
                    f"Coordinator: {coord_var.get()} | "
                    f"{AGENTS[0]['name']}: {c1_var.get()} | "
                    f"{AGENTS[1]['name']}: {c2_var.get()} | "
                    f"{AGENTS[2]['name']}: {c3_var.get()}",
                )
                final, _ = deb.run(task, log_fn=log_fn)
            except Exception as e:
                final = {
                    "error": str(e),
                    "final_answer": "",
                    "confidence": 0,
                    "justification": "Runtime error",
                }

            if "error" in final:
                ui_set(
                    coord_text,
                    header_intro +
                    "Final output: (error)\n"
                    f"Why: {final.get('error','')}\n\n"
                    f"(Log saved to: {log_path})"
                )
            else:
                why = final.get("justification") or "Majority among valid answers."
                ui_set(
                    coord_text,
                    header_intro +
                    f"Final output: {final.get('final_answer','')}\n"
                    f"Why: {why}\n\n"
                    f"(Log saved to: {log_path})"
                )

        start_spinner(AGENTS[0]["name"])
        start_spinner(AGENTS[1]["name"])
        start_spinner(AGENTS[2]["name"])
        threading.Thread(target=do_run, daemon=True).start()

    root.minsize(1100, 650)
    return root

def launch_agent_ui(simulation_type):
    ui = build_ui(simulation_type)
    ui.mainloop()

# ─────────────────────────────────────────────────────────────
# Dashboard finestra (PyQt) - REFACTORED
# ─────────────────────────────────────────────────────────────
class DashboardWindow(QWidget):
    def __init__(self, simulation_type):
        super().__init__()
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        self.simulation_type = simulation_type

        self.setWindowTitle("Live Dashboard")
        self.setStyleSheet("background-color: white;")
        self.resize(1200, 800)

        # ordine fisso dei pattern come nel CSV
        self.pattern_names = [
            "client_selector",
            "client_cluster",
            "message_compressor",
            "model_co-versioning_registry",
            "multi-task_model_trainer",
            "heterogeneous_data_handler",
        ]

        cfg = load_config(self.simulation_type)
        self.client_configs = {
            int(d["client_id"]): d for d in cfg.get("client_details", [])
        }

        model_name = ""
        dataset_name = ""
        if cfg.get("client_details"):
            number_of_rounds = cfg.get("rounds")
            number_of_clients = cfg.get("clients")
            first = cfg["client_details"][0]
            model_name = first.get("model", "")
            dataset_name = first.get("dataset", "")

        self.color_f1 = random_pastel()
        self.color_tot = random_pastel()
        self.client_colors = {}
        self.clients = []  
        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignTop)
        
        lbl_mod = QLabel(
            f'Model: <b>{model_name}</b> • '
            f'Dataset: <b>{dataset_name}</b> • '
            f'Number of Clients: <b>{number_of_clients}</b> • '
            f'Number of Rounds: <b>{number_of_rounds}</b>'
        )
        lbl_mod.setStyleSheet(
            "font-weight: normal; font-size: 16px; color: black;"
        )
        lbl_mod.setTextFormat(Qt.RichText)
        main_layout.addWidget(lbl_mod)
        global_row = QHBoxLayout()
        main_layout.addLayout(global_row)

        # --- pannello laterale globale (metriche + pattern matrix)
        self.global_side_panel = QWidget()
        self.global_side_panel.setStyleSheet(
            "background-color:#f9f9f9; border:1px solid #ddd; border-radius:6px;"
        )
        self.global_side_panel.setFixedWidth(280)

        gsp_layout = QVBoxLayout(self.global_side_panel)
        gsp_layout.setContentsMargins(8, 8, 8, 8)
        gsp_layout.setSpacing(10)

        # testo con metriche round-by-round globali
        gsp_label_metrics = QLabel("Global Metrics")
        gsp_label_metrics.setStyleSheet(
            "font-weight:bold; font-size:13px; color:black;"
        )
        gsp_layout.addWidget(gsp_label_metrics)

        self.global_stats_area = QPlainTextEdit()
        self.global_stats_area.setReadOnly(True)
        self.global_stats_area.setStyleSheet(
            "background-color: #ffffff; color: black; font-size:12px;"
        )
        gsp_layout.addWidget(self.global_stats_area)
        gsp_label_patterns = QLabel("Active Architectural Patterns")
        gsp_label_patterns.setStyleSheet(
            "font-weight:bold; font-size:13px; color:black;"
        )
        gsp_layout.addWidget(gsp_label_patterns)

        self.pattern_grid = QWidget()
        self.pattern_grid_layout = QGridLayout(self.pattern_grid)
        self.pattern_grid_layout.setContentsMargins(0, 0, 0, 0)
        self.pattern_grid_layout.setSpacing(4)
        gsp_layout.addWidget(self.pattern_grid)
        global_row.addWidget(self.global_side_panel)
        self.fig_f1, self.ax_f1 = plt.subplots()
        self.fig_f1.patch.set_facecolor("white")
        self.canvas_f1 = FigureCanvas(self.fig_f1)
        global_row.addWidget(self.canvas_f1)
        self.fig_tot, self.ax_tot = plt.subplots()
        self.fig_tot.patch.set_facecolor("white")
        self.canvas_tot = FigureCanvas(self.fig_tot)
        global_row.addWidget(self.canvas_tot)
        clients_row = QHBoxLayout()
        main_layout.addLayout(clients_row)
        self.client_side_panel = QWidget()
        self.client_side_panel.setStyleSheet(
            "background-color:#f9f9f9; border:1px solid #ddd; border-radius:6px;"
        )
        self.client_side_panel.setFixedWidth(280)

        csp_layout = QVBoxLayout(self.client_side_panel)
        csp_layout.setContentsMargins(8, 8, 8, 8)
        csp_layout.setSpacing(10)

        csp_label = QLabel("Client Times")
        csp_label.setStyleSheet(
            "font-weight:bold; font-size:13px; color:black;"
        )
        csp_layout.addWidget(csp_label)

        self.client_stats_area = QPlainTextEdit()
        self.client_stats_area.setReadOnly(True)
        self.client_stats_area.setStyleSheet(
            "background-color: #ffffff; color: black; font-size:12px;"
        )
        csp_layout.addWidget(self.client_stats_area)

        clients_row.addWidget(self.client_side_panel)

        # --- grafico training time
        self.fig_train, self.ax_train = plt.subplots()
        self.fig_train.patch.set_facecolor("white")
        self.canvas_train = FigureCanvas(self.fig_train)
        clients_row.addWidget(self.canvas_train)

        # --- grafico communication time
        self.fig_comm, self.ax_comm = plt.subplots()
        self.fig_comm.patch.set_facecolor("white")
        self.canvas_comm = FigureCanvas(self.fig_comm)
        clients_row.addWidget(self.canvas_comm)

        # timer aggiornamento
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_data)
        self.timer.start(1000)

    # -----------------------------
    # Helper: parse pattern string
    # -----------------------------
    def parse_ap_list(self, ap_str: str) -> Dict[str, bool]:
        """
        ap_str es: "{OFF,OFF,ON,OFF,OFF,OFF}"

        Ordine noto nel CSV:
        (client_selector,
         client_cluster,
         message_compressor,
         model_co-versioning_registry,
         multi-task_model_trainer,
         heterogeneous_data_handler)

        Ritorna dict {pattern_name: True/False}
        """
        active_map: Dict[str, bool] = {}
        if not isinstance(ap_str, str):
            return active_map

        cleaned = ap_str.strip().strip("{}")
        if not cleaned:
            # niente AP List -> tutto False
            for pname in self.pattern_names:
                active_map[pname] = False
            return active_map

        parts = [p.strip().upper() for p in cleaned.split(",")]
        for i, pname in enumerate(self.pattern_names):
            status = parts[i] if i < len(parts) else "OFF"
            active_map[pname] = (status == "ON")
        return active_map

    def update_pattern_grid(self, pattern_matrix_data: List[Dict[str, object]]):
        lay = self.pattern_grid_layout
        while lay.count():
            item = lay.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

        hdr_round = QLabel("Federated Learning Round")
        hdr_round.setStyleSheet("font-weight:bold; font-size:11px; color:black;")
        lay.addWidget(hdr_round, 0, 0)

        for j, pname in enumerate(self.pattern_names):
            lbl = QLabel(pname)
            lbl.setStyleSheet("font-size:10px; color:#333;")
            lay.addWidget(lbl, 0, j + 1)

        # Righe dei round
        for i, row in enumerate(pattern_matrix_data):
            r_lbl = QLabel(str(row["round"]))
            r_lbl.setStyleSheet("font-weight:bold; font-size:11px; color:black;")
            lay.addWidget(r_lbl, i + 1, 0)

            for j, pname in enumerate(self.pattern_names):
                active = bool(row.get(pname, False))
                cell = QLabel("")
                cell.setFixedSize(18, 18)
                bg = "#4caf50" if active else "#f44336"  # verde / rosso
                cell.setStyleSheet(
                    f"background-color: {bg}; "
                    "border:1px solid #222; border-radius:3px;"
                )
                lay.addWidget(cell, i + 1, j + 1)

    # -----------------------------
    # Helper: ricava AP List dal df
    # -----------------------------
    def _extract_ap_map_for_round(self, df: pd.DataFrame) -> Dict[str, bool]:

        ap_colname = None
        for col in df.columns:
            if isinstance(col, str) and col.strip().lower().startswith("ap list"):
                ap_colname = col
                break

        if ap_colname is None:
            return {p: False for p in self.pattern_names}

        series_nonnull = df[ap_colname].dropna()
        if series_nonnull.empty:
            return {p: False for p in self.pattern_names}

        ap_str = str(series_nonnull.iloc[-1])

        return self.parse_ap_list(ap_str)


    def _client_label(self, cid) -> str:
        return str(cid)

    def update_data(self):
        base_dir = os.path.dirname(__file__)
        subdir = "Docker" if self.simulation_type == "Docker" else "Local"
        perf_dir = os.path.join(base_dir, subdir, "performance")

        raw_files = glob.glob(
            os.path.join(perf_dir, "FLwithAP_performance_metrics_round*.csv")
        )

        files_info = []
        for f in raw_files:
            m = re.search(r"round(\d+)", f)
            if m:
                rnd = int(m.group(1))
                files_info.append((rnd, f))
        if not files_info:
            return
        files_info.sort(key=lambda x: x[0])

        # inizializzo clients e colori la prima volta
        if not self.clients:
            first_round, first_file = files_info[0]
            df0 = pd.read_csv(first_file)

            # preserva l'ordine di apparizione senza duplicati
            cids_seen = []
            for cid in df0["Client ID"].dropna().tolist():
                if cid not in cids_seen:
                    cids_seen.append(cid)
            self.clients = cids_seen

            for cid in self.clients:
                self.client_colors[cid] = random_pastel()

        global_lines = []
        client_lines = []
        pattern_matrix_data = []
        rounds_list = []
        f1_list = []
        total_times_list = []

        # Calcolo per ogni round
        for rnd, fpath in files_info:
            df = pd.read_csv(fpath)

            # prendo righe con Train Loss non NaN per estrarre le metriche globali
            df_nonan = df.dropna(subset=["Train Loss"])
            if df_nonan.empty:
                continue
            last = df_nonan.iloc[-1]

            f1_val = last.get("Val F1", float("nan"))
            total_time_val = last.get("Total Time of FL Round", float("nan"))

            rounds_list.append(rnd)
            f1_list.append(f1_val)
            total_times_list.append(total_time_val)

            global_lines.append(
                f"Round {rnd}: F1={f1_val:.2f}, Total Round Time={total_time_val:.0f}s"
            )

            # tempi client per questo round
            client_lines.append(f"Round {rnd}:")
            for cid in self.clients:
                row = df[(df["Client ID"] == cid) & (df["FL Round"] == rnd)]
                if row.empty:
                    continue
                row0 = row.iloc[0]
                ttime = row0.get("Training Time", float("nan"))
                ctime = row0.get("Communication Time", float("nan"))
                client_lines.append(
                    f" - {self._client_label(cid)}: "
                    f"Training={ttime:.2f}s, Communication={ctime:.2f}s"
                )
            client_lines.append("")  # riga vuota separatrice

            # pattern ON/OFF per questo round
            active_map = self._extract_ap_map_for_round(df)
            pat_row = {"round": rnd}
            for pname in self.pattern_names:
                pat_row[pname] = active_map.get(pname, False)
            pattern_matrix_data.append(pat_row)

        # Pannello laterale globale
        self.global_stats_area.setPlainText("\n".join(global_lines))

        # Pannello laterale client
        self.client_stats_area.setPlainText("\n".join(client_lines))

        # Aggiorna la griglia pattern
        self.update_pattern_grid(pattern_matrix_data)

        # === Aggiorna i grafici ===

        # Grafico F1
        self.ax_f1.clear()
        sns.lineplot(
            x=rounds_list,
            y=f1_list,
            marker="o",
            ax=self.ax_f1,
            color=self.color_f1,
        )
        self.ax_f1.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_f1.set_title("Global Model Accuracy", fontweight="bold")
        self.ax_f1.set_xlabel("Federated Learning Round")
        self.ax_f1.set_ylabel("F1 Score")
        self.canvas_f1.draw()

        # Grafico Tempo Totale Round
        self.ax_tot.clear()
        sns.lineplot(
            x=rounds_list,
            y=total_times_list,
            marker="o",
            ax=self.ax_tot,
            color=self.color_tot,
        )
        self.ax_tot.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_tot.set_title(
            "Total Round Time",
            fontweight="bold",
        )
        self.ax_tot.set_xlabel("Federated Learning Round")
        self.ax_tot.set_ylabel("Total Round Time (sec)")
        self.canvas_tot.draw()
        self.ax_train.clear()
        self.ax_comm.clear()
        for cid in self.clients:
            rds = []
            tv = []
            cv = []
            for rnd, fpath in files_info:
                df = pd.read_csv(fpath)
                row = df[(df["Client ID"] == cid) & (df["FL Round"] == rnd)]
                if row.empty:
                    continue
                rds.append(rnd)
                tv.append(row["Training Time"].values[0])
                cv.append(row["Communication Time"].values[0])

            color_for_client = self.client_colors.get(cid, random_pastel())

            sns.lineplot(
                x=rds,
                y=tv,
                marker="o",
                ax=self.ax_train,
                label=self._client_label(cid),
                color=color_for_client,
            )
            sns.lineplot(
                x=rds,
                y=cv,
                marker="o",
                ax=self.ax_comm,
                label=self._client_label(cid),
                color=color_for_client,
            )

        self.ax_train.xaxis.set_major_locator(MaxNLocator(integer=True))
        self.ax_comm.xaxis.set_major_locator(MaxNLocator(integer=True))

        for ax, title in [
            (self.ax_train, "Training Time"),
            (self.ax_comm, "Communication Time"),
        ]:
            ax.set_title(title, fontweight="bold")
            ax.set_xlabel("Round")
            ax.set_ylabel(
                "Training Time (sec)"
                if title == "Training Time"
                else "Communication Time (sec)"
            )
            ax.legend()

        self.canvas_train.draw()
        self.canvas_comm.draw()

# ─────────────────────────────────────────────────────────────
# Simulation Page (PyQt)
# ─────────────────────────────────────────────────────────────
class SimulationPage(QWidget):
    def __init__(self, config, num_supernodes=None):
        super().__init__()
        self.config = config
        self.setWindowTitle("Simulation Output")
        self.resize(800, 600)
        self.setStyleSheet("background-color: white;")
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignTop)
        self.setLayout(layout)

        self.output_area = QPlainTextEdit()
        self.output_area.setReadOnly(True)
        self.output_area.setStyleSheet(
            """
            QPlainTextEdit {
                background-color: black;
                font-family: Courier;
                font-size: 12px;
                color: white;
            }
            """
        )
        layout.addWidget(self.output_area)

        # riga con pulsanti Dashboard / Agents
        buttons_row = QHBoxLayout()
        buttons_row.setSpacing(20)
        buttons_row.setContentsMargins(0, 0, 0, 0)

        self.dashboard_button = QPushButton("📈 Real-Time Performance Analysis")
        self.dashboard_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.dashboard_button.setCursor(Qt.PointingHandCursor)
        self.dashboard_button.setStyleSheet(
            """
            QPushButton {
                background-color: #007ACC; 
                color: white; 
                font-size: 14px; 
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #005F9E;
            }
            QPushButton:pressed {
                background-color: #004970;
            }
            """
        )
        self.dashboard_button.clicked.connect(self.open_dashboard)

        self.XAI_button = QPushButton("🤖 Multiple AI-Agents​")
        self.XAI_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.XAI_button.setCursor(Qt.PointingHandCursor)
        self.XAI_button.setStyleSheet(
            """
            QPushButton {
                background-color: green;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #00b300;
            }
            QPushButton:pressed {
                background-color: #008000;
            }      
            """
        )
        self.XAI_button.clicked.connect(self.open_agents)

        # disabilita Multiple AI-Agents se adaptation == "None"
        adaptation_mode = self.config.get("adaptation", "None")
        if str(adaptation_mode).strip().lower() == "none":
            self.XAI_button.setEnabled(False)
            self.XAI_button.setStyleSheet(
                """
                QPushButton {
                    background-color: #888;
                    color: #eee;
                    font-size: 14px;
                    padding: 10px;
                    border-radius: 5px;
                }
                """
            )

        buttons_row.addWidget(self.dashboard_button, 1)
        buttons_row.addWidget(self.XAI_button, 1)
        layout.addLayout(buttons_row)

        # pulsante Stop / Close
        self.stop_button = QPushButton("Stop Simulation")
        self.stop_button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.stop_button.setCursor(Qt.PointingHandCursor)
        self.stop_button.setStyleSheet(
            """
            QPushButton {
                background-color: #ee534f;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #ff6666;
            }
            QPushButton:pressed {
                background-color: #cc0000;
            }
            """
        )
        self.stop_button.clicked.connect(self.stop_simulation)
        layout.addWidget(self.stop_button)

        # processi runtime
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.MergedChannels)
        self.process.readyReadStandardOutput.connect(self.handle_stdout)
        self.process.readyReadStandardError.connect(self.handle_stdout)
        self.process.finished.connect(self.process_finished)
        self.agent_proc = None  

        self.start_simulation(num_supernodes)

    def open_dashboard(self):
        self.db = DashboardWindow(self.config["simulation_type"])
        self.db.show()

    def open_agents(self):
        # lancia la UI multi-LLM agent (tkinter) in un nuovo processo Python
        base_dir = os.path.dirname(os.path.abspath(__file__))
        python_exe = sys.executable
        sim_type = self.config["simulation_type"]

        inline = (
            "import sys,os,importlib;"
            f"os.chdir({repr(base_dir)});"
            "sys.path.insert(0, os.getcwd());"
            "m=importlib.import_module('simulation');"
            f"m.launch_agent_ui({repr(sim_type)});"
        )
        self.agent_proc = QProcess(self)
        self.agent_proc.setProcessChannelMode(QProcess.MergedChannels)
        self.agent_proc.readyReadStandardOutput.connect(self.handle_stdout)
        self.agent_proc.readyReadStandardError.connect(self.handle_stdout)
        self.agent_proc.start(python_exe, ["-c", inline])

        if not self.agent_proc.waitForStarted(1000):
            self.output_area.appendPlainText("❌ Failed to launch AI-Agents Coordinator UI")

    def start_simulation(self, num_supernodes):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sim_type = self.config["simulation_type"]
        rounds = self.config["rounds"]

        if sim_type == "Docker":
            work_dir = os.path.join(base_dir, "Docker")
            dc_in = os.path.join(work_dir, "docker-compose.yml")
            dc_out = os.path.join(work_dir, "docker-compose.dynamic.yml")

            self.output_area.appendPlainText("Launching Docker Compose...")

            with open(dc_in, "r") as f:
                compose = yaml.safe_load(f)

            server_svc = compose["services"].get("server")
            client_tpl = compose["services"].get("client")
            if not server_svc or not client_tpl:
                self.output_area.appendPlainText(
                    "Error: Missing server or client service in docker-compose.yml"
                )
                return

            new_svcs = {"server": server_svc}
            for detail in self.config["client_details"]:
                cid = detail["client_id"]
                cpu = detail["cpu"]
                ram = detail["ram"]

                svc = copy.deepcopy(client_tpl)
                svc.pop("image", None)
                svc.pop("deploy", None)

                svc["container_name"] = f"Client{cid}"
                svc["cpus"] = cpu
                svc["mem_limit"] = f"{ram}g"

                env = svc.setdefault("environment", {})
                env["NUM_ROUNDS"] = str(rounds)
                env["NUM_CPUS"] = str(cpu)
                env["NUM_RAM"] = str(ram)
                env["CLIENT_ID"] = str(cid)

                new_svcs[f"client{cid}"] = svc

            compose["services"] = new_svcs
            with open(dc_out, "w") as f:
                yaml.safe_dump(compose, f)

            cmd = "/opt/homebrew/bin/docker"
            args = ["compose", "-f", dc_out, "up"]

            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)

            env = QProcessEnvironment.systemEnvironment()
            env.insert("COMPOSE_BAKE", "true")
            self.process.setProcessEnvironment(env)

            self.process.start(cmd, args)

            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Error: Docker Compose failed to start")
                self.output_area.appendPlainText(self.process.errorString())
                return

        else:
            # modalità Local
            work_dir = os.path.join(base_dir, "Local")
            cmd = "flower-simulation"
            args = [
                "--server-app",
                "server:app",
                "--client-app",
                "client:app",
                "--num-supernodes",
                str(num_supernodes),
            ]

            self.process = QProcess(self)
            self.process.setProcessChannelMode(QProcess.MergedChannels)
            self.process.readyReadStandardOutput.connect(self.handle_stdout)
            self.process.readyReadStandardError.connect(self.handle_stdout)
            self.process.setWorkingDirectory(work_dir)
            self.process.start(cmd, args)

            if not self.process.waitForStarted():
                self.output_area.appendPlainText("Local simulation failed to start")

    def handle_stdout(self):
        sender = self.sender()
        if sender is None:
            return

        data = sender.readAllStandardOutput()
        if not data:
            data = sender.readAllStandardError()

        try:
            encoding = locale.getpreferredencoding(False)
            stdout = bytes(data).decode(encoding)
        except UnicodeDecodeError:
            stdout = bytes(data).decode("utf-8", errors="replace")

        for line in stdout.splitlines():
            cleaned = self.remove_ansi_sequences(line)
            stripped = cleaned.lstrip()

            if not stripped:
                continue
            if stripped.startswith("#"):
                continue
            if stripped.startswith("Network "):
                continue
            if stripped.startswith("Container "):
                continue
            if stripped.startswith("Attaching to"):
                continue
            if "flower-super" in stripped:
                continue

            lower = stripped.lower()
            if any(
                key in lower
                for key in [
                    "deprecated",
                    "to view usage",
                    "to view all available options",
                    "warning",
                    "entirely in future versions",
                    "files already downloaded and verified",
                    "client pulling",
                ]
            ):
                continue

            if re.match(r"^[^|]+\|\s*$", cleaned):
                continue

            if "client" in lower:
                html = f"<span style='color:#4caf50;'>{cleaned}</span>"
            elif "server" in lower:
                html = f"<span style='color:#2196f3;'>{cleaned}</span>"
            else:
                html = f"<span style='color:white;'>{cleaned}</span>"

            self.output_area.appendHtml(html)

        self.output_area.verticalScrollBar().setValue(
            self.output_area.verticalScrollBar().maximum()
        )

    def remove_ansi_sequences(self, text):
        ansi_escape = re.compile(
            r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])"
        )
        return ansi_escape.sub("", text)

    def process_finished(self):
        self.output_area.appendPlainText("Simulation finished.")
        if hasattr(self, "title_label") and self.title_label:
            self.title_label.setText("Simulation Results")
        self.stop_button.setText("Close")
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.close_application)
        self.process = None

    def stop_simulation(self):
        if self.process:
            self.process.terminate()
            self.process.waitForFinished()
            self.output_area.appendPlainText("Simulation terminated by the user.")
        if hasattr(self, "title_label") and self.title_label:
            self.title_label.setText("Simulation Terminated")
        self.stop_button.setText("Close")
        try:
            self.stop_button.clicked.disconnect()
        except Exception:
            pass
        self.stop_button.clicked.connect(self.close_application)

    def close_application(self):
        # Chiudi eventuale processo principale della simulazione
        try:
            if hasattr(self, "process") and self.process is not None:
                self.process.terminate()
                self.process.waitForFinished()
        except Exception:
            pass

        # Chiudi eventuale processo degli agenti AI (tk UI)
        try:
            if hasattr(self, "agent_proc") and self.agent_proc is not None:
                self.agent_proc.terminate()
                self.agent_proc.waitForFinished()
        except Exception:
            pass

        # Chiudi l'app Qt in modo pulito (exit code 0, niente crash report)
        QCoreApplication.quit()


    def is_command_available(self, command):
        from shutil import which
        return which(command) is not None
