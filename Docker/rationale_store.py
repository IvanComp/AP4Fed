import csv
import os
import re
from typing import Dict, List

from adaptation_settings import AGENT_LOG_FILE, PERFORMANCE_DIR, RATIONALE_CSV_FILE

def _append_agent_log(lines):
    p = AGENT_LOG_FILE
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "a", encoding="utf-8") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def _extract_rationale_entries(round_idx: int, policy: str, decision_logs: List[str]) -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    patterns = [
        (
            re.compile(r"^\[Rationale\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": "[Rationale]",
                "turn": "",
                "agent": "",
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Rationale A(?P<agent>\d+)\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Rationale A{m.group('agent')}]",
                "turn": "",
                "agent": m.group("agent"),
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Rationale (?P<role>CS|MC|HDH)\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Rationale {m.group('role')}]",
                "turn": "",
                "agent": "",
                "role": m.group("role"),
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(r"^\[Coordinator Rationale\]\s*(?P<text>.*)$", re.S),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": "[Coordinator Rationale]",
                "turn": "",
                "agent": "",
                "role": "Coordinator",
                "rationale": m.group("text").strip(),
            },
        ),
        (
            re.compile(
                r"^\[Debate\]\[Turn (?P<turn>\d+)\] Rationale A(?P<agent>\d+):\s*(?P<text>.*)$",
                re.S,
            ),
            lambda m: {
                "round": str(round_idx),
                "policy": policy,
                "tag": f"[Debate][Turn {m.group('turn')}] Rationale A{m.group('agent')}",
                "turn": m.group("turn"),
                "agent": m.group("agent"),
                "role": "",
                "rationale": m.group("text").strip(),
            },
        ),
    ]

    for raw_line in decision_logs:
        if raw_line is None:
            continue
        text = str(raw_line).strip()
        if not text:
            continue
        for pattern, factory in patterns:
            match = pattern.match(text)
            if not match:
                continue
            entry = factory(match)
            rationale = entry.get("rationale", "").strip()
            if not rationale or rationale.lower().startswith("(omitted:"):
                break
            entry["rationale"] = rationale
            entries.append(entry)
            break

    return entries


def _persist_round_rationales(round_idx: int, policy: str, decision_logs: List[str]) -> None:
    entries = _extract_rationale_entries(round_idx, policy, decision_logs)
    if not entries:
        return

    os.makedirs(PERFORMANCE_DIR, exist_ok=True)

    fieldnames = ["round", "policy", "tag", "turn", "agent", "role", "rationale"]
    write_header = not os.path.exists(RATIONALE_CSV_FILE)
    with open(RATIONALE_CSV_FILE, "a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(entries)

    round_txt_path = os.path.join(PERFORMANCE_DIR, f"FLwithAP_adaptation_rationales_round{round_idx}.txt")
    with open(round_txt_path, "w", encoding="utf-8") as txt_file:
        txt_file.write(f"Round: {round_idx}\n")
        txt_file.write(f"Policy: {policy}\n\n")
        for entry in entries:
            header_bits = [entry["tag"]]
            if entry["turn"]:
                header_bits.append(f"turn={entry['turn']}")
            if entry["agent"]:
                header_bits.append(f"agent={entry['agent']}")
            if entry["role"]:
                header_bits.append(f"role={entry['role']}")
            txt_file.write(" | ".join(header_bits) + "\n")
            txt_file.write(entry["rationale"] + "\n\n")
