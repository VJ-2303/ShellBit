#!/usr/bin/env python3
"""Parse generated commands and compute syntax, lint, grounding and hazard diagnostics."""

import json
import re
import shlex
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple


# Deterministic destructive / dangerous command patterns
BLOCKED_PATTERNS = [
    r"rm\s+[^\n]*-[^\n]*r[^\n]*f[^\n]*\s+/(?:\s|$|\*)",
    r"rm\s+[^\n]*--no-preserve-root",
    r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",  # Fork bomb
    r">\s*/dev/sd[a-z]",
    r"\bdd\b[^\n]*\bof=/dev/(sd[a-z]|nvme[0-9]|hd[a-z])",
    r"\bmkfs(\.[a-z0-9]+)?\b",
    r"\bcurl\b[^\n]*\|\s*(sudo\s+)?(ba)?sh\b",
    r"\bwget\b[^\n]*\|\s*(sudo\s+)?(ba)?sh\b",
]

CONFIRM_DESTRUCTIVE_PATTERNS = [
    r"\brm\s+[^\n]*-[^\n]*r",
    r"\brm\s+[^\n]*-[^\n]*f",
    r"\bfind\b[^\n]*-delete\b",
    r"\brsync\b[^\n]*--delete\b",
    r"\bchmod\s+-R\s+777\b",
    r"\bchown\s+-R\b",
    r"\bkill\s+-9\b",
    r"\bpkill\b",
    r"\bshutdown\b|\breboot\b|\bsystemctl\s+(stop|disable|poweroff|reboot)\b",
    r"\bgit\s+reset\s+--hard\b|\bgit\s+push\s+[^\n]*--force\b",
    r"\bdrop\s+(table|database)\b",
]

# Read-only safe binaries allowed to run inside isolated bwrap/tmpfs dry-run
SAFE_DRYRUN_BINS = {
    "echo", "printf", "pwd", "uname", "date", "whoami", "id", "ls", "wc",
    "sort", "uniq", "cut", "tr", "head", "tail", "grep", "awk", "sed",
    "find", "cat", "stat", "basename", "dirname", "realpath", "seq", "jq",
}


@dataclass
class VerificationResult:
    command: str
    plan: str
    syntax_ok: bool
    shellcheck_warnings: int
    shellcheck_errors: int
    schema_score: float
    exec_or_sem_score: float
    format_score: float
    safety_level: str          # SAFE | CONFIRM_DESTRUCTIVE | BLOCKED
    total_reward: float
    diagnostics: str = ""


def extract_plan_and_cmd(raw_text: str) -> Tuple[str, str, bool]:
    """Parse <plan>...</plan><cmd>...</cmd> from model output."""
    plan_match = re.search(r"<plan>(.*?)</plan>", raw_text, flags=re.DOTALL)
    cmd_match = re.search(r"<cmd>(.*?)</cmd>", raw_text, flags=re.DOTALL)

    if cmd_match:
        plan = plan_match.group(1).strip() if plan_match else ""
        cmd = cmd_match.group(1).strip()
        # Strip markdown fences if model accidentally emitted them inside <cmd>
        cmd = re.sub(r"^```(?:bash|sh)?\s*|\s*```$", "", cmd).strip()
        return plan, cmd, bool(cmd)

    # Fallback: extract fenced bash block or first non-empty line
    fence = re.search(r"```(?:bash|sh)?\n(.*?)```", raw_text, flags=re.DOTALL)
    if fence:
        return "", fence.group(1).strip(), False

    lines = [ln.strip() for ln in raw_text.strip().splitlines() if ln.strip()]
    return "", (lines[-1] if lines else ""), False


def check_safety(cmd: str) -> str:
    """Classify command hazard level deterministically."""
    if not cmd:
        return "SAFE"
    for pat in BLOCKED_PATTERNS:
        if re.search(pat, cmd, flags=re.IGNORECASE):
            return "BLOCKED"
    for pat in CONFIRM_DESTRUCTIVE_PATTERNS:
        if re.search(pat, cmd, flags=re.IGNORECASE):
            return "CONFIRM_DESTRUCTIVE"
    return "SAFE"


def check_bash_syntax(cmd: str) -> Tuple[bool, str]:
    """Run `bash -n` to verify grammatical validity."""
    if not cmd or not cmd.strip():
        return False, "empty command"
    if shutil.which("bash") is None:
        # Pure-Python fallback if bash is absent
        try:
            shlex.split(cmd)
            return True, "shlex fallback ok"
        except ValueError as e:
            return False, str(e)
    try:
        proc = subprocess.run(
            ["bash", "-n", "-c", cmd],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=1.0,
            check=False,
        )
        return proc.returncode == 0, proc.stderr.strip()
    except Exception as e:
        return False, str(e)


def check_shellcheck(cmd: str) -> Tuple[int, int, str]:
    """Run `shellcheck -s bash -f json` if available; returns (errors, warnings, msg)."""
    if not cmd or shutil.which("shellcheck") is None:
        # Lightweight static heuristic fallback when shellcheck binary isn't installed
        warnings = 0
        # Flag unquoted variable expansion like `rm $file` (simple heuristic)
        if re.search(r'(?<!["\'])\$[A-Za-z_][A-Za-z0-9_]*\b', cmd) and '"' not in cmd:
            warnings += 1
        # Flag useless use of `cat file | grep`
        if re.search(r"\bcat\s+[^\s|]+\s*\|\s*(grep|awk|sed)\b", cmd):
            warnings += 1
        return 0, warnings, "heuristic-fallback"

    try:
        script = f"#!/usr/bin/env bash\n{cmd}\n"
        proc = subprocess.run(
            ["shellcheck", "-s", "bash", "-f", "json", "-"],
            input=script,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=2.0,
            check=False,
        )
        if proc.returncode not in (0, 1) or not proc.stdout.strip():
            return 1, 0, "shellcheck-failed"
        items = json.loads(proc.stdout)
        errors = sum(1 for it in items if it.get("level") == "error")
        warnings = sum(1 for it in items if it.get("level") in ("warning", "info"))
        codes = ",".join(f"SC{it.get('code')}" for it in items[:4])
        return errors, warnings, codes
    except Exception as e:
        return 1, 0, f"shellcheck-err:{e}"


def extract_primary_binaries(cmd: str) -> List[str]:
    """Extract command binaries across pipelines and && / ; chains."""
    segments = re.split(r"\s*(?:\|\||&&|\||;)\s*", cmd)
    bins = []
    skip = {"sudo", "env", "nohup", "time", "nice", "ionice", "command", "builtin"}
    for seg in segments:
        try:
            parts = shlex.split(seg)
        except ValueError:
            parts = seg.split()
        for p in parts:
            if "=" in p and not p.startswith("-"):
                continue  # env var assignment
            if p in skip:
                continue
            bins.append(p)
            break
    return bins


def check_schema_grounding(cmd: str, schema_text: str = "") -> float:
    """Verify generated binaries exist on PATH or in injected Schema-RAG card."""
    bins = extract_primary_binaries(cmd)
    if not bins:
        return 0.0

    schema_tools = set(re.findall(r"^([a-zA-Z0-9_-]+):", schema_text, flags=re.MULTILINE))
    shell_builtins = {
        "cd", "echo", "printf", "export", "unset", "alias", "read", "test",
        "[", "[[", "if", "for", "while", "case", "true", "false", "pwd",
        "umask", "ulimit", "kill", "wait", "trap", "set", "shopt", "source", ".",
    }

    valid = 0
    for b in bins:
        if b in schema_tools or b in shell_builtins or shutil.which(b) is not None:
            valid += 1
    return valid / len(bins)


def bash_token_f1(pred_cmd: str, gold_cmd: str) -> float:
    """Compute AST-aware normalized token F1 between predicted and gold Bash commands."""
    if not pred_cmd or not gold_cmd:
        return 0.0
    if pred_cmd.strip() == gold_cmd.strip():
        return 1.0

    def tokenize(c: str) -> List[str]:
        try:
            return shlex.split(c)
        except ValueError:
            return re.findall(r"[^\s|&;]+|\|\||&&|[|&;]", c)

    p_toks = tokenize(pred_cmd)
    g_toks = tokenize(gold_cmd)
    if not p_toks or not g_toks:
        return 0.0

    p_counts: Dict[str, int] = {}
    for t in p_toks:
        p_counts[t] = p_counts.get(t, 0) + 1
    g_counts: Dict[str, int] = {}
    for t in g_toks:
        g_counts[t] = g_counts.get(t, 0) + 1

    overlap = sum(min(p_counts[t], g_counts.get(t, 0)) for t in p_counts)
    if overlap == 0:
        return 0.0
    prec = overlap / len(p_toks)
    rec = overlap / len(g_toks)
    f1 = 2 * prec * rec / (prec + rec)

    # Bonus if primary binary matches gold primary binary
    p_bins = extract_primary_binaries(pred_cmd)
    g_bins = extract_primary_binaries(gold_cmd)
    if p_bins and g_bins and p_bins[0] == g_bins[0]:
        f1 = min(1.0, f1 * 0.85 + 0.15)
    return f1


def sandboxed_dry_run(cmd: str) -> Optional[bool]:
    """
    Execute read-only safe commands inside an isolated bubblewrap (bwrap) + tmpfs
    sandbox with zero network access and 1s timeout.
    Returns True if exit_code==0, False if non-zero, or None if skipped (unsafe/no bwrap).
    """
    if check_safety(cmd) != "SAFE":
        return None
    bins = extract_primary_binaries(cmd)
    if not bins or not all(b in SAFE_DRYRUN_BINS for b in bins):
        return None
    if shutil.which("bwrap") is None:
        return None

    try:
        with tempfile.TemporaryDirectory(prefix="shellforge_sbx_") as tmpdir:
            # Create sample dummy files for realistic find/grep/ls testing
            open(f"{tmpdir}/sample.txt", "w").write("alpha\nbeta\ngamma\n")
            open(f"{tmpdir}/app.log", "w").write("INFO ok\nERROR failed\n")
            bwrap_cmd = [
                "bwrap",
                "--ro-bind", "/usr", "/usr",
                "--ro-bind", "/bin", "/bin",
                "--ro-bind", "/lib", "/lib",
                "--bind", tmpdir, "/work",
                "--chdir", "/work",
                "--unshare-net",
                "--unshare-pid",
                "--die-with-parent",
                "--",
                "bash", "-c", cmd,
            ]
            # Add /lib64 if present on system
            if shutil.os.path.exists("/lib64"):
                bwrap_cmd.insert(7, "--ro-bind")
                bwrap_cmd.insert(8, "/lib64")
                bwrap_cmd.insert(9, "/lib64")

            proc = subprocess.run(
                bwrap_cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=1.5,
                check=False,
            )
            return proc.returncode == 0
    except Exception:
        return None


def compute_composite_reward(
    raw_output: str,
    gold_cmd: Optional[str] = None,
    schema_text: str = "",
    weights: Optional[Dict[str, float]] = None,
    enable_sandbox_exec: bool = True,
) -> VerificationResult:
    """Compute the full R-GRPO composite reward in [0.0, 1.0]."""
    w = weights or {
        "syntax": 0.25,
        "robustness": 0.30,
        "schema_grounding": 0.15,
        "exec_or_semantic": 0.20,
        "format": 0.10,
    }

    plan, cmd, format_ok = extract_plan_and_cmd(raw_output)
    safety = check_safety(cmd)

    # Hard penalty for catastrophic blocked commands
    if safety == "BLOCKED":
        return VerificationResult(
            command=cmd,
            plan=plan,
            syntax_ok=False,
            shellcheck_warnings=0,
            shellcheck_errors=1,
            schema_score=0.0,
            exec_or_sem_score=0.0,
            format_score=0.0,
            safety_level=safety,
            total_reward=-0.5,
            diagnostics="BLOCKED destructive pattern",
        )

    syntax_ok, syn_msg = check_bash_syntax(cmd)
    r_syntax = 1.0 if syntax_ok else 0.0

    sc_errors, sc_warnings, sc_msg = check_shellcheck(cmd) if syntax_ok else (1, 0, "syntax-fail")
    if not syntax_ok or sc_errors > 0:
        r_robust = 0.0
    elif sc_warnings == 0:
        r_robust = 1.0
    else:
        r_robust = max(0.2, 1.0 - 0.3 * sc_warnings)

    r_schema = check_schema_grounding(cmd, schema_text) if syntax_ok else 0.0

    # Combine semantic token F1 vs gold with optional bwrap sandboxed dry-run
    sem_score = bash_token_f1(cmd, gold_cmd) if gold_cmd else (0.5 if syntax_ok else 0.0)
    exec_res = sandboxed_dry_run(cmd) if (syntax_ok and enable_sandbox_exec) else None
    if exec_res is True:
        r_exec = min(1.0, 0.7 * sem_score + 0.3)
    elif exec_res is False:
        r_exec = 0.5 * sem_score
    else:
        r_exec = sem_score

    # Format compliance + concise plan bonus (keep <plan> compact for sub-150M speed)
    if format_ok:
        plan_words = len(plan.split())
        r_format = 1.0 if not plan or 3 <= plan_words <= 45 else 0.7
    else:
        r_format = 0.2 if cmd else 0.0

    total = (
        w["syntax"] * r_syntax
        + w["robustness"] * r_robust
        + w["schema_grounding"] * r_schema
        + w["exec_or_semantic"] * r_exec
        + w["format"] * r_format
    )

    return VerificationResult(
        command=cmd,
        plan=plan,
        syntax_ok=syntax_ok,
        shellcheck_warnings=sc_warnings,
        shellcheck_errors=sc_errors,
        schema_score=r_schema,
        exec_or_sem_score=r_exec,
        format_score=r_format,
        safety_level=safety,
        total_reward=round(total, 4),
        diagnostics=f"syn={syn_msg or 'ok'} sc={sc_msg or 'clean'}",
    )
