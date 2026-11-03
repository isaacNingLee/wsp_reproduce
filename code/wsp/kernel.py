"""A persistent Python kernel for the editor LLM's code, run in a bubblewrap sandbox.

Like the blog's harness, the LLM's code runs with exec() in one namespace that survives across turns; the
namespace is built by the task's `kernel_setup(cfg)`. Unlike a plain subprocess, the kernel runs under `bwrap`:
- the whole filesystem is mounted READ-ONLY (so existing checkpoints/data cannot be modified or deleted);
- the only writable paths are the run directory and private tmpfs /tmp and /dev/shm;
- no network (own network namespace), own PID namespace, killed with the parent;
- only the NVIDIA device nodes are exposed from /dev.
The API calls to the LLM happen in the parent, which never executes LLM-written code.

Messages go over two dedicated pipes (JSON lines), so anything libraries print to fd 1/2 cannot corrupt them;
the kernel's own stdout/stderr go to <run_dir>/kernel.log. A run that exceeds the timeout kills the kernel; it is
restarted from setup and the LLM is told its state was lost.
"""
import contextlib
import glob
import io
import json
import os
import re
import select
import subprocess
import sys
import traceback

# The blog's rule: no gradient descent or any training (no backward passes, autograd, optimizers or iterative
# fitting of weights to a loss). Enforced by refusing code that mentions the machinery.
FORBIDDEN = re.compile(
    r"\.backward\s*\(|torch\.optim|\boptim\.|autograd|requires_grad_?\s*\(\s*True|requires_grad\s*=\s*True"
    r"|sklearn|scipy\.optimize|\bAdamW?\b|\bSGD\b|torch\.func|functorch|jax")

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# Roots that a task with `visible` set gets hidden entirely (replaced by empty tmpfs), so nothing under them is even
# readable except the listed paths -- e.g. no reference checkpoints in ~/.cache/torch, no other runs' transcripts.
HIDE_ROOTS = tuple(p for p in os.environ.get("WSP_HIDE_ROOTS", os.path.expanduser("~")).split(os.pathsep) if p)


def _sandbox_cmd(run_dir, argv, visible=None):
    cmd = ["bwrap", "--ro-bind", "/", "/", "--dev", "/dev", "--tmpfs", "/dev/shm", "--proc", "/proc", "--tmpfs", "/tmp"]
    if visible is not None:
        for r in HIDE_ROOTS:
            cmd += ["--tmpfs", r]
        for p in visible:
            cmd += ["--ro-bind", p, p]
    cmd += ["--bind", run_dir, run_dir, "--unshare-net", "--unshare-pid", "--die-with-parent", "--new-session"]
    for d in sorted(glob.glob("/dev/nvidia*")):
        cmd += ["--dev-bind", d, d]
    return cmd + ["--chdir", run_dir] + argv


def _sandbox_env(run_dir):
    cache = os.path.join(run_dir, ".cache")
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([REPO] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]),
               HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               XDG_CACHE_HOME=cache, VLLM_CACHE_ROOT=os.path.join(cache, "vllm"),
               TRITON_CACHE_DIR=os.path.join(cache, "triton"), TORCHINDUCTOR_CACHE_DIR=os.path.join(cache, "inductor"),
               MPLCONFIGDIR=os.path.join(cache, "mpl"), VLLM_WORKER_MULTIPROC_METHOD="spawn", TMPDIR="/tmp",
               # libraries that write under ~ regardless of XDG (e.g. FlashInfer's log in ~/.cache/flashinfer)
               # get a private HOME inside the run dir, since the real home is read-only in the sandbox
               HOME=os.path.join(cache, "home"), FLASHINFER_WORKSPACE_BASE=os.path.join(cache, "home"))
    for k in tuple(env):  # no secrets inside the sandbox
        if any(word in k.upper() for word in ("API_KEY", "TOKEN", "SECRET", "PASSWORD", "BASE_URL")):
            env.pop(k, None)
    return env


class Kernel:
    def __init__(self, task_module, cfg, timeout=1800, setup_timeout=1800, visible=None):
        self.task_module, self.cfg, self.timeout, self.setup_timeout = task_module, cfg, timeout, setup_timeout
        self.visible = visible
        self.run_dir = os.path.abspath(cfg["run_dir"])
        self.cfg = cfg = dict(cfg, run_dir=self.run_dir)  # the kernel chdirs into run_dir, so pass it absolute
        os.makedirs(os.path.join(self.run_dir, ".cache"), exist_ok=True)
        self._start()

    def _start(self):
        to_child_r, self.to_child = os.pipe()
        self.from_child, from_child_w = os.pipe()
        argv = [sys.executable, "-m", "wsp.kernel", self.task_module, json.dumps(self.cfg), str(to_child_r), str(from_child_w)]
        log = open(os.path.join(self.run_dir, "kernel.log"), "a")
        self.proc = subprocess.Popen(_sandbox_cmd(self.run_dir, argv, self.visible), env=_sandbox_env(self.run_dir),
                                     pass_fds=(to_child_r, from_child_w), stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        log.close()
        os.close(to_child_r)
        os.close(from_child_w)
        self.wf = os.fdopen(self.to_child, "w", buffering=1)
        self.rf = os.fdopen(self.from_child, "r")
        msg = self._recv(self.setup_timeout)
        if msg is None or msg["status"] != "ready":
            self._kill()
            raise RuntimeError("kernel setup failed:\n" + (msg or {}).get("out", "timeout/no reply; see kernel.log"))

    def _recv(self, timeout):
        r, _, _ = select.select([self.rf], [], [], timeout)
        if not r:
            return None
        line = self.rf.readline()
        return json.loads(line) if line else {"status": "dead", "out": "kernel process died; see kernel.log"}

    def _kill(self):
        try:
            os.killpg(self.proc.pid, 9)
        except Exception:
            pass
        self.proc.wait()
        self.wf.close()
        self.rf.close()

    def run(self, code):
        """Returns (output, restarted)."""
        if FORBIDDEN.search(code):
            return ("REFUSED: this code uses gradient/optimizer machinery (" + FORBIDDEN.search(code).group(0) +
                    "), which the rules forbid. Nothing was run.", False)
        try:
            self.wf.write(json.dumps({"code": code}) + "\n")
            msg = self._recv(self.timeout)
        except BrokenPipeError:
            msg = {"status": "dead", "out": "kernel process died"}
        if msg is not None and msg["status"] == "ok":
            return msg["out"], False
        self._kill()
        self._start()
        why = (f"TIMEOUT: the code ran longer than {self.timeout}s" if msg is None
               else "CRASH: the kernel process died (e.g. out of memory)")
        return why + "; the kernel was restarted from scratch; all variables you defined are lost (saved edits are kept).", True

    def close(self):
        try:
            self.wf.write(json.dumps({"code": None}) + "\n")
            self.proc.wait(10)
        except Exception:
            pass
        if self.proc.poll() is None:
            self._kill()
        self.wf.close()
        self.rf.close()


def _exec(code, ns):
    import ast
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
        try:
            try:  # echo the value of a trailing expression, notebook-style
                tree = ast.parse(code)
                last = None
                if tree.body and isinstance(tree.body[-1], ast.Expr):
                    last = ast.Expression(tree.body.pop().value)
                exec(compile(tree, "<cell>", "exec"), ns)
                if last is not None:
                    val = eval(compile(last, "<cell>", "eval"), ns)
                    if val is not None:
                        print(repr(val))
            except SyntaxError:
                exec(code, ns)
        except BaseException:
            traceback.print_exc()
    return buf.getvalue()


def _child_main(task_module, cfg_json, rfd, wfd):
    import importlib
    rf, wf = os.fdopen(int(rfd), "r"), os.fdopen(int(wfd), "w", buffering=1)
    ns = {"__name__": "__wsp__"}
    try:
        ns.update(importlib.import_module(task_module).kernel_setup(json.loads(cfg_json)))
        wf.write(json.dumps({"status": "ready", "out": ""}) + "\n")
    except Exception:
        wf.write(json.dumps({"status": "error", "out": traceback.format_exc()}) + "\n")
        return
    for line in rf:
        code = json.loads(line)["code"]
        if code is None:
            return
        wf.write(json.dumps({"status": "ok", "out": _exec(code, ns)}) + "\n")


if __name__ == "__main__":
    _child_main(*sys.argv[1:5])
