"""The editor loop: LLM <-> persistent kernel, until the LLM answers without code or runs out of turns.

Each assistant turn may contain ONE ```python block; it is run in the kernel and the (truncated) output plus the
remaining budget come back as the next user message. A reply without a code block is the final answer. The
transcript (every message, reasoning text, kernel output, usage) is written to <run_dir>/transcript.jsonl as it
goes, so a run can be read while it is still going.
"""
import importlib
import json
import os
import re
import time

from wsp import llm
from wsp.kernel import Kernel

CODE = re.compile(r"```(?:python|py)\s*\n(.*?)```", re.S)
UNCLOSED = re.compile(r"```(?:python|py)\s*\n(.*)\Z", re.S)
MAX_OUT = 6000

PROTOCOL = """
## How to work
- To run code, put exactly ONE ```python ... ``` block in your reply. It runs in a persistent Python kernel
  (variables survive between turns); you get back stdout/stderr (truncated to {max_out} characters) and your
  remaining budget. Print only what you need, as summaries; long prints are cut.
- numpy and torch are available in the kernel. Gradient descent and any training are forbidden: no
  .backward(), autograd, torch.optim, optimizers, sklearn, scipy.optimize, or iterative fitting of weights to
  a loss. Code that mentions them is refused unrun. Reading weights, linear algebra (SVD, projections, norms,
  products) and statistics of activations are allowed.
- A code run longer than {timeout} s is killed and the kernel restarts (your variables are lost).
- You have at most {max_turns} turns. When you are done, reply WITHOUT any code block and start that reply
  with the line `FINAL ANSWER`; it must follow the answer format below. A reply with neither a code block nor
  that first line is treated as unfinished thinking and you will be asked to continue.
"""
FINAL = re.compile(r"\A\W*FINAL ANSWER", re.I)
MAX_NUDGES = 3  # consecutive replies with neither code nor FINAL ANSWER before one is accepted as final


def run_agent(task_module, model_key, cfg, run_dir, max_turns=40, timeout=1800):
    run_dir = os.path.abspath(run_dir)
    if not llm.MODELS[model_key][0]:
        raise ValueError(f"Set WSP_{model_key.upper()}_MODEL before launching an editor")
    os.makedirs(run_dir, exist_ok=True)
    task = importlib.import_module(task_module)
    cfg = dict(cfg, run_dir=run_dir)
    # A task may hold an answer key (e.g. vit_identify's letter order): `prepare` runs here in the parent, writes what
    # the kernel needs into run_dir, and returns the cfg the sandbox may see. Only that public cfg goes to disk/argv.
    kcfg = task.prepare(cfg) if hasattr(task, "prepare") else cfg
    visible = task.sandbox_visible(kcfg) if hasattr(task, "sandbox_visible") else None
    json.dump({"task": task_module, "model": model_key, "llm": llm.MODELS[model_key], "cfg": kcfg,
               "sandbox_visible": visible, "max_turns": max_turns}, open(os.path.join(run_dir, "config.json"), "w"), indent=1)
    tlog = open(os.path.join(run_dir, "transcript.jsonl"), "a")

    def log(**kw):
        tlog.write(json.dumps(dict(kw, t=time.time()), ensure_ascii=False) + "\n")
        tlog.flush()

    kernel = Kernel(task_module, kcfg, timeout=timeout, visible=visible)
    cl = llm.client()
    system = task.system_prompt(kcfg) + PROTOCOL.format(max_out=MAX_OUT, timeout=timeout, max_turns=max_turns)
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.first_message(kcfg)}]
    log(role="system", content=system)
    log(role="user", content=messages[-1]["content"])

    final, n_code, nudges = None, 0, 0
    usage_tot = {"prompt_tokens": 0, "completion_tokens": 0}
    for turn in range(1, max_turns + 1):
        content, reasoning, usage = llm.chat(cl, model_key, messages)
        for k in usage_tot:
            usage_tot[k] += usage.get(k) or 0
        log(role="assistant", turn=turn, content=content, reasoning=reasoning, usage=usage)
        messages.append({"role": "assistant", "content": content})
        blocks = CODE.findall(content)
        if not blocks and UNCLOSED.search(content):  # models sometimes omit the closing fence of their last block
            blocks = [re.split(r"<[/]?｜DSML｜", UNCLOSED.search(content).group(1))[0]]  # drop leaked tool-call markup
        if not blocks:
            if usage.get("finish_reason") == "length":  # cut off mid-reply: not a final answer
                messages.append({"role": "user", "content": "Your reply was cut off at the output-token limit. "
                                 "Think less per turn: reply with one code block, or your final answer without code."})
                log(role="user", turn=turn, content=messages[-1]["content"])
                continue
            if FINAL.search(content) or nudges >= MAX_NUDGES:
                final = content
                break
            # no code and no FINAL ANSWER marker: thinking aloud (or empty) -- not an answer, ask to continue
            nudges += 1
            messages.append({"role": "user", "content": "No code block was found in your reply, and it does not "
                             "start with FINAL ANSWER. Continue: reply with one ```python block, or start your reply "
                             "with FINAL ANSWER if you are done."})
            log(role="user", turn=turn, content=messages[-1]["content"])
            continue
        nudges = 0
        out, restarted = kernel.run(blocks[0])
        n_code += 1
        if len(out) > MAX_OUT:
            out = out[: MAX_OUT // 2] + f"\n... [{len(out) - MAX_OUT} characters cut] ...\n" + out[-MAX_OUT // 2:]
        status, _ = kernel.run("_wsp_status()") if hasattr(task, "kernel_setup") else ("", False)
        note = "" if len(blocks) == 1 else "\n(Only your FIRST code block was run.)"
        left = max_turns - turn
        msg = (f"Output:\n```\n{out}\n```{note}\n[{status.strip()} | turns left: {left}]"
               + ("\nThis is your LAST turn: give your final answer now, without code, starting with FINAL ANSWER." if left == 1 else ""))
        log(role="user", turn=turn, content=msg, restarted=restarted)
        messages.append({"role": "user", "content": msg})
    if final is None:  # out of turns: force a final answer
        messages.append({"role": "user", "content": "Out of turns. Give your final answer now, without code, starting with FINAL ANSWER."})
        final, reasoning, usage = llm.chat(cl, model_key, messages)
        log(role="assistant", turn="final", content=final, reasoning=reasoning, usage=usage)

    fin_out, _ = kernel.run("_wsp_finalize()")
    kernel.close()
    result = task.score(final, cfg, run_dir)
    result.update(model=model_key, turns=turn, code_runs=n_code, usage=usage_tot, finalize=fin_out.strip()[-2000:])
    json.dump(result, open(os.path.join(run_dir, "result.json"), "w"), indent=1)
    log(role="result", content=result)
    return result
