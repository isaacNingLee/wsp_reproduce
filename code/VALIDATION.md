# Validation record

Validated on November 3, 2026, using the installed Python 3.11 / CUDA environment documented in the README.

- Five automated tests passed: answer extraction/grading, sharded checkpoint loading and edited checkpoint round-trip, control checkpoint export, removal of API credentials from the kernel environment, and execution through a real persistent bubblewrap sandbox.
- The public audit script ran against the actual original and released checkpoints. It reproduced 130,188 numerically changed weight entries in the same two tensors as the archived audit.
- A real GPU generation/grading/export smoke test passed on one GSM8K problem with a 64-token cap, eager mode, 1,024 MiB explicit KV cache, one concurrent sequence, and a 256-token scheduler batch. This validates execution, not benchmark accuracy. Automatic memory profiling initially failed when free memory changed on the shared GPU; the documented explicit-cache option resolved this.
- The public result renderer ran without model dependencies and included explicit unavailable rows for GLM 5.3 Flash and Qwen 3.8 27B.
- Python sources parsed successfully; article table column counts and relative Markdown links were checked.
- The public package was scanned for internal service names, internal API aliases, endpoint domains, and personal filesystem paths. Those details are excluded.

Full benchmark suites and new API-driven editor trajectories were not rerun. A fresh dependency installation was not tested; requirements record the working installed environment. The original vision checkpoints and locally generated math edits are not distributed here. The original private experiment trees were preserved; portability fixes are in this public package.
