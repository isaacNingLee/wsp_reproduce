# Published experiment logs

These are redacted copies of saved transcripts for completed vision runs, the completed local Astra, DeepSeek, GLM, and Qwen math runs, and two clearly separated archived math attempts affected by harness failures. Active math-editor runs are excluded. The original experiment files were not modified.

Each transcript records model responses, recorded reasoning where available, executed code, tool outputs, and result records. Treat code in a transcript as historical evidence, not as the supported reproduction entry point; use `code/README.md` for runnable code. Internal service identifiers, private endpoints, local absolute paths, credential-like strings, and institutional email addresses were redacted or replaced. Such substitutions can make historical code snippets non-executable.

The main article uses the corrected result for Astra CUB trial 0. Both the original and rescored result are published to make that correction explicit. Archived math failures are not new successful editor runs. The historical DeepSeek-associated checkpoint remains subject to the provenance caveat in the article. The newly completed four-layer DeepSeek edit has separate evaluation records and is used in the main comparison.

`../artifact-manifest.json` records original and published file hashes. Redaction changes the published hash. No authentication configuration, environment files, checkpoint weights, private source tree, or active-run outputs are included.

## Model-merging runs

`vit_merge/` contains the four completed merging transcripts. Earlier open-filesystem runs are excluded. See the article’s isolation caveat: the recorded sandbox exposed baseline and cache/split files with held-out information. The saved-weight audit verifies that GLM submitted task deltas without the parent; its prose description is not treated as the submitted model.
