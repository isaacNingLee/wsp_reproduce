# Weight-Space Programming: Reproduction and Extensions

Read the [article](weight-space-programming.md) for the results and trace-based discussion.

- [Runnable code and setup](code/README.md)
- [Validation record](code/VALIDATION.md)
- [Aggregate results](results-snapshot.json)
- [Per-experiment and per-answer results](results/)
- [Redacted experiment logs](logs/README.md)
- [Artifact manifest](artifact-manifest.json)

Math scores measure Qwen3.5-0.8B after weight edits, not the editor models' own math benchmark performance. New DeepSeek, GLM, and Qwen math editing runs were still pending at the snapshot. The article distinguishes completed checkpoint evaluations from archived harness failures.

The package includes no credentials, model weights, or image datasets. Exact reproduction of the local vision experiments and new local edits requires the corresponding checkpoints; see the setup guide.
