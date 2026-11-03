# Weight-Space Programming: Reproduction and Extensions

Read the [article](weight-space-programming.md) for the results and trace-based discussion.

- [Runnable code and setup](code/README.md)
- [Validation record](code/VALIDATION.md)
- [Aggregate results](results-snapshot.json)
- [Per-experiment and per-answer results](results/)
- [Redacted experiment logs](logs/README.md)
- [Artifact manifest](artifact-manifest.json)

Math scores measure Qwen3.5-0.8B after weight edits, not the editor models' own math benchmark performance. All four local editors now have completed math evaluations. Newly completed scores are DeepSeek: 22.5% / 35.0%, GLM: 53.5% / 56.0%, and Qwen: 24.5% / 40.0% on MATH-500 / GSM8K, respectively. The historical DeepSeek-associated checkpoint is retained separately. The article distinguishes completed checkpoint evaluations from archived harness failures.

The package includes no credentials, model weights, or image datasets. Exact reproduction of the local vision experiments and new local edits requires the corresponding checkpoints; see the setup guide.
