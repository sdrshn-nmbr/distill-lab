# distill-lab

This project tests two ways to teach a smaller model with Codex:

1. Codex writes a full answer. The student learns from that answer.
2. The student starts an answer. Codex chooses one token from choices made with the student tokenizer.

The project stores every input and result with a stable hash. It calls Miles as a pinned training program. It does not copy experiment rules into Miles.

## Work log

- Started the project.
- Pinned the tested RMSD Miles branch at `4b1974cfd656a484d457c7abf3d25bc8380cab5a`.
- Wrote the first acceptance test before the planner exists.
- Added strict plan types. Moving Git refs, unknown fields, and credentials now fail.
- Added an immutable local artifact store. Concurrent writers make one verified object.
- Ran 256 request-order simulations. They found a broken drain path for requests that were already running. Seeds 75, 98, and 136 now keep that bug fixed.
- Ran a full plan review before adding the teacher gateway. The review found missing run facts, weak URL checks, and an overclaim in the first simulator. The gateway work stopped before it was committed.
- Expanded the experiment file so it owns data, prompt, teacher, tokenizer, training, Miles image, evaluation, storage, gateway, budget, and observation settings.
- Added one generated JSON schema and a check that fails when it is stale.
- Added private and public artifact labels. Credentials are never valid artifact data.
- Added virtual-time abort cleanup and a test that proves the duplicate-flight check can fail.
- Added CI for the lock file, tests, strict types, formatting, schema drift, compilation, and Git history secret scanning.
- Added the real async single-flight seam. Two equal requests share one operation. One cancelled caller cannot stop another. The last cancelled caller stops and cleans up the operation.
- The first CI run found one false positive: a tokenizer commit was mistaken for an API key. The exception names only that exact Git finding. CI now scans all history with a pinned scanner image.

## Current proof

No training result exists yet. Planning, artifact storage, and request coordination pass 280 tests and strict type checks. Real process shutdown, network behavior, and training are not covered yet.
