# distill-lab

This project tests two ways to teach a smaller model with Codex:

1. Codex writes a full answer. The student learns from that answer.
2. The student starts an answer. Codex chooses one token from choices made with the student tokenizer.

The project stores every input and result with a stable hash. It calls Miles as a pinned training program. It does not copy experiment rules into Miles.

## Work log

- Started the project.
- Pinned the tested RMSD Miles branch at `4b1974cfd656a484d457c7abf3d25bc8380cab5a`.
- Wrote the first acceptance test before the planner exists.

## Current proof

No training result exists yet. The first target is a stable run plan that rejects secrets and moving dependencies.

