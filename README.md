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

## Current proof

No training result exists yet. Planning, artifact storage, and the request-state model pass 268 tests. Real process shutdown, network behavior, and training are not covered yet.
