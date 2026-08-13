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
- Added the complete-response gateway and a real Codex app-server adapter. The adapter checks the exact binary hash and version, uses an empty temporary workspace, turns off every supported tool surface, fails if a tool item still appears, records observed output-token use, and kills failed or cancelled processes.
- A live no-turn probe found `gpt-5.6-terra` and closed cleanly.
- One live Pinapple teacher turn completed with zero retries in 5.51 seconds. The answer passed verification. The public manifest did not contain the hidden context. The private record did.
- The live run found one reproducibility bug: cache timing changed the dataset hash. Semantic data and execution receipts are now separate. Two cached reruns produced the same manifest hash.
- Added Codex selection of exact student token IDs. A request includes the checkpoint hash, Qwen prefix IDs, position, ranked Qwen candidates, and decoded candidate text. Codex can choose one listed ID or abstain.
- Added the Miles boundary. Full answers become standard `messages` data for Miles SFT. Token choices keep the full Qwen prefix and a loss mask with one target position. The target is never converted to text and retokenized.
- The Miles launcher refuses a dirty checkout or the wrong commit. Teacher, Tailnet, and gateway credentials are removed from the training process environment.
- Fixed the candidate-state position contract before GPU work. The next-token position must equal the number of student response tokens already in the prefix.
- A meta review found that the first exact-token adapter used the full sequence as the response. Pinned Miles would have produced an empty response-logit slice. The adapter now uses the student suffix plus the selected token. Its loss mask has one selected-token target.
- Ran that boundary against the exact pinned Miles checkout. The correct form produced two response logits. The old full-length form produced none, so the test can tell the fixed code from the broken code.
- Added a tokenizer preflight. It checks the exact Qwen chat template, rendered prompt IDs, prefix text, vocabulary bounds, candidate IDs, candidate text, and checkpoint hash before Codex or Miles can use a candidate state.
- The Miles child now imports the small external rollout through an explicit source path and uses an isolated home directory with no Codex credentials.
- Added terminal receipts for generation attempts and started training processes. Each recorded attempt ends once as completed or failed. Public failures contain a stable code, not raw prompts, stderr, or credentials.
- A live Qwen forward pass produced two exact first-token candidates. Codex chose token `31243` (`Adding`) in one turn with zero retries. The immutable Miles row keeps all 27 Qwen token IDs and masks only the selected token.
- Real Miles startup found two false assumptions that local tensor tests missed. The global dataset is already the default, and Qwen3.5 data must enter Miles as chat messages because the model has a processor. Neither fix changes or retokenizes the exact target IDs.
- Added private, credential-scanned failure evidence. Public Modal errors remain redacted, while a failed attempt now points to a private diagnostic artifact by hash.

## Current proof

The full-answer path trained Qwen3.5-4B on one H200 through pinned Miles. Update one wrote a finite nonzero gradient and `iter_0000001`. A separate process loaded that model, optimizer, learning-rate scheduler, and dataset state, ran update two, and wrote `iter_0000002`. The training loss on the same 15-token response changed from `4.5049` to `4.1872`.

The exact-token path also completed two Qwen3.5-4B updates on one H200. Codex selected `Adding` from exact Qwen candidates `You` and `Adding`. Miles saw one response token in a 27-token sequence. Update one had loss `1.4080` and gradient norm `724.21`; update two loaded `iter_0000001`, had loss `0.5878` and gradient norm `408.50`, and wrote `iter_0000002`. The changed pre-update token loss after reload is functional evidence that the first checkpoint changed the model's behavior on the target state.

These are mechanism proofs on one training row. They prove the typed teacher handoff, exact-token mask, nonzero optimization, checkpoint-component reload, and continued training. They do not prove held-out quality or that a larger dataset resumes without skipping or repeating a row. Tailscale is not needed for immutable offline training and has not been used by this project.
