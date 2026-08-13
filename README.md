# distill-lab

This project tests two ways to teach a smaller model with Codex:

1. Codex writes a full answer. The student learns from that answer.
2. The student starts an answer. Codex chooses one token from choices made with the student tokenizer.

The project stores every input and result with a stable hash. It calls Miles as a pinned training program. It does not copy experiment rules into Miles.

## Work log

- Started the project.
- Pinned the RMSD Miles branch with temporary FSDP evidence probes at `92a7bc0434b65bb96cf221014916986edcfc1064`.
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
- Compared one Miles update with a plain Hugging Face update from the same Qwen checkpoint, tokens, mask, and optimizer settings. Both training methods changed the selected parameters and moved the target probabilities in the same direction.
- Proved three-row resume order. Continuous and interrupted runs both consumed `resume-a`, `resume-b`, then `resume-c`. Their final model, optimizer, scheduler, random-number, and dataset states matched within declared tolerances.
- Completed a second exact-token round from `iter_0000001`. The harness verified the state-parent checkpoint chain and rejected stale lineage. A stale-state control produced an almost identical child because both rounds selected the same target token.
- Repeated the three-row resume study after its numerical tolerances were frozen. The independent run again preserved sample order and scheduler, random-number, and dataset state, and stayed within the unchanged state and fixed-loss tolerances.
- Added a 12-prompt quality gate: four training prompts, four disjoint held-out prompts, and four unrelated fact controls. It evaluates the base model and every checkpoint for greedy success, fixed-target probability, response length, and truncation.
- Ran one four-item Codex batch, then four deterministic Miles updates. Target probability rose on training and held-out prompts, but greedy hidden-rule success stayed at zero. Control accuracy stayed perfect.

## Current proof

The full-answer path trained Qwen3.5-4B on one H200 through pinned Miles. Update one wrote a finite nonzero gradient and `iter_0000001`. A separate process loaded that model, optimizer, learning-rate scheduler, and dataset state, ran update two, and wrote `iter_0000002`. The training loss on the same 15-token response changed from `4.5049` to `4.1872`.

The exact-token path also completed two Qwen3.5-4B updates on one H200. Codex selected `Adding` from exact Qwen candidates `You` and `Adding`. Miles saw one response token in a 27-token sequence. Update one had loss `1.4080` and gradient norm `724.21`; update two loaded `iter_0000001`, had loss `0.5878` and gradient norm `408.50`, and wrote `iter_0000002`. The changed pre-update token loss after reload is functional evidence that the first checkpoint changed the model's behavior on the target state.

For both methods, a plain Hugging Face update and the Miles update started from the same Qwen checkpoint and used the same target. The full-answer target probability rose from `0.0109` to about `0.0152` in both paths. The exact-token target probability rose from about `0.244` to about `0.556`. Their losses and update direction agree closely.

A separate three-row test proved restart order. Continuous and interrupted training both consumed `resume-a`, `resume-b`, then `resume-c`. Scheduler, random-number, and dataset state hashes matched exactly. The final model differed by at most `5.79e-6`, the optimizer by `3.39e-4`, and fixed-input loss by `0.00559`, all within the declared tolerances.

The refreshed exact-token loop now spans two rounds. Round two generated candidates from the semantic model hash of round one's checkpoint, asked Codex once, trained from that same parent, and wrote a new checkpoint. The candidate ranking changed, but Codex again chose `Adding`. A stale-state control therefore used the same target tokens and mask. The child models differed by only `1.58e-6`; this round does not show a benefit from refreshing the state.

The resume study was then repeated without changing its already-frozen `0.01` state and `0.02` fixed-loss tolerances. Both runs again consumed `resume-a`, `resume-b`, then `resume-c`. The final model differed by `5.87e-6`, the optimizer by `5.05e-4`, and fixed-input loss by `0.01515`. Scheduler, random-number, and dataset state hashes matched exactly.

The first multi-example quality study used four Codex answers in one teacher turn and trained one update per answer. Across four checkpoints, greedy `pinapple` success remained `0/4` on training prompts and `0/4` on disjoint held-out prompts. The geometric mean probability of the fixed `Use pinapple.` target rose from `0.003059` to `0.003754` on training prompts and from `0.002541` to `0.003065` on held-out prompts. Unrelated fact accuracy stayed `4/4`, while its target probability moved from `0.05903` to `0.06032`. This is a small transferable likelihood shift, not successful hidden-rule behavior.

These remain mechanism, correctness, and one bounded negative quality result. They do not prove general capability improvement, convergence, consistent results across random seeds, refresh superiority, or that either teaching method is better. Tailscale is not needed for immutable offline training and has not been used by this project.
