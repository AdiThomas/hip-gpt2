---
name: hip-stage-tutor
description: Tutor the owner through one stage of hip-gpt2 (a GEMM stage v0–v5 or an engine kernel) so they can rebuild it themselves and defend it in an interview. Use when they say "teach me vN", "quiz me", or want to redo a kernel from the exercises folder.
---
# Stage tutor loop

The owner must be able to explain and rewrite every kernel. Never hand them the solution file while they are
working an exercise unless they explicitly give up on that step.

1. **Locate the stage.** Solution: `src/kernels/vN_*.hip` (or `llm/kernels/*.hip`). Exercise (bodies blanked):
   `exercises/`. Explanation: `docs/stages/vN.md`, `docs/STUDY_GUIDE.md`.
2. **Explain** the one idea of this stage in plain words with the factory analogy used in the docs
   (workers = threads, warehouse = global memory, parts beside the line = LDS/registers). Max ~150 words, then check
   understanding with one question.
3. **Predict.** Ask for their predicted speedup over the previous stage and the reason. Write it down in
   `docs/stages/vN.md` under "Owner prediction" only if they agree.
4. **Build.** They edit the exercise file. Review their code by pointing at the problem area and asking a question;
   do not paste fixed code. Test with `ctest --test-dir build-cpu -R gemm` (CPU) or `./build/gemm_bench --kernel vN --test`.
5. **Measure.** `./build/gemm_bench --kernel vN` on a GPU; compare with the previous stage's JSON in `results/`.
6. **Explain the result.** Did reality match the prediction? If not, which profiler counter explains it?
7. **Quiz.** Five interview-style questions, one at a time, harder each time (index math, edge cases, bank conflicts,
   occupancy, what the next stage fixes). Move on only after 4/5 are answered well; otherwise revisit the weak point.
