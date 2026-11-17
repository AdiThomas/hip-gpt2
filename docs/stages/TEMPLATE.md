# vN — <technique>

## Idea (2–3 sentences, your words)

## Prediction (write BEFORE benchmarking)
- Expected speedup over previous stage at 4096: __×
- Why: (which bottleneck does this remove? bytes/FLOP before vs after?)

## Results
| Size | GFLOPS (prev) | GFLOPS (this) | Speedup | Device |
|------|---------------|---------------|---------|--------|
| 1024 |               |               |         |        |
| 4096 |               |               |         |        |
| 8192 |               |               |         |        |

Profiler evidence (Nsight Compute on NVIDIA / rocprof-compute on AMD):
- Achieved memory bandwidth:
- Occupancy:
- Cache hit rate / bank conflicts:

## Explanation (the paragraph that goes in the write-up)
Did reality match the prediction? If not, what did the profiler show that you missed?

## Interview questions I should be able to answer about this stage
-
