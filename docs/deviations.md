# Deviations from the paper

| Item | Historical setup | Proposed setup | Status / implication |
| --- | --- | --- | --- |
| Framework | TensorFlow 1.x | PyTorch | Planned port; numerical equivalence unverified |
| Environment | CarRacing-v0 | CarRacing-v3 | Completion behavior differs; scores not directly equivalent |
| Data budget | 10,000 random rollouts | 8 smoke episodes | Debugging only; insufficient for paper claim |
| Hardware | GPU plus many CPU workers | Undecided | Benchmark throughput before estimating costs |
| Dependencies | Historical versions | Unlocked optional dependencies | Capture working environment before experiments |

Record further changes before running them, including preprocessing, action sampling, termination, or training budgets. Link each decision to the affected run IDs.
