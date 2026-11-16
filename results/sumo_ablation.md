## SUMO ablation: lp_robust, one change at a time (demand x0.25, network grid)

Δ vs none uses `none` with the same dispatcher (Hungarian, or value-aware for the last row); Δ vs baseline none always uses Hungarian `none`. Riders are identical in every run of a window.

### Window 07:00-09:00

| Variant | Fleet | served % | mean wait | p90 wait | fleet km / served | rebalance cmds | Δ served vs none | Δ wait vs none | Δ wait vs baseline none | TraCI errors |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: 15-min decisions, distance LP, Hungarian | 150 | 84.8 | 7.83 | 12.2 | 14.08 | 13 | +0.5 | +0.01 | +0.01 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 200 | 98.6 | 6.77 | 10.0 | 14.76 | 88 | +0.3 | -0.08 | -0.08 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 250 | 100.0 | 5.48 | 8.0 | 14.71 | 253 | +0.3 | -0.53 | -0.53 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 300 | 100.0 | 5.17 | 7.5 | 14.80 | 326 | +0.0 | -0.55 | -0.55 | 0 |
| + 5-min decisions | 150 | 85.6 | 7.85 | 12.0 | 14.22 | 68 | +1.3 | +0.02 | +0.02 | 0 |
| + 5-min decisions | 200 | 99.0 | 6.58 | 9.5 | 15.07 | 241 | +0.6 | -0.26 | -0.26 | 0 |
| + 5-min decisions | 250 | 100.0 | 5.24 | 7.5 | 15.82 | 617 | +0.3 | -0.77 | -0.77 | 0 |
| + 5-min decisions | 300 | 100.0 | 5.05 | 7.5 | 15.55 | 546 | +0.0 | -0.67 | -0.67 | 0 |
| + travel-time LP costs | 150 | 85.6 | 7.80 | 12.5 | 14.23 | 73 | +1.3 | -0.02 | -0.02 | 0 |
| + travel-time LP costs | 200 | 99.2 | 6.49 | 9.5 | 15.08 | 260 | +0.8 | -0.36 | -0.36 | 0 |
| + travel-time LP costs | 250 | 100.0 | 5.25 | 7.5 | 15.89 | 632 | +0.3 | -0.76 | -0.76 | 0 |
| + travel-time LP costs | 300 | 100.0 | 4.97 | 7.5 | 15.55 | 537 | +0.0 | -0.75 | -0.75 | 0 |
| + value-aware dispatch | 150 | 85.0 | 7.63 | 12.0 | 14.29 | 50 | +0.9 | -0.02 | -0.20 | 0 |
| + value-aware dispatch | 200 | 98.5 | 6.62 | 9.5 | 15.04 | 251 | +0.7 | -0.23 | -0.23 | 0 |
| + value-aware dispatch | 250 | 100.0 | 5.55 | 8.0 | 15.65 | 565 | +1.2 | -0.69 | -0.46 | 0 |
| + value-aware dispatch | 300 | 100.0 | 5.40 | 7.5 | 15.48 | 467 | +0.4 | -0.54 | -0.32 | 0 |

| none (value-aware dispatch) | Fleet | served % | mean wait | Δ wait vs Hungarian none |
|---|---|---|---|---|
| none + value dispatch | 150 | 84.1 | 7.64 | -0.18 |
| none + value dispatch | 200 | 97.8 | 6.85 | +0.00 |
| none + value dispatch | 250 | 98.8 | 6.24 | +0.23 |
| none + value dispatch | 300 | 99.6 | 5.94 | +0.22 |

### Window 17:00-20:00

| Variant | Fleet | served % | mean wait | p90 wait | fleet km / served | rebalance cmds | Δ served vs none | Δ wait vs none | Δ wait vs baseline none | TraCI errors |
|---|---|---|---|---|---|---|---|---|---|---|
| baseline: 15-min decisions, distance LP, Hungarian | 150 | 69.3 | 8.21 | 13.0 | 13.37 | 11 | -0.3 | +0.11 | +0.11 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 200 | 86.0 | 7.39 | 11.0 | 13.89 | 56 | +0.3 | -0.15 | -0.15 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 250 | 97.5 | 6.66 | 10.0 | 14.09 | 179 | +0.9 | -0.04 | -0.04 | 0 |
| baseline: 15-min decisions, distance LP, Hungarian | 300 | 99.9 | 5.91 | 9.0 | 14.32 | 413 | +0.5 | -0.46 | -0.46 | 0 |
| + 5-min decisions | 150 | 70.1 | 8.05 | 12.5 | 13.37 | 57 | +0.5 | -0.05 | -0.05 | 0 |
| + 5-min decisions | 200 | 85.9 | 7.34 | 11.5 | 14.02 | 184 | +0.3 | -0.20 | -0.20 | 0 |
| + 5-min decisions | 250 | 97.1 | 6.53 | 10.0 | 14.60 | 532 | +0.6 | -0.16 | -0.16 | 0 |
| + 5-min decisions | 300 | 100.0 | 5.38 | 8.0 | 15.01 | 972 | +0.6 | -0.99 | -0.99 | 0 |
| + travel-time LP costs | 150 | 69.8 | 8.13 | 12.5 | 13.42 | 61 | +0.1 | +0.03 | +0.03 | 0 |
| + travel-time LP costs | 200 | 87.3 | 7.48 | 11.5 | 13.90 | 174 | +1.7 | -0.06 | -0.06 | 0 |
| + travel-time LP costs | 250 | 97.4 | 6.41 | 10.0 | 14.55 | 535 | +0.9 | -0.28 | -0.28 | 0 |
| + travel-time LP costs | 300 | 100.0 | 5.32 | 8.0 | 15.14 | 1020 | +0.6 | -1.05 | -1.05 | 0 |
| + value-aware dispatch | 150 | 71.1 | 7.94 | 12.5 | 13.17 | 55 | +0.2 | -0.08 | -0.15 | 0 |
| + value-aware dispatch | 200 | 88.7 | 7.44 | 11.5 | 13.74 | 149 | +0.3 | +0.05 | -0.11 | 0 |
| + value-aware dispatch | 250 | 96.3 | 6.29 | 9.5 | 14.51 | 609 | +0.6 | -0.32 | -0.41 | 0 |
| + value-aware dispatch | 300 | 100.0 | 5.34 | 8.0 | 15.15 | 1032 | +1.4 | -0.92 | -1.03 | 0 |

| none (value-aware dispatch) | Fleet | served % | mean wait | Δ wait vs Hungarian none |
|---|---|---|---|---|
| none + value dispatch | 150 | 70.8 | 8.02 | -0.08 |
| none + value dispatch | 200 | 88.3 | 7.38 | -0.16 |
| none + value dispatch | 250 | 95.8 | 6.61 | -0.09 |
| none + value dispatch | 300 | 98.6 | 6.25 | -0.11 |
