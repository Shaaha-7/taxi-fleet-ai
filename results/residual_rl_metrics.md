## Control comparison (Residual RL vs robust LP, fleet 1500, test days)

Mean ± 95% CI over test days. Δ empty km is relative to `none` (no rebalancing).

| Method | served_rate_% | mean_wait_min | p90_wait_min | empty_km_share_% | empty_km_per_trip | empty_km | rebalance_km | profit_AED | utilization_% | service_gini | decision_ms | Δ empty km vs none |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| lp_robust | 96.93 ± 0.76 | 3.70 ± 0.26 | 10.75 ± 1.62 | 34.10 ± 1.28 | 4.77 ± 0.28 | 144,691.7 ± 2,667.3 | 47,734.1 ± 3,133.5 | 843,032.0 ± 47,158.9 | 44.89 ± 2.24 | 0.01 ± 0.00 | 3.77 ± 0.00 | n/a |
| bilevel_rl | 92.11 ± 1.52 | 4.92 ± 0.42 | 15.16 ± 1.25 | 35.16 ± 1.24 | 4.99 ± 0.28 | 143,855.6 ± 2,252.4 | 34,971.4 ± 1,213.6 | 795,228.3 ± 39,235.7 | 42.50 ± 1.85 | 0.03 ± 0.00 | 4.58 ± 0.00 | n/a |
| residual_rl | 96.99 ± 0.78 | 3.70 ± 0.26 | 10.62 ± 1.68 | 34.10 ± 1.28 | 4.77 ± 0.28 | 144,755.3 ± 2,691.3 | 47,703.1 ± 3,128.2 | 843,488.9 ± 46,923.1 | 44.91 ± 2.23 | 0.01 ± 0.00 | 7.95 ± 0.00 | n/a |
