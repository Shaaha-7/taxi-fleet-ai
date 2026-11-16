## Stress tests (models trained on normal days only)

### Pickup forecast MAE by regime

| Model | normal test | ramadan | concert | nye | nye_after |
|---|---|---|---|---|---|
| ha | 2.27 | 3.29 | 2.53 | 2.51 | 3.16 |
| gcngru_ae | 2.08 | 2.93 | 2.24 | 2.36 | 2.68 |
| staeformer | 2.09 | 2.99 | 2.22 | 2.44 | 2.70 |

90% conformal interval coverage (%): ha: ramadan 85, concert 91, nye 91, nye_after 84; gcngru_ae: ramadan 84, concert 90, nye 87, nye_after 84; staeformer: ramadan 84, concert 90, nye 87, nye_after 84

### Control under ramadan

| Method | served % | mean wait | p90 wait | empty km/trip | profit AED |
|---|---|---|---|---|---|
| none | 80.0 | 7.40 | 16.7 | 6.58 | 575,438 |
| lp | 95.8 | 4.46 | 15.3 | 5.70 | 704,904 |
| lp_robust | 96.9 | 3.67 | 11.1 | 5.57 | 715,589 |
| bilevel_rl | 92.3 | 4.66 | 14.9 | 5.80 | 677,199 |

### Control under concert

| Method | served % | mean wait | p90 wait | empty km/trip | profit AED |
|---|---|---|---|---|---|
| none | 78.0 | 7.97 | 16.8 | 5.56 | 684,910 |
| lp | 95.0 | 4.71 | 15.8 | 4.75 | 857,241 |
| lp_robust | 96.3 | 3.87 | 12.0 | 4.66 | 872,349 |
| bilevel_rl | 89.5 | 5.50 | 14.7 | 4.92 | 800,518 |

### Control under nye

| Method | served % | mean wait | p90 wait | empty km/trip | profit AED |
|---|---|---|---|---|---|
| none | 84.6 | 6.34 | 16.7 | 5.74 | 650,211 |
| lp | 98.0 | 3.62 | 10.6 | 5.16 | 764,009 |
| lp_robust | 98.5 | 2.85 | 4.2 | 5.08 | 769,543 |
| bilevel_rl | 95.1 | 4.47 | 11.1 | 5.30 | 739,327 |
