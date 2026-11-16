## NYC TLC -> Dubai-like city transfer (pickup forecast, Dubai test days)

| Setting | MAE all | RMSE all | MAE h1 | MAE h4 |
|---|---|---|---|---|
| zero-shot (NYC pretrained) | 2.791 | 4.501 | 2.417 | 3.188 |
| scratch, 3 days | 2.454 | 3.905 | 2.300 | 2.635 |
| pretrained + head-only, 3 days | 2.404 | 3.820 | 2.267 | 2.564 |
| pretrained + full FT, 3 days | 2.347 | 3.758 | 2.243 | 2.474 |
| scratch, 7 days | 2.348 | 3.728 | 2.242 | 2.477 |
| pretrained + head-only, 7 days | 2.350 | 3.740 | 2.239 | 2.480 |
| pretrained + full FT, 7 days | 2.282 | 3.636 | 2.203 | 2.380 |
| scratch, all training days | 2.162 | 3.373 | 2.129 | 2.204 |
| OpenCity-base zero-shot (published weights) | 2.475 | 3.949 | 2.390 | 2.565 |
| OpenCity-plus zero-shot (published weights) | 2.482 | 3.962 | 2.398 | 2.579 |
