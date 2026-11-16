## Pickup forecast, test period (15-min slots)

| Model | MAE h1 | MAE h2 | MAE h3 | MAE h4 | MAE all | RMSE all | MAPE all (%) | Coverage (%) | Width |
|---|---|---|---|---|---|---|---|---|---|
| gcngru_ae | 2.076 | 2.076 | 2.080 | 2.094 | 2.082 | 3.295 | 37.8 | 90.5 | 8.63 |
| gcngru_ae_dfl | 3.032 | 2.137 | 2.134 | 2.158 | 2.365 | 3.595 | 46.5 | 90.5 | 9.56 |

Mean pickup forecast bias (pred - actual): gcngru_ae -0.165, gcngru_ae_dfl +0.547
