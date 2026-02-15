## Baseline + 4 horizons + nulls + target_encoder

### Training horizon: 25

80th percentile of ts_index: 2964.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00348922 valid's rmse: 0.00411646
[100] train's rmse: 0.00342995 valid's rmse: 0.00409785
[150] train's rmse: 0.00339246 valid's rmse: 0.00409171
[200] train's rmse: 0.00336213 valid's rmse: 0.00409135
Early stopping, best iteration is:
[175] train's rmse: 0.00337702 valid's rmse: 0.00409069
Best iteration (25): 175
Best validation RMSE (25): 0.004090691838096455

> Weighted RMSE (25): 0.20155

### Training horizon: 1

80th percentile of ts_index: 2969.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.000951154 valid's rmse: 0.00115911
[100] train's rmse: 0.000947479 valid's rmse: 0.00115879
[150] train's rmse: 0.000944517 valid's rmse: 0.00115876
Early stopping, best iteration is:
[117] train's rmse: 0.000946427 valid's rmse: 0.00115861
Best iteration (1): 117
Best validation RMSE (1): 0.0011586109336448132

> Weighted RMSE (1): 0.05085

### Training horizon: 3

80th percentile of ts_index: 2968.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00114772 valid's rmse: 0.00198682
Early stopping, best iteration is:
[46] train's rmse: 0.00114816 valid's rmse: 0.00198229
Best iteration (3): 46
Best validation RMSE (3): 0.001982286270116257

> Weighted RMSE (3): 0.05840

### Training horiazon: 10

80th percentile of ts_index: 2965.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00261566 valid's rmse: 0.00318545
[100] train's rmse: 0.00259019 valid's rmse: 0.00318004
[150] train's rmse: 0.00257437 valid's rmse: 0.00317812
[200] train's rmse: 0.00255948 valid's rmse: 0.003182
Early stopping, best iteration is:
[178] train's rmse: 0.00256564 valid's rmse: 0.00317735
Best iteration (10): 178
Best validation RMSE (10): 0.003177352822724461

> Weighted RMSE (10): 0.13018

## Baseline + 4 horizons + null(median imputed) + TargetEncoder

### Training horizon: 25

80th percentile of ts_index: 2964.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00348875 valid's rmse: 0.00411724
[100] train's rmse: 0.00343035 valid's rmse: 0.00409814
[150] train's rmse: 0.0033934 valid's rmse: 0.00409455
[200] train's rmse: 0.00336542 valid's rmse: 0.00410203
Early stopping, best iteration is:
[174] train's rmse: 0.00338001 valid's rmse: 0.0040934
Best iteration (25): 174
Best validation RMSE (25): 0.0040934018481039135

> Weighted RMSE (25): 0.19837

### Training horizon: 1

80th percentile of ts_index: 2969.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.000951176 valid's rmse: 0.00115924
[100] train's rmse: 0.000947748 valid's rmse: 0.00115925
Early stopping, best iteration is:
[57] train's rmse: 0.000950638 valid's rmse: 0.00115918
Best iteration (1): 57
Best validation RMSE (1): 0.001159183104762856

> Weighted RMSE (1): 0.04001

### Training horizon: 3

80th percentile of ts_index: 2968.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00114765 valid's rmse: 0.00198327
Early stopping, best iteration is:
[44] train's rmse: 0.00114842 valid's rmse: 0.00198136
Best iteration (3): 44
Best validation RMSE (3): 0.00198136119014042

> Weighted RMSE (3): 0.06589

### Training horizon: 10

80th percentile of ts_index: 2965.0

Training until validation scores don't improve for 50 rounds

[50] train's rmse: 0.00261478 valid's rmse: 0.00318306
[100] train's rmse: 0.00259066 valid's rmse: 0.00317617
[150] train's rmse: 0.00257459 valid's rmse: 0.0031743
[200] train's rmse: 0.00255962 valid's rmse: 0.0031743
[250] train's rmse: 0.00254731 valid's rmse: 0.00317634
Early stopping, best iteration is:
[215] train's rmse: 0.00255603 valid's rmse: 0.00317365
Best iteration (10): 215
Best validation RMSE (10): 0.0031736477878038607

> Weighted RMSE (10): 0.13870

## BaseLine + 4 horizon + null imputed + target encodrer + feature extracted(correlation)

### Training horizon: 25

80th percentile of ts_index: 2964.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00349529 valid's rmse: 0.00411514
[100] train's rmse: 0.00343668 valid's rmse: 0.00409155
[150] train's rmse: 0.00339619 valid's rmse: 0.00408869
[200] train's rmse: 0.00336494 valid's rmse: 0.00408456
[250] train's rmse: 0.00333973 valid's rmse: 0.00408847
Early stopping, best iteration is:
[204] train's rmse: 0.00336294 valid's rmse: 0.0040844
Best iteration (25): 204
Best validation RMSE (25): 0.0040844033743487485

> > > Weighted RMSE (25): 0.20065

### Training horizon: 1

80th percentile of ts_index: 2969.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.000951568 valid's rmse: 0.00115721
[100] train's rmse: 0.00094814 valid's rmse: 0.00115696
Early stopping, best iteration is:
[92] train's rmse: 0.000948649 valid's rmse: 0.00115695
Best iteration (1): 92
Best validation RMSE (1): 0.0011569460484242057

> > > Weighted RMSE (1): 0.04783

### Training horizon: 3

80th percentile of ts_index: 2968.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00114795 valid's rmse: 0.00197918
[100] train's rmse: 0.00114283 valid's rmse: 0.00197758
[150] train's rmse: 0.00113876 valid's rmse: 0.00197684
[200] train's rmse: 0.00113521 valid's rmse: 0.00197656
Early stopping, best iteration is:
[187] train's rmse: 0.00113616 valid's rmse: 0.00197626
Best iteration (3): 187
Best validation RMSE (3): 0.0019762572527070974

> > > Weighted RMSE (3): 0.08302

### Training horizon: 10

80th percentile of ts_index: 2965.0

Training until validation scores don't improve for 50 rounds
[50] train's rmse: 0.00261684 valid's rmse: 0.00318542
[100] train's rmse: 0.00259335 valid's rmse: 0.00318006
[150] train's rmse: 0.0025758 valid's rmse: 0.00317768
[200] train's rmse: 0.00256138 valid's rmse: 0.00318149
Early stopping, best iteration is:
[180] train's rmse: 0.00256688 valid's rmse: 0.00317687
Best iteration (10): 180
Best validation RMSE (10): 0.0031768655913778266

> > > Weighted RMSE (10): 0.1235

## BaseLine + 4 horizon + null imputed + target encoder + feature extracted(correlation) + input sorted(ts_index)
