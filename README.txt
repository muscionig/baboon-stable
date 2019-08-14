# Introduction
This is part of the framework for behavior identification published and presented at KDD19. The code include a sample of data and the related time window scores (comparing different classifiers). We will also publish the entire feature extraction process from the raw time series (aggregation and temporal features) as well as the network inference process for the group classification.

Link to the published paper: https://sites.google.com/usc.edu/kdd19-dmaic-workshop/accepted-papers

# Dataset Structural Characteristics

label encoding
{'Feeding': 0, 'Running': 1, 'Sitting': 2, 'Standing at rest': 3, 'Walking': 4}

Class Balance

Day: 0808
3.0    4976
1.0    4366
2.0    1392
0.0     444
Name: G_LABEL, dtype: int64
Day: 0810
1.0    11537
3.0     1014
2.0       81
0.0        9
Name: G_LABEL, dtype: int64
Day: 0812
1.0    8928
3.0    1123
2.0     154
0.0     131
Name: G_LABEL, dtype: int64
Day: 0813
1.0    2250
3.0    1743
2.0    1616
0.0     137
Name: G_LABEL, dtype: int64
Day: 0814
1.0    28543
3.0     4482
2.0     2204
0.0      305
Name: G_LABEL, dtype: int64
Day: 0817
1.0    7799
3.0    1589
2.0     733
0.0      33
Name: G_LABEL, dtype: int64
Day: 0818
3.0    3473
1.0    2116
2.0    1523
0.0      50
Name: G_LABEL, dtype: int64
Day: 0827
1.0    44052
2.0    11174
3.0     5531
0.0      448
Name: G_LABEL, dtype: int64
Day: 0828
1.0    23065
2.0     4684
3.0     3901
0.0      177
Name: G_LABEL, dtype: int64
Day: 0829
1.0    4477
3.0    3440
2.0     508
0.0      23
Name: G_LABEL, dtype: int64

Normalized:

Day: 0808
3.0    0.445160
1.0    0.390589
2.0    0.124530
0.0    0.039721
Name: G_LABEL, dtype: float64
Day: 0810
1.0    0.912665
3.0    0.080215
2.0    0.006408
0.0    0.000712
Name: G_LABEL, dtype: float64
Day: 0812
1.0    0.863777
3.0    0.108649
2.0    0.014899
0.0    0.012674
Name: G_LABEL, dtype: float64
Day: 0813
1.0    0.391577
3.0    0.303341
2.0    0.281239
0.0    0.023843
Name: G_LABEL, dtype: float64
Day: 0814
1.0    0.803259
3.0    0.126133
2.0    0.062025
0.0    0.008583
Name: G_LABEL, dtype: float64
Day: 0817
1.0    0.768072
3.0    0.156490
2.0    0.072188
0.0    0.003250
Name: G_LABEL, dtype: float64
Day: 0818
3.0    0.484920
1.0    0.295448
2.0    0.212650
0.0    0.006981
Name: G_LABEL, dtype: float64
Day: 0827
1.0    0.719745
2.0    0.182567
3.0    0.090368
0.0    0.007320
Name: G_LABEL, dtype: float64
Day: 0828
1.0    0.724699
2.0    0.147171
3.0    0.122569
0.0    0.005561
Name: G_LABEL, dtype: float64
Day: 0829
1.0    0.529948
3.0    0.407197
2.0    0.060133
0.0    0.002723
Name: G_LABEL, dtype: float64
