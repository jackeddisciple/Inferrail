# Policy comparison

Medians over repetitions, with the range in brackets when they differ.
Latencies count from each request's scheduled send time.

## W1: Shared system prompt

360 requests at 6.0 req/s, 3 repetitions, seed 0.

| policy | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p99 ms | goodput req/s | hit rate | engine hit rate | errors | send lag p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 511 [422, 653] | 1842 [1734, 2414] | 20.0 | 22.2 [22.1, 22.2] | 5.89 [5.83, 5.93] | 0.39 [0.39, 0.45] | 0.38 [0.37, 0.43] | 0 | 2.1 [2.1, 2.2] |
| random | 714 [691, 916] | 3394 [2690, 3859] | 20.0 | 22.1 [22.1, 22.2] | 5.29 [5.04, 5.32] | 0.39 [0.37, 0.43] | 0.37 [0.35, 0.40] | 0 | 2.2 [2.0, 2.2] |
| least_request | 776 [535, 866] | 2207 [1735, 2703] | 20.0 | 22.1 [22.1, 22.2] | 5.74 [5.62, 5.91] | 0.40 [0.34, 0.42] | 0.38 [0.33, 0.39] | 0 | 2.1 [2.1, 2.2] |
| p2c | 711 [516, 1042] | 2255 [1818, 3894] | 20.0 | 22.1 [22.1, 22.2] | 5.68 [4.76, 5.93] | 0.40 [0.31, 0.43] | 0.38 [0.30, 0.41] | 0 | 2.0 [2.0, 2.3] |

![TTFT distribution](w1_ttft_cdf.png)
![TTFT by policy](w1_ttft.png)
![Hit rate](w1_hit_rate.png)

## W2: Multi turn chat

1500 requests at 25.0 req/s, 3 repetitions, seed 0.

| policy | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p99 ms | goodput req/s | hit rate | engine hit rate | errors | send lag p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 101 [100, 110] | 470 [456, 483] | 20.0 | 21.5 [21.4, 21.6] | 24.72 [24.27, 25.08] | 0.46 [0.45, 0.46] | 0.45 [0.44, 0.46] | 0 | 1.3 [1.2, 2.0] |
| random | 143 [136, 150] | 622 [566, 690] | 20.0 | 21.4 [21.4, 21.5] | 24.72 [24.27, 25.08] | 0.46 [0.45, 0.47] | 0.45 [0.45, 0.46] | 0 | 1.1 [1.1, 1.3] |
| least_request | 138 [133, 147] | 601 [573, 619] | 20.0 | 21.5 [21.4, 21.5] | 24.72 [24.27, 25.08] | 0.46 [0.45, 0.46] | 0.45 | 0 | 1.5 [1.1, 1.6] |
| p2c | 142 [139, 152] | 666 [657, 702] | 20.0 | 21.4 [21.4, 21.5] | 24.72 [24.27, 25.08] | 0.45 [0.45, 0.46] | 0.44 [0.44, 0.46] | 0 | 1.2 [1.2, 1.3] |

![TTFT distribution](w2_ttft_cdf.png)
![TTFT by policy](w2_ttft.png)
![Hit rate](w2_hit_rate.png)

## W6: Unique prompts

540 requests at 9.0 req/s, 3 repetitions, seed 0.

| policy | TTFT p50 ms | TTFT p99 ms | TPOT p50 ms | ITL p99 ms | goodput req/s | hit rate | engine hit rate | errors | send lag p99 ms |
|---|---|---|---|---|---|---|---|---|---|
| round_robin | 313 [301, 332] | 846 [763, 1201] | 20.0 | 22.0 [22.0, 22.1] | 9.01 [8.71, 9.06] | 0.00 | 0.00 | 0 | 1.6 [1.6, 2.2] |
| random | 399 [397, 435] | 1657 [1478, 2512] | 20.0 | 22.0 | 8.71 [8.67, 9.04] | 0.00 | 0.00 | 0 | 1.7 [1.5, 1.9] |
| least_request | 363 [350, 363] | 1194 [985, 1346] | 20.0 | 22.0 | 9.01 [8.71, 9.06] | 0.00 | 0.00 | 0 | 1.8 |
| p2c | 354 [346, 394] | 1106 [927, 1470] | 20.0 | 22.0 [22.0, 22.1] | 9.01 [8.71, 9.06] | 0.00 | 0.00 | 0 | 1.6 [1.4, 1.8] |

![TTFT distribution](w6_ttft_cdf.png)
![TTFT by policy](w6_ttft.png)
![Hit rate](w6_hit_rate.png)
