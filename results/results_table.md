| Bug | Description | Random: caught (first test #) | Guided: caught (first test #) | Unsafe (random) | Unsafe (guided) |
|---|---|---|---|---|---|
| 0 | fault-free | no | no | 1 | 1 |
| 1 | clamp at 63 | yes (1) | yes (15) | 7 | 5 |
| 2 | no ReLU | yes (1) | yes (15) | 19 | 19 |
| 3 | L1 bias ignored | yes (1) | yes (2) | 71 | 76 |
| 4 | shift off by one | yes (1) | yes (15) | 7 | 5 |
| 5 | L2 bias ignored | yes (63) | yes (159) | 2 | 2 |
| 6 | acc clamp 14500 | no | no | 1 | 1 |

Accuracy: float 97.2%, INT8 97.1%. Cycles per inference: 279 (at 100 MHz = 2.79 us).
