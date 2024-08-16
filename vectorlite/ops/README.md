# Overview
The following ops are implemented using Google's Highway SIMD library
1.`InnerProductDistance`
2.`L2DistanceSquared`
3. `Normalize`(L2 Norm)

Based on the benchmark on my PC(i5-12600KF with AVX2 support),
InnerProductDistance is 1.5x-3x faster than HNSWLIB's SIMD implementation
when dealing with vectors with 256 elements or more. The performance gain is
mainly due to Highway can leverage fused Multiply-Add instructions of AVX2
while HNSWLIB can't(HNSWLIB uses Multiply-Add for AVX512 though). Due to
using dynamic dispatch, the performance gain is not as good when dealing with
vectors with less than 256 elements. Because the overhead of dynamic dispatch
is not negligible.

Each benchmark follows pattern `Name/Dimension/Whether to do self-product`.
For example, `BM_InnerProduct_Scalar/128/0` means benchmmarking scalar inner product on vectors with 128 dimension without doing self-product.

```
2024-08-16T00:47:22+08:00
Running build/release/vectorlite/ops/ops_benchmark
Run on (16 X 3686.4 MHz CPU s)
CPU Caches:
  L1 Data 48 KiB (x8)
  L1 Instruction 32 KiB (x8)
  L2 Unified 1280 KiB (x8)
  L3 Unified 20480 KiB (x1)
Load Average: 0.06, 0.21, 0.22
--------------------------------------------------------------------------------
Benchmark                                      Time             CPU   Iterations
--------------------------------------------------------------------------------
BM_InnerProduct_Scalar/128/0                69.9 ns         69.9 ns      9899464
BM_InnerProduct_Scalar/256/0                 167 ns          167 ns      4044812
BM_InnerProduct_Scalar/512/0                 410 ns          410 ns      1730771
BM_InnerProduct_Scalar/1024/0                870 ns          870 ns       787328
BM_InnerProduct_Scalar/2048/0               1796 ns         1796 ns       388851
BM_InnerProduct_Scalar/4096/0               3722 ns         3722 ns       190221
BM_InnerProduct_Scalar/8192/0               7495 ns         7495 ns        93092
BM_InnerProduct_Scalar/16384/0             15065 ns        15065 ns        46103
BM_InnerProduct_Scalar/128/1                71.4 ns         71.4 ns      9751776
BM_InnerProduct_Scalar/256/1                 170 ns          170 ns      4075881
BM_InnerProduct_Scalar/512/1                 397 ns          397 ns      1747276
BM_InnerProduct_Scalar/1024/1                875 ns          875 ns       775140
BM_InnerProduct_Scalar/2048/1               1814 ns         1814 ns       390957
BM_InnerProduct_Scalar/4096/1               3678 ns         3678 ns       192242
BM_InnerProduct_Scalar/8192/1               7476 ns         7476 ns        92873
BM_InnerProduct_Scalar/16384/1             14923 ns        14923 ns        46420
BM_InnerProduct_HNSWLIB/128/0               4.79 ns         4.79 ns    147391631
BM_InnerProduct_HNSWLIB/256/0               11.3 ns         11.3 ns     62204252
BM_InnerProduct_HNSWLIB/512/0               27.5 ns         27.5 ns     25478820
BM_InnerProduct_HNSWLIB/1024/0              71.6 ns         71.5 ns      9644424
BM_InnerProduct_HNSWLIB/2048/0               163 ns          163 ns      4280963
BM_InnerProduct_HNSWLIB/4096/0               404 ns          404 ns      1751642
BM_InnerProduct_HNSWLIB/8192/0               900 ns          900 ns       812263
BM_InnerProduct_HNSWLIB/16384/0             1844 ns         1844 ns       383885
BM_InnerProduct_HNSWLIB/128/1               4.68 ns         4.68 ns    149916850
BM_InnerProduct_HNSWLIB/256/1               12.1 ns         12.1 ns     59371040
BM_InnerProduct_HNSWLIB/512/1               28.7 ns         28.7 ns     24560878
BM_InnerProduct_HNSWLIB/1024/1              73.4 ns         73.4 ns      9957453
BM_InnerProduct_HNSWLIB/2048/1               165 ns          165 ns      4093859
BM_InnerProduct_HNSWLIB/4096/1               405 ns          405 ns      1702073
BM_InnerProduct_HNSWLIB/8192/1               884 ns          884 ns       805253
BM_InnerProduct_HNSWLIB/16384/1             1813 ns         1813 ns       375131
BM_InnerProduct_Vectorlite/128/0            6.43 ns         6.43 ns    112883562
BM_InnerProduct_Vectorlite/256/0            8.87 ns         8.87 ns     78666535
BM_InnerProduct_Vectorlite/512/0            16.1 ns         16.1 ns     42847839
BM_InnerProduct_Vectorlite/1024/0           31.2 ns         31.2 ns     22533448
BM_InnerProduct_Vectorlite/2048/0           61.2 ns         61.2 ns     11228980
BM_InnerProduct_Vectorlite/4096/0            125 ns          125 ns      5979538
BM_InnerProduct_Vectorlite/8192/0            494 ns          494 ns      1388740
BM_InnerProduct_Vectorlite/16384/0           988 ns          988 ns       710480
BM_InnerProduct_Vectorlite/128/1            6.44 ns         6.44 ns    110012038
BM_InnerProduct_Vectorlite/256/1            7.70 ns         7.70 ns     91759836
BM_InnerProduct_Vectorlite/512/1            11.6 ns         11.6 ns     62003309
BM_InnerProduct_Vectorlite/1024/1           23.0 ns         23.0 ns     30713178
BM_InnerProduct_Vectorlite/2048/1           53.6 ns         53.6 ns     13192811
BM_InnerProduct_Vectorlite/4096/1            109 ns          109 ns      6206796
BM_InnerProduct_Vectorlite/8192/1            233 ns          233 ns      3024667
BM_InnerProduct_Vectorlite/16384/1           476 ns          476 ns      1475109
BM_Normalize_Vectorlite/128                 18.4 ns         18.4 ns     38176598
BM_Normalize_Vectorlite/256                 33.3 ns         33.3 ns     20490508
BM_Normalize_Vectorlite/512                 38.0 ns         38.0 ns     18313796
BM_Normalize_Vectorlite/1024                72.3 ns         72.3 ns      9697467
BM_Normalize_Vectorlite/2048                 122 ns          122 ns      5632044
BM_Normalize_Vectorlite/4096                 228 ns          228 ns      3095533
BM_Normalize_Vectorlite/8192                 425 ns          425 ns      1666335
BM_Normalize_Vectorlite/16384               1454 ns         1454 ns       499331
BM_Normalize_Scalar/128                     72.1 ns         72.1 ns      9828631
BM_Normalize_Scalar/256                      137 ns          137 ns      5192520
BM_Normalize_Scalar/512                      274 ns          274 ns      2579448
BM_Normalize_Scalar/1024                     523 ns          523 ns      1292813
BM_Normalize_Scalar/2048                    1065 ns         1065 ns       643672
BM_Normalize_Scalar/4096                    2136 ns         2136 ns       334229
BM_Normalize_Scalar/8192                    4267 ns         4267 ns       164707
BM_Normalize_Scalar/16384                   8520 ns         8520 ns        83869
BM_L2DistanceSquared_Scalar/128             36.1 ns         36.1 ns     19270008
BM_L2DistanceSquared_Scalar/256             82.4 ns         82.4 ns      8468353
BM_L2DistanceSquared_Scalar/512              203 ns          203 ns      3435362
BM_L2DistanceSquared_Scalar/1024             441 ns          441 ns      1580202
BM_L2DistanceSquared_Scalar/2048             925 ns          925 ns       755086
BM_L2DistanceSquared_Scalar/4096            1867 ns         1867 ns       378141
BM_L2DistanceSquared_Scalar/8192            3793 ns         3793 ns       190928
BM_L2DistanceSquared_Scalar/16384           7549 ns         7549 ns        90998
BM_L2DistanceSquared_Vectorlite/128         6.44 ns         6.45 ns    104236778
BM_L2DistanceSquared_Vectorlite/256        10.00 ns        10.00 ns     70830344
BM_L2DistanceSquared_Vectorlite/512         19.8 ns         19.8 ns     33857463
BM_L2DistanceSquared_Vectorlite/1024        36.5 ns         36.5 ns     18063887
BM_L2DistanceSquared_Vectorlite/2048        71.7 ns         71.7 ns      9993262
BM_L2DistanceSquared_Vectorlite/4096         147 ns          147 ns      4725818
BM_L2DistanceSquared_Vectorlite/8192         511 ns          511 ns      1377505
BM_L2DistanceSquared_Vectorlite/16384       1007 ns         1007 ns       686767
BM_L2DistanceSquared_HNSWLIB/128            6.05 ns         6.05 ns    118783991
BM_L2DistanceSquared_HNSWLIB/256            14.2 ns         14.2 ns     46527797
BM_L2DistanceSquared_HNSWLIB/512            37.9 ns         37.9 ns     19404810
BM_L2DistanceSquared_HNSWLIB/1024           87.2 ns         87.2 ns      7999479
BM_L2DistanceSquared_HNSWLIB/2048            231 ns          231 ns      3083078
BM_L2DistanceSquared_HNSWLIB/4096            527 ns          527 ns      1268927
BM_L2DistanceSquared_HNSWLIB/8192           1123 ns         1123 ns       632412
BM_L2DistanceSquared_HNSWLIB/16384          2307 ns         2307 ns       304141

```