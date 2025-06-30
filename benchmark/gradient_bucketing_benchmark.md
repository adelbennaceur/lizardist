## Gradient Communication Benchmark

To benchmark our gradient communication strategy under the **Data Parallel** training setup, we tracked the following statistics during training of a simple CNN on MNIST (results shown for one epoch):

### Metrics Tracked

- `average_bytes_per_call`: Average data size per allreduce call.
- `total_allreduce_calls`: Number of allreduce calls made.
- `total_bytes_sent`: Total bytes sent over all allreduce calls.

### Results

| Metric                | With Bucketing | Without Bucketing | Ratio (With / Without) |
|-----------------------|---------------|--------------------|-------------------------|
| Average Bytes/Call   | 827,688       | 103,461            | ~8x larger              |
| Total Allreduce Calls | 235            | 1880               | ~8x fewer               |
| Total Bytes Sent      | 194,506,680    | 194,506,680        |  same            |

- Bucketing leads to fewer allreduce calls with larger average data per call, making communication more efficient overall.
