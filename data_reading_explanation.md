# Data Reading Efficiency in SCONE

## Overview
The code uses TensorFlow's advanced data pipeline for highly efficient parallel data reading, not sequential single-chunk reading.

## Key Efficiency Mechanisms

### 1. Parallel File Reading
```
┌─────────────┐
│  TFRecord 1 │──┐
├─────────────┤  │
│  TFRecord 2 │──┤
├─────────────┤  │    ┌──────────────────┐
│  TFRecord 3 │──┼───►│  80 Parallel     │
├─────────────┤  │    │  Readers         │───► Raw Dataset
│     ...     │──┤    │  (Simultaneous)  │
├─────────────┤  │    └──────────────────┘
│ TFRecord 80 │──┘
└─────────────┘
```
- **80 files** are read simultaneously using `num_parallel_reads=80`
- Each reader operates independently on different files
- I/O operations are parallelized across multiple threads

### 2. Parallel Processing Pipeline
```
Raw Dataset ──► [Chunk 1] ──┐
            ──► [Chunk 2] ──┤
            ──► [Chunk 3] ──┼──► Parallel Processing ──► Processed Dataset
            ──► [Chunk N] ──┘    (AUTOTUNE threads)
```
- `num_parallel_calls=tf.data.AUTOTUNE` automatically optimizes parallelism
- Multiple chunks processed simultaneously based on available CPU cores
- Typically uses all available CPU threads

### 3. Prefetch Pipeline
```
Timeline:
Step 1: [Read Batch 1] [Process Batch 1] [Train on Batch 1]
Step 2:                [Read Batch 2]    [Process Batch 2]   [Train on Batch 2]
Step 3:                                  [Read Batch 3]      [Process Batch 3]   [Train on Batch 3]
        └─────────────────── All happening in parallel ──────────────────┘
```
- Prefetching overlaps I/O with computation
- While GPU trains on current batch, CPU prepares next batches
- No idle time waiting for data

### 4. Memory Management
- **Streaming**: Data is streamed, not loaded all at once
- **Buffering**: Internal buffers manage data flow
- **Caching**: Processed data cached in memory for training epochs

## Performance Characteristics

### Typical Processing Flow:
1. **Initial Load**: 80 files start reading simultaneously
2. **Processing**: Multiple CPU threads process chunks in parallel
3. **Buffering**: Processed chunks buffered for training
4. **Prefetching**: Next batches prepared while current batch trains

### Efficiency Metrics:
- **Parallelism Level**: 80 file readers + AUTOTUNE processing threads
- **Memory Usage**: Controlled through streaming and buffering
- **I/O Optimization**: Overlapped with computation
- **Cache Efficiency**: Training data cached after first pass

## Configuration Options

### Default Mode:
```yaml
# Automatic optimization
# Minimal logging
```

### Verbose Mode (via config):
```yaml
verbose_data_loading: true
# Or use debug flag
debug_flag: 1  # Enable verbose logging
```

### Debug Modes (via command line or config):
```bash
# Use legacy implementation
--debug_flag 900

# Use refactored implementation with verbose logging
--debug_flag 902
```

## Monitoring Output Example:
```
Found 5 heatmap files
Total data size to load: 515.8 MB
Dataset created with 80 parallel readers
Calculating dataset size...
Total dataset size: 5861 records
Dataset pipeline created with -1 parallel processing
Running scone prediction on full dataset of 5861 examples
184/184 [==============================] - 3s 16ms/step
```

### Important Notes on Progress Reporting:
- Due to TensorFlow's **lazy evaluation**, data processing happens during `model.predict()`, not during dataset creation
- The progress bar `184/184 [==============================]` shows actual batch processing
- 184 batches × 32 batch_size = 5,888 ≈ 5,861 total records
- Progress messages during dataset setup have been removed to avoid confusion

## Why This is Efficient:

1. **No Sequential Bottleneck**: Multiple files and chunks processed simultaneously
2. **CPU/GPU Overlap**: Data preparation happens while model trains
3. **Automatic Optimization**: TensorFlow adjusts parallelism based on hardware
4. **Memory Efficiency**: Streaming prevents loading entire dataset into memory
5. **I/O Hiding**: Disk reads overlapped with processing

## Potential Bottlenecks:

1. **Disk I/O Speed**: Limited by storage system bandwidth
2. **CPU Processing**: `get_images()` transformation speed
3. **Memory Bandwidth**: Moving data between CPU and GPU
4. **Network**: If reading from network storage

## Optimization Tips:

1. Ensure data is on fast local storage (SSD preferred)
2. Adjust `num_parallel_reads` based on storage system
3. Use `verbose_data_loading: true` or `--debug_flag 1` to identify bottlenecks
4. Monitor memory usage to avoid swapping
5. Consider data format optimization (compression, serialization)

## Debug Flag System for Testing

The code supports switching between different data reading implementations using debug flags:

### Available Implementations:
- **Legacy** (`debug_flag: 900`): Original `_retrieve_data_legacy()` implementation
- **Refactored** (`debug_flag: 901`): New memory-efficient implementation
- **Refactored+Verbose** (`debug_flag: 902`): New implementation with detailed logging

### Usage Example:
```bash
# Compare implementations
$HOME/soft/scone/model_utils.py --config_path CONFIG --debug_flag 900  # Legacy
$HOME/soft/scone/model_utils.py --config_path CONFIG --debug_flag 902  # Refactored+Verbose
```

This allows A/B testing of different implementations without code changes.

## Technical Details

### Lazy Evaluation in TensorFlow:
- Dataset operations (`.map()`, `.batch()`, etc.) create a **pipeline blueprint**
- Actual data processing only happens when the dataset is **consumed**
- In prediction mode, consumption happens during `model.predict()`
- This is why progress tracking during dataset creation shows minimal activity

### Memory Efficiency:
- The refactored implementation uses `tf.data.AUTOTUNE` for optimal parallelism
- Data is processed in streaming fashion, not loaded all at once
- Memory usage stays controlled even with large datasets