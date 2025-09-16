#!/usr/bin/env python3
"""
Performance benchmark script for data_utils.py optimizations
Compares old vs new implementation performance and memory usage
"""

import time
import numpy as np
import tensorflow as tf
import psutil
import gc
from contextlib import contextmanager
from data_utils import extract_ids_from_dataset


@contextmanager
def memory_monitor():
    """Context manager to monitor memory usage"""
    gc.collect()
    mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    start_time = time.time()
    
    yield
    
    end_time = time.time()
    gc.collect()
    mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
    
    print(f"  Time: {end_time - start_time:.3f}s")
    print(f"  Memory increase: {mem_after - mem_before:.1f}MB")


def create_test_dataset(size):
    """Create test dataset with specified size"""
    def generate_data():
        for i in range(size):
            heatmap = tf.random.normal([64, 64, 3], dtype=tf.float64)
            label = tf.constant(i % 10, dtype=tf.int64)  # 10 classes
            snid = tf.constant(i + 1000, dtype=tf.int32)
            yield ({"image": heatmap}, {"label": label}, {"id": snid})
    
    return tf.data.Dataset.from_generator(
        generate_data,
        output_signature=(
            {"image": tf.TensorSpec(shape=(64, 64, 3), dtype=tf.float64)},
            {"label": tf.TensorSpec(shape=(), dtype=tf.int64)}, 
            {"id": tf.TensorSpec(shape=(), dtype=tf.int32)}
        )
    )


def old_extract_ids_from_dataset(cached_dataset):
    """Original TensorArray-based implementation for comparison"""
    ids = tf.TensorArray(dtype=tf.int32, size=0, dynamic_size=True)
    dataset = cached_dataset.map(lambda heatmap, label, snid: (heatmap, label))
    
    def extract_id_func(ids, x):
        # Handle both dict and tensor formats
        if isinstance(x[2], dict):
            return ids.write(ids.size(), x[2]["id"])
        else:
            return ids.write(ids.size(), x[2])
    
    ids = cached_dataset.reduce(ids, extract_id_func)
    return ids.stack().numpy(), dataset


def benchmark_implementations():
    """Compare old vs new implementation performance"""
    test_sizes = [100, 500, 1000, 2000]
    
    print("=== Data Utils Performance Benchmark ===\n")
    
    for size in test_sizes:
        print(f"Dataset size: {size} items")
        test_dataset = create_test_dataset(size)
        
        # Benchmark old implementation
        print("  Old implementation (TensorArray):")
        with memory_monitor():
            try:
                ids_old, dataset_old = old_extract_ids_from_dataset(test_dataset)
                old_success = True
            except Exception as e:
                print(f"    ERROR: {e}")
                old_success = False
        
        # Reset dataset for fair comparison
        test_dataset = create_test_dataset(size)
        
        # Benchmark new implementation  
        print("  New implementation (Direct list):")
        with memory_monitor():
            try:
                ids_new, dataset_new = extract_ids_from_dataset(test_dataset)
                new_success = True
            except Exception as e:
                print(f"    ERROR: {e}")
                new_success = False
        
        # Verify correctness if both succeeded
        if old_success and new_success:
            if np.array_equal(ids_old, ids_new):
                print("  ✓ Results match between implementations")
            else:
                print("  ✗ Results differ between implementations!")
                print(f"    Old IDs sample: {ids_old[:5]}")
                print(f"    New IDs sample: {ids_new[:5]}")
        
        print()


def stress_test():
    """Stress test with larger datasets to identify limits"""
    print("=== Stress Test ===")
    
    large_sizes = [5000, 10000]
    
    for size in large_sizes:
        print(f"\nStress testing with {size} items:")
        test_dataset = create_test_dataset(size)
        
        try:
            with memory_monitor():
                ids, dataset = extract_ids_from_dataset(test_dataset)
                
            print(f"  ✓ Successfully processed {len(ids)} items")
            
            # Quick verification
            expected_first_id = 1000
            expected_last_id = 1000 + size - 1
            
            if ids[0] == expected_first_id and ids[-1] == expected_last_id:
                print("  ✓ ID sequence is correct")
            else:
                print(f"  ✗ ID sequence error: got {ids[0]}-{ids[-1]}, expected {expected_first_id}-{expected_last_id}")
                
        except Exception as e:
            print(f"  ✗ Failed: {e}")


if __name__ == "__main__":
    print("Starting data_utils performance benchmarks...")
    print("This will test the optimized ID extraction function.\n")
    
    benchmark_implementations()
    stress_test()
    
    print("\n=== Summary ===")
    print("The new implementation should show:")
    print("- Similar or better performance")
    print("- Comparable memory usage")  
    print("- Identical correctness")
    print("\nIf any tests fail, the optimization may need adjustment.")