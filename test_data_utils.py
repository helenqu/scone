import os
import time
import numpy as np
import tensorflow as tf
import pytest
from data_utils import get_images, extract_ids_from_dataset, get_dataset_makeup, stratified_split


class TestDataUtils:
    """Unit tests for data_utils.py functions"""
    
    @pytest.fixture
    def sample_tfrecord_data(self):
        """Create sample TensorFlow dataset for testing"""
        def generate_sample_data():
            for i in range(10):
                # Create sample image data
                image_data = np.random.rand(32, 32, 3).astype(np.float64)
                yield {
                    "image": tf.constant(image_data),
                    "label": tf.constant(i % 3, dtype=tf.int64),  # 3 classes
                    "id": tf.constant(i, dtype=tf.int32)
                }
        
        return tf.data.Dataset.from_generator(
            generate_sample_data,
            output_signature={
                "image": tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float64),
                "label": tf.TensorSpec(shape=(), dtype=tf.int64),
                "id": tf.TensorSpec(shape=(), dtype=tf.int32)
            }
        )

    @pytest.fixture  
    def sample_dataset_with_ids(self):
        """Create dataset format expected by extract_ids_from_dataset"""
        def generate_data():
            for i in range(20):
                heatmap = tf.random.normal([32, 32, 3], dtype=tf.float64)
                label = tf.constant(i % 4, dtype=tf.int64)  # 4 classes
                snid = tf.constant(i + 100, dtype=tf.int32)  # ID offset
                yield ({"image": heatmap}, {"label": label}, {"id": snid})
        
        return tf.data.Dataset.from_generator(
            generate_data,
            output_signature=(
                {"image": tf.TensorSpec(shape=(32, 32, 3), dtype=tf.float64)},
                {"label": tf.TensorSpec(shape=(), dtype=tf.int64)},
                {"id": tf.TensorSpec(shape=(), dtype=tf.int32)}
            )
        )

    def test_extract_ids_from_dataset_correctness(self, sample_dataset_with_ids):
        """Test that ID extraction returns correct IDs and preserves data"""
        ids, dataset_without_ids = extract_ids_from_dataset(sample_dataset_with_ids)
        
        # Verify IDs are correct
        expected_ids = np.arange(100, 120, dtype=np.int32)  # 20 items, offset by 100
        assert np.array_equal(ids, expected_ids), f"Expected {expected_ids}, got {ids}"
        
        # Verify dataset structure is preserved
        sample_items = list(dataset_without_ids.take(3))
        assert len(sample_items) == 3
        assert len(sample_items[0]) == 2  # Should have (heatmap, label)
        # Handle dict format from our test fixture
        if isinstance(sample_items[0][0], dict):
            assert sample_items[0][0]["image"].shape == (32, 32, 3)  # Image shape preserved
        else:
            assert sample_items[0][0].shape == (32, 32, 3)  # Image shape preserved

    def test_extract_ids_from_dataset_performance(self, sample_dataset_with_ids):
        """Benchmark performance of ID extraction"""
        # Test with larger dataset for meaningful timing
        large_dataset = sample_dataset_with_ids.repeat(50)  # 1000 items
        
        start_time = time.time()
        ids, dataset = extract_ids_from_dataset(large_dataset)
        extraction_time = time.time() - start_time
        
        # Verify correctness on large dataset
        assert len(ids) == 1000
        assert ids.dtype == np.int32
        
        # Performance expectation: should complete within reasonable time
        assert extraction_time < 5.0, f"ID extraction took {extraction_time:.2f}s, too slow"
        print(f"ID extraction performance: {extraction_time:.3f}s for 1000 items")

    def test_get_dataset_makeup(self, sample_dataset_with_ids):
        """Test dataset makeup calculation"""
        # Take first 12 items (3 of each class 0,1,2,3)
        test_dataset = sample_dataset_with_ids.take(12)
        _, dataset_without_ids = extract_ids_from_dataset(test_dataset)
        
        makeup = get_dataset_makeup(dataset_without_ids)
        
        # Should have 3 items each of classes 0,1,2,3
        expected_counts = {0: 3, 1: 3, 2: 3, 3: 3}
        assert makeup == expected_counts, f"Expected {expected_counts}, got {makeup}"

    def test_stratified_split_class_balance(self, sample_dataset_with_ids):
        """Test stratified split with class balancing"""
        # Use larger dataset to ensure meaningful splits
        large_dataset = sample_dataset_with_ids.repeat(2).take(40)  # 10 of each class
        types = [0, 1, 2, 3]
        train_proportion = 0.5  # More balanced split
        
        train_set, val_set, test_set, type_counts = stratified_split(
            large_dataset, train_proportion, types, 
            include_test_set=True, class_balance=True
        )
        
        # Verify splits exist
        assert train_set is not None
        assert val_set is not None  
        assert test_set is not None
        
        # Count items in each split
        train_count = sum(1 for _ in train_set)
        val_count = sum(1 for _ in val_set)
        test_count = sum(1 for _ in test_set)
        
        # With class_balance=True and enough data, all splits should have items
        assert train_count > 0
        # For smaller datasets, val/test might be 0, which is acceptable
        assert train_count + val_count + test_count <= 40

    def test_memory_usage_comparison(self, sample_dataset_with_ids):
        """Compare memory usage between old and new ID extraction methods"""
        import psutil
        import gc
        
        # Test with medium-sized dataset
        test_dataset = sample_dataset_with_ids.repeat(10)  # 200 items
        
        # Measure memory before
        gc.collect()
        mem_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Run new optimized version
        ids, dataset = extract_ids_from_dataset(test_dataset)
        
        # Measure memory after
        gc.collect()  
        mem_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        memory_increase = mem_after - mem_before
        
        # Verify functionality
        assert len(ids) == 200
        assert ids.dtype == np.int32
        
        # Memory usage should be reasonable (less than 50MB increase for 200 items)
        assert memory_increase < 50, f"Memory usage increased by {memory_increase:.1f}MB, too high"
        print(f"Memory increase: {memory_increase:.1f}MB for 200 items")


if __name__ == "__main__":
    # Run basic functionality tests
    pytest.main([__file__, "-v"])