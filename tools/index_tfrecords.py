#!/usr/bin/env python3
"""
Build a CSV index of SNIDs across all TFRecord files in a directory.

After one initial run, the index can be passed to visualize_tfrecords.py
via --index so it jumps directly to the right file instead of scanning all files.

Usage:
    # Build index
    python index_tfrecords.py --tfrecord heatmaps/ --output snid_index.csv.gz

    # Then visualize using the index (no full scan needed)
    python visualize_tfrecords.py --tfrecord heatmaps/ --index snid_index.csv.gz --sample_ids 1009,1521,2034

 Nov 25 2026 RK 
   + use logger to print and add timer. If time is long (e.g. for BCOR),
     might move this funcionality into create_heatmaps to be much more efficient.

"""

import os, sys, time
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_VISIBLE_DEVICES'] = ''
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import csv
import gzip
import argparse
from pathlib import Path


def parse_tfrecord_id_only(raw_record):
    """Parse only the metadata fields (skip image decoding for speed)."""
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'id':    tf.io.FixedLenFeature([], tf.int64),
        'z':     tf.io.FixedLenFeature([], tf.float32),
        'z_err': tf.io.FixedLenFeature([], tf.float32),
    }
    example = tf.io.parse_single_example(raw_record, feature_description)
    return {
        'id':    int(example['id'].numpy()),
        'label': int(example['label'].numpy()),
        'z':     float(example['z'].numpy()),
        'z_err': float(example['z_err'].numpy()),
    }


def build_index(tfrecord_files, output_csv):
    total_records = 0
    with gzip.open(output_csv, 'wt', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snid', 'label', 'z', 'z_err', 'tfrecord_file'])

        for i, tfrecord_file in enumerate(tfrecord_files):
            file_count = 0
            dataset = tf.data.TFRecordDataset(tfrecord_file)
            for raw_record in dataset:
                meta = parse_tfrecord_id_only(raw_record)
                writer.writerow([
                    meta['id'],
                    meta['label'],
                    f"{meta['z']:.6f}",
                    f"{meta['z_err']:.6f}",
                    tfrecord_file,
                ])
                file_count += 1

            total_records += file_count
            logging.info(f"  [{i+1}/{len(tfrecord_files)}] {Path(tfrecord_file).name}: {file_count} records")

    # - - - -
    logging.info(f"")
    logging.info(f"Indexed {total_records} records from {len(tfrecord_files)} heatmap files")
    logging.info(f"Wrote {output_csv}")

def setup_logging():
    logging.basicConfig(level=logging.INFO,
                        format="[%(levelname)6s |%(filename)15s] %(message)s")


def get_args():

    # created Nov 25 2026 by R.Kessler
    # Cleanup main, and expandvars for input paths.

    parser = argparse.ArgumentParser(description='Build a SNID->tfrecord CSV index')
    parser.add_argument('--tfrecord', required=True,
                        help='Directory containing *.tfrecord files')
    parser.add_argument('--output', type=str, default='snid_index.csv.gz',
                        help='Output CSV file (default: snid_index.csv.gz)')
    args = parser.parse_args()

    args.tfrecord = os.path.expandvars(args.tfrecord)
    args.output   = os.path.expandvars(args.output)

    return args

# ================================

def main():

    setup_logging()

    logging.info(f'')
    logging.info(f'Begin {sys.argv[0]}')
    t_start = time.time()

    args = get_args()

    tfrecord_path = Path(args.tfrecord)
    if not tfrecord_path.is_dir():
        logging.info(f"Error: {tfrecord_path} is not a directory")
        return

    tfrecord_files = sorted(str(f) for f in tfrecord_path.glob('*.tfrecord'))
    if not tfrecord_files:
        logging.info(f"ERROR: no .tfrecord files found in {tfrecord_path}")
        return

    logging.info(f"Found {len(tfrecord_files)} tfrecord files in {tfrecord_path}")
    logging.info(f"Building index (image data skipped for speed)...\n")

    build_index(tfrecord_files, args.output)

    # - - -
    t_elapse = (time.time() - t_start)/60.0
    logging.info(f"Finished buidling index list in {t_elapse:.2f} minutes")

if __name__ == '__main__':
    main()
