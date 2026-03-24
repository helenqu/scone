#!/usr/bin/env python3
"""
Visualize SCONE TFRecord heatmaps

This script reads SCONE tfrecord files and creates visualizations of the
heatmap data, including:
- Flux heatmaps (wavelength vs time)
- Error heatmaps
- Wavelength-integrated light curves
- Time-averaged spectra
- Statistics across multiple records

Usage:
    python visualize_tfrecords.py --tfrecord heatmaps_0000.tfrecord [--num_samples 5]
    python visualize_tfrecords.py --tfrecord heatmaps_0000.tfrecord --output_dir ./plots
    python visualize_tfrecords.py --tfrecord heatmaps/ --sample_ids 1009,2034,5678
    python visualize_tfrecords.py --tfrecord heatmaps_0000.tfrecord --sample_ids 1009,2034,5678

When --sample_ids is used and --tfrecord points to a directory, snid_index.csv.gz is
automatically looked up in that directory (produced by index_tfrecords.py or run.py).
Pass --index explicitly only if the index lives elsewhere.
"""

import os
import warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'          # suppress TF C++ logs (CUDA/GPU)
os.environ['CUDA_VISIBLE_DEVICES'] = ''            # don't even try GPU
warnings.filterwarnings('ignore', category=UserWarning, module='google.protobuf')

import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)            # suppress TF Python-level warnings
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import csv
import gzip
from pathlib import Path

# Configuration from SCONE
NUM_WAVELENGTH_BINS = 32
NUM_MJD_BINS = 180
INPUT_SHAPE = (NUM_WAVELENGTH_BINS, NUM_MJD_BINS, 2)

# Physical parameters (from SCONE create_heatmaps)
WAVELENGTH_MIN = 3000  # Angstroms
WAVELENGTH_MAX = 10100  # Angstroms
MJD_RANGE_START = -50  # days relative to peak
MJD_RANGE_END = 130    # days relative to peak

def parse_tfrecord(raw_record):
    """Parse a single TFRecord example"""
    feature_description = {
        'label': tf.io.FixedLenFeature([], tf.int64),
        'image_raw': tf.io.FixedLenFeature([], tf.string),
        'id': tf.io.FixedLenFeature([], tf.int64),
        'z': tf.io.FixedLenFeature([], tf.float32),
        'z_err': tf.io.FixedLenFeature([], tf.float32),
    }

    example = tf.io.parse_single_example(raw_record, feature_description)

    # Decode image
    image = tf.reshape(tf.io.decode_raw(example['image_raw'], tf.float64), INPUT_SHAPE)

    return {
        'id': example['id'].numpy(),
        'label': example['label'].numpy(),
        'z': example['z'].numpy(),
        'z_err': example['z_err'].numpy(),
        'flux': image[:, :, 0].numpy(),  # (wavelength, mjd)
        'flux_err': image[:, :, 1].numpy()
    }

def get_wavelength_array():
    """Get wavelength bin centers in Angstroms"""
    return np.linspace(WAVELENGTH_MIN, WAVELENGTH_MAX, NUM_WAVELENGTH_BINS)

def get_mjd_array():
    """Get MJD bin centers in days relative to peak"""
    return np.linspace(MJD_RANGE_START, MJD_RANGE_END, NUM_MJD_BINS)

def visualize_single_heatmap(data, output_file=None):
    """Create comprehensive visualization for a single supernova"""

    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.3, wspace=0.3)

    wavelengths = get_wavelength_array()
    mjds = get_mjd_array()

    # Normalize flux for better visualization
    flux_norm = data['flux'] / np.max(data['flux']) if np.max(data['flux']) > 0 else data['flux']

    # Title
    label_name = "SNIa" if data['label'] == 1 else "Non-Ia"
    fig.suptitle(f"SNID {data['id']} | {label_name} | z={data['z']:.4f}±{data['z_err']:.4f}",
                 fontsize=14, fontweight='bold')

    # 1. Main flux heatmap (wavelength vs time)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    im1 = ax1.imshow(flux_norm, aspect='auto', origin='lower',
                     extent=[mjds[0], mjds[-1], wavelengths[0], wavelengths[-1]],
                     cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Days from Peak MJD', fontsize=11)
    ax1.set_ylabel('Wavelength (Å)', fontsize=11)
    ax1.set_title('Normalized Flux Heatmap', fontsize=12, fontweight='bold')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1)
    plt.colorbar(im1, ax=ax1, label='Normalized Flux')
    ax1.grid(True, alpha=0.3)

    # 2. Error heatmap
    ax2 = fig.add_subplot(gs[0:2, 2])
    im2 = ax2.imshow(data['flux_err'], aspect='auto', origin='lower',
                     extent=[mjds[0], mjds[-1], wavelengths[0], wavelengths[-1]],
                     cmap='hot', interpolation='nearest')
    ax2.set_xlabel('Days from Peak', fontsize=9)
    ax2.set_ylabel('Wavelength (Å)', fontsize=9)
    ax2.set_title('Flux Error', fontsize=10, fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Error', pad=0.02)
    ax2.grid(True, alpha=0.3)

    # 3. Wavelength-integrated light curve
    ax3 = fig.add_subplot(gs[2, 0])
    light_curve = np.sum(data['flux'], axis=0)  # Sum over wavelength
    light_curve_err = np.sqrt(np.sum(data['flux_err']**2, axis=0))  # Quadrature sum
    ax3.plot(mjds, light_curve, 'b-', linewidth=1.5, label='Total Flux')
    ax3.fill_between(mjds, light_curve - light_curve_err, light_curve + light_curve_err,
                     alpha=0.3, color='blue')
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1, label='Peak')
    ax3.set_xlabel('Days from Peak MJD', fontsize=10)
    ax3.set_ylabel('Total Flux (all λ)', fontsize=10)
    ax3.set_title('Light Curve', fontsize=11, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    # 4. Time-averaged spectrum
    ax4 = fig.add_subplot(gs[2, 1])
    # Average spectrum (around peak: -10 to +20 days)
    peak_idx_start = np.argmin(np.abs(mjds - (-10)))
    peak_idx_end = np.argmin(np.abs(mjds - 20))
    spectrum = np.mean(data['flux'][:, peak_idx_start:peak_idx_end], axis=1)
    spectrum_err = np.sqrt(np.mean(data['flux_err'][:, peak_idx_start:peak_idx_end]**2, axis=1))

    ax4.plot(wavelengths, spectrum, 'r-', linewidth=1.5, label='Avg Spectrum')
    ax4.fill_between(wavelengths, spectrum - spectrum_err, spectrum + spectrum_err,
                     alpha=0.3, color='red')
    ax4.set_xlabel('Wavelength (Å)', fontsize=10)
    ax4.set_ylabel('Flux (avg -10 to +20d)', fontsize=10)
    ax4.set_title('Peak Spectrum', fontsize=11, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    # 5. Signal-to-Noise heatmap
    ax5 = fig.add_subplot(gs[2, 2])
    snr = np.where(data['flux_err'] > 0, data['flux'] / data['flux_err'], 0)
    snr = np.clip(snr, 0, 100)  # Clip for visualization
    im5 = ax5.imshow(snr, aspect='auto', origin='lower',
                     extent=[mjds[0], mjds[-1], wavelengths[0], wavelengths[-1]],
                     cmap='RdYlGn', interpolation='nearest', vmin=0, vmax=50)
    ax5.set_xlabel('Days from Peak', fontsize=9)
    ax5.set_ylabel('Wavelength (Å)', fontsize=9)
    ax5.set_title('Signal-to-Noise', fontsize=10, fontweight='bold')
    plt.colorbar(im5, ax=ax5, label='SNR', pad=0.02)
    ax5.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()

    plt.close()

def visualize_statistics(tfrecord_files, num_samples=100, output_file=None):
    """Create statistical visualizations across multiple samples"""

    dataset = tf.data.TFRecordDataset(tfrecord_files)

    all_fluxes = []
    all_labels = []
    all_redshifts = []

    label = tfrecord_files if isinstance(tfrecord_files, str) else f"{len(tfrecord_files)} files"
    print(f"Reading {num_samples} samples from {label}...")
    for i, raw_record in enumerate(dataset.take(num_samples)):
        data = parse_tfrecord(raw_record)
        all_fluxes.append(data['flux'])
        all_labels.append(data['label'])
        all_redshifts.append(data['z'])

        if (i+1) % 10 == 0:
            print(f"  Processed {i+1}/{num_samples}")

    all_fluxes = np.array(all_fluxes)
    all_labels = np.array(all_labels)
    all_redshifts = np.array(all_redshifts)

    wavelengths = get_wavelength_array()
    mjds = get_mjd_array()

    # Create figure
    fig = plt.figure(figsize=(16, 10))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle(f'Statistics from {num_samples} samples', fontsize=14, fontweight='bold')

    # 1. Mean heatmap
    ax1 = fig.add_subplot(gs[0, 0])
    mean_flux = np.mean(all_fluxes, axis=0)
    mean_flux_norm = mean_flux / np.max(mean_flux) if np.max(mean_flux) > 0 else mean_flux
    im1 = ax1.imshow(mean_flux_norm, aspect='auto', origin='lower',
                     extent=[mjds[0], mjds[-1], wavelengths[0], wavelengths[-1]],
                     cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('Days from Peak')
    ax1.set_ylabel('Wavelength (Å)')
    ax1.set_title('Mean Flux (all samples)')
    plt.colorbar(im1, ax=ax1, label='Normalized Flux')
    ax1.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # 2. Std deviation heatmap
    ax2 = fig.add_subplot(gs[0, 1])
    std_flux = np.std(all_fluxes, axis=0)
    im2 = ax2.imshow(std_flux, aspect='auto', origin='lower',
                     extent=[mjds[0], mjds[-1], wavelengths[0], wavelengths[-1]],
                     cmap='plasma', interpolation='nearest')
    ax2.set_xlabel('Days from Peak')
    ax2.set_ylabel('Wavelength (Å)')
    ax2.set_title('Std Dev of Flux')
    plt.colorbar(im2, ax=ax2, label='Std Dev')
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # 3. Class distribution
    ax3 = fig.add_subplot(gs[0, 2])
    labels_unique, counts = np.unique(all_labels, return_counts=True)
    label_names = ['Non-Ia' if l == 0 else 'SNIa' for l in labels_unique]
    colors = ['orange' if l == 0 else 'blue' for l in labels_unique]
    ax3.bar(label_names, counts, color=colors, alpha=0.7, edgecolor='black')
    ax3.set_ylabel('Count')
    ax3.set_title('Class Distribution')
    ax3.grid(True, alpha=0.3, axis='y')
    for i, (name, count) in enumerate(zip(label_names, counts)):
        ax3.text(i, count, f'{count}', ha='center', va='bottom', fontweight='bold')

    # 4. Redshift distribution
    ax4 = fig.add_subplot(gs[1, 0])
    ax4.hist(all_redshifts, bins=30, alpha=0.7, color='green', edgecolor='black')
    ax4.set_xlabel('Redshift (z)')
    ax4.set_ylabel('Count')
    ax4.set_title(f'Redshift Distribution (mean={np.mean(all_redshifts):.3f})')
    ax4.grid(True, alpha=0.3)
    ax4.axvline(x=np.mean(all_redshifts), color='red', linestyle='--',
                label=f'Mean: {np.mean(all_redshifts):.3f}')
    ax4.legend()

    # 5. Mean light curves by class
    ax5 = fig.add_subplot(gs[1, 1])
    for label in np.unique(all_labels):
        mask = all_labels == label
        lc = np.mean(np.sum(all_fluxes[mask], axis=1), axis=0)  # Mean over samples, sum over wavelength
        label_name = 'Non-Ia' if label == 0 else 'SNIa'
        color = 'orange' if label == 0 else 'blue'
        ax5.plot(mjds, lc, label=label_name, linewidth=2, color=color)
    ax5.set_xlabel('Days from Peak')
    ax5.set_ylabel('Mean Total Flux')
    ax5.set_title('Mean Light Curves by Class')
    ax5.axvline(x=0, color='red', linestyle='--', alpha=0.5)
    ax5.grid(True, alpha=0.3)
    ax5.legend()

    # 6. Mean spectra by class
    ax6 = fig.add_subplot(gs[1, 2])
    peak_idx_start = np.argmin(np.abs(mjds - (-10)))
    peak_idx_end = np.argmin(np.abs(mjds - 20))
    for label in np.unique(all_labels):
        mask = all_labels == label
        spec = np.mean(np.mean(all_fluxes[mask, :, peak_idx_start:peak_idx_end], axis=2), axis=0)
        label_name = 'Non-Ia' if label == 0 else 'SNIa'
        color = 'orange' if label == 0 else 'blue'
        ax6.plot(wavelengths, spec, label=label_name, linewidth=2, color=color)
    ax6.set_xlabel('Wavelength (Å)')
    ax6.set_ylabel('Mean Flux (peak phase)')
    ax6.set_title('Mean Spectra by Class')
    ax6.grid(True, alpha=0.3)
    ax6.legend()

    plt.tight_layout()

    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_file}")
    else:
        plt.show()

    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize SCONE TFRecord heatmaps')
    parser.add_argument('--tfrecord', required=True, help='Path to a TFRecord file or a directory containing *.tfrecord files')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of individual samples to visualize')
    parser.add_argument('--sample_ids', type=str, default=None,
                       help='Comma-separated list of specific SNID to visualize')
    parser.add_argument('--statistics', action='store_true',
                       help='Create statistical plots across many samples')
    parser.add_argument('--stat_samples', type=int, default=100,
                       help='Number of samples for statistics')
    parser.add_argument('--output_dir', type=str, default='./tfrecord_plots',
                       help='Output directory for plots')
    parser.add_argument('--index', type=str, default=None,
                       help='Path to snid_index.csv.gz (optional override; auto-detected from --tfrecord dir when --sample_ids is used)')

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    # Resolve --tfrecord to a list of files (accepts file or directory)
    tfrecord_path = Path(args.tfrecord)
    if tfrecord_path.is_dir():
        tfrecord_files = sorted(str(f) for f in tfrecord_path.glob('*.tfrecord'))
        if not tfrecord_files:
            print(f"No .tfrecord files found in {tfrecord_path}")
            return
        print(f"Found {len(tfrecord_files)} tfrecord files in {tfrecord_path}")
    else:
        tfrecord_files = [str(tfrecord_path)]

    print(f"Output directory: {output_dir}")

    # Auto-detect index when --sample_ids is used and --index not explicitly provided
    if args.sample_ids and args.index is None and tfrecord_path.is_dir():
        candidate = tfrecord_path / 'snid_index.csv.gz'
        if candidate.exists():
            args.index = str(candidate)
            print(f"Auto-detected index: {args.index}")
        else:
            print("Note: no snid_index.csv.gz found in tfrecord dir; falling back to sequential scan.")

    # Statistics plot
    if args.statistics:
        print("\nCreating statistical plots...")
        output_file = output_dir / "statistics.png"
        visualize_statistics(tfrecord_files, num_samples=args.stat_samples,
                           output_file=str(output_file))

    # Individual sample plots
    if args.sample_ids:
        # Visualize specific SNIDs
        target_ids = [int(sid.strip()) for sid in args.sample_ids.split(',')]
        print(f"\nLooking for specific SNIDs: {target_ids}")

        if args.index:
            # Use pre-built index to find which file each SNID lives in
            id_to_file = {}
            opener = gzip.open if args.index.endswith('.gz') else open
            with opener(args.index, 'rt', newline='') as f:
                for row in csv.DictReader(f):
                    snid = int(row['snid'])
                    if snid in target_ids:
                        id_to_file[snid] = row['tfrecord_file']

            missing_in_index = set(target_ids) - id_to_file.keys()
            if missing_in_index:
                print(f"Warning: SNIDs not found in index: {missing_in_index}")

            # For each file that contains at least one target, scan only that file
            file_to_ids = {}
            for snid, fpath in id_to_file.items():
                file_to_ids.setdefault(fpath, set()).add(snid)

            found_ids = set()
            for fpath, ids_in_file in file_to_ids.items():
                dataset = tf.data.TFRecordDataset(fpath)
                for raw_record in dataset:
                    data = parse_tfrecord(raw_record)
                    if data['id'] in ids_in_file:
                        print(f"Found SNID {data['id']}")
                        output_file = output_dir / f"snid_{data['id']}.png"
                        visualize_single_heatmap(data, output_file=str(output_file))
                        found_ids.add(data['id'])
                        if found_ids >= ids_in_file:
                            break
        else:
            # No index: scan all files sequentially
            dataset = tf.data.TFRecordDataset(tfrecord_files)
            found_ids = set()
            for raw_record in dataset:
                data = parse_tfrecord(raw_record)
                if data['id'] in target_ids:
                    print(f"Found SNID {data['id']}")
                    output_file = output_dir / f"snid_{data['id']}.png"
                    visualize_single_heatmap(data, output_file=str(output_file))
                    found_ids.add(data['id'])
                    if len(found_ids) == len(target_ids):
                        break

        missing = set(target_ids) - found_ids
        if missing:
            print(f"Warning: Did not find SNIDs: {missing}")
    else:
        # Visualize first N samples
        print(f"\nVisualizing first {args.num_samples} samples...")
        dataset = tf.data.TFRecordDataset(tfrecord_files)
        for i, raw_record in enumerate(dataset.take(args.num_samples)):
            data = parse_tfrecord(raw_record)
            output_file = output_dir / f"sample_{i:04d}_snid_{data['id']}.png"
            print(f"Processing sample {i+1}/{args.num_samples}: SNID {data['id']}")
            visualize_single_heatmap(data, output_file=str(output_file))

    print(f"\nDone! Plots saved to: {output_dir}")

if __name__ == '__main__':
    main()
