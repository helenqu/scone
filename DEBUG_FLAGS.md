# SCONE Debug Flags Documentation

*Last Updated: 30th October 2025*

## Overview
The debug flag system allows developers to switch between different data retrieval implementations for testing and comparison purposes. Verbose logging is now controlled independently via a separate flag.

## Quick Reference

| Flag | Implementation | Description |
|------|----------------|-------------|
| **0** | Refactored (Default) | Production mode using optimized implementation with memory monitoring |
| **-901** | Legacy | Original implementation for comparison or fallback purposes |

**Verbose Logging**: Use `--verbose` or `-v` flag (works with any implementation)

## Usage

### Command Line (overrides config file)

```bash
# Default (refactored implementation, quiet)
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT

# Legacy implementation
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag -901

# Refactored with verbose logging
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --verbose

# Legacy with verbose logging
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag -901 --verbose

# Short form for verbose
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT -v
```

### Config File

```yaml
# In your SCONE_TEST.INPUT or config file:
debug_flag: 0              # Default - refactored implementation
debug_flag: -901           # Legacy implementation

# Verbose logging (separate control)
verbose_data_loading: True  # Enable verbose logging
```

## Debug Flag Values (Simplified)

### Flag 0: Production Mode (Default)
- **Implementation**: Refactored `_retrieve_data()`
- **Features**:
  - Enhanced memory monitoring
  - Intelligent streaming threshold adjustment
  - Auto-configuration for large datasets
  - Optimized for production use
- **When to use**: Normal production runs (default)

### Flag -901: Legacy Mode
- **Implementation**: Legacy `_retrieve_data_legacy()`
- **Features**:
  - Original implementation
  - No auto-configuration
  - Fixed behavior for reproducibility
- **When to use**:
  - Comparing with legacy results
  - Troubleshooting new features
  - Fallback if issues occur with refactored version

### Verbose Logging: `--verbose` or `-v`
- **Control**: Independent of implementation choice
- **Features**:
  - Detailed progress tracking during data loading
  - More frequent batch/chunk progress reports
  - Estimated time remaining for long operations
  - Enhanced memory usage reporting
- **When to use**: Debugging, monitoring long-running jobs

## Examples

### Testing Legacy vs Refactored Implementation

```bash
# Test with refactored (default)
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT

# Test with legacy implementation
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag -901

# Compare both with verbose logging
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --verbose
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag -901 --verbose
```

### Memory-Constrained Environments

```bash
# Force streaming with refactored implementation
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --force_streaming --verbose

# Custom streaming threshold
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --streaming_threshold 50000
```

## Implementation Details

### Code Structure
The debug flag system is implemented in:
- `SconeClassifier.__init__()`: Reads debug_flag from config
- `SconeClassifier._setup_debug_modes()`: Configures debug settings (simplified to 2 modes)
- `SconeClassifier.predict()`: Switches implementations based on flag
- `SconeClassifier._split_and_retrieve_data()`: Uses selected implementation

### Key Methods
- **Refactored**: `_retrieve_data()` - Enhanced with monitoring and auto-tuning
- **Legacy**: `_retrieve_data_legacy()` - Original implementation preserved

## Auto-Configuration (New Feature)

When using the refactored implementation (flag 0), SCONE automatically configures memory optimization settings for large datasets:

**Trigger Conditions**:
- Dataset ≥ 75,000 samples
- `--force_streaming` flag enabled
- Total dataset size > 40 GB

**Auto-configured Settings**:
```yaml
enable_micro_batching: true  # Automatically enabled
micro_batch_size: 16        # Fixed optimal value
chunk_size: 400             # Fixed optimal value
```

**Note**: User-specified values always take precedence over auto-configuration.

## Priority Order

1. Command-line `--debug_flag` (highest priority)
2. Config file `debug_flag` setting
3. Default value of 0 (refactored/production mode)

Command-line `--verbose` flag overrides config file `verbose_data_loading` setting.

## Migration from Old System

**Old System** (deprecated as of Oct 30, 2025):
```bash
--debug_flag 1    # Verbose logging with legacy
--debug_flag 901  # Refactored
--debug_flag 902  # Refactored + verbose
```

**New System** (current):
```bash
--verbose         # Verbose logging (works with any flag)
--debug_flag -901 # Legacy implementation
# Flag 0 is default (refactored)
```

## Adding New Debug Modes

To add a new debug mode:

1. Add a new constant in `_setup_debug_modes()`:
   ```python
   self.DEBUG_MODES = {
       'PRODUCTION': 0,
       'LEGACY_RETRIEVE': -901,
       'NEW_MODE': -902,  # Use negative numbers for alternatives
   }
   ```

2. Add logic in the appropriate method to handle the new mode:
   ```python
   elif self.debug_flag == self.DEBUG_MODES['NEW_MODE']:
       logging.info("Debug Mode: Using NEW_MODE")
       # Implementation here
   ```

3. Update this documentation file
4. Update README.md and config_example.yml

## Notes

- **Production use**: Always use flag 0 (default) - it's the recommended, stable version
- **Testing/Comparison**: Use flag -901 to compare with legacy behavior
- **Verbose logging**: Use `--verbose` or `-v` for detailed progress tracking
- **Negative flags**: Convention for alternative/legacy implementations
- **Reserved ranges**: 1000+ reserved for future debug modes

## Troubleshooting

**Problem**: Results differ between flag 0 and -901
**Solution**: This is expected - the refactored version includes optimizations. Use -901 only for comparison or if issues occur.

**Problem**: Too much log output
**Solution**: Remove `--verbose` flag or set `verbose_data_loading: False` in config.

**Problem**: Auto-configuration not triggering
**Solution**: Check dataset size (must be ≥75K samples or >40GB). Explicitly set values in config to override.

## See Also

- [README.md](README.md) - Main documentation with usage examples
- [config/config_example.yml](config/config_example.yml) - Configuration file template
- LSST benchmark results in README.md for performance comparisons
