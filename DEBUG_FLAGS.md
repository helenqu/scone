# SCONE Debug Flags Documentation

## Overview
The debug flag system allows developers to switch between different implementations and enable various debugging features without modifying the code.

## Usage

### Command Line (overrides config file)
```bash
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag 902
```

### Config File
```yaml
# In your SCONE_TEST.INPUT or config file:
debug_flag: 902
```

## Debug Flag Values

| Flag | Mode | Description |
|------|------|-------------|
| **0** | PRODUCTION | Default production mode, minimal logging |
| **1** | VERBOSE | Enable verbose logging throughout |
| **900** | LEGACY_RETRIEVE | Use legacy `_retrieve_data_legacy()` implementation |
| **901** | REFAC_RETRIEVE | Use refactored `_retrieve_data()` with basic logging |
| **902** | REFAC_RETRIEVE_VERBOSE | Use refactored `_retrieve_data()` with verbose logging |
| **1000+** | RESERVED | Reserved for future debug modes |

## Examples

### Testing Legacy vs Refactored Implementation
```bash
# Test with legacy implementation
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag 900

# Test with refactored implementation (verbose)
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag 902
```

### Enable Verbose Logging Only
```bash
$HOME/soft/scone/model_utils.py --config_path SCONE_TEST.INPUT --debug_flag 1
```

## Implementation Details

The debug flag system is implemented in:
- `SconeClassifier.__init__()`: Reads debug_flag from config
- `SconeClassifier._setup_debug_modes()`: Configures debug settings
- `SconeClassifier.run()`: Switches implementations based on flag

## Priority Order
1. Command-line `--debug_flag` (highest priority)
2. Config file `debug_flag` setting
3. Default value of 0 (production mode)

## Adding New Debug Modes

To add a new debug mode:
1. Add a new constant in `_setup_debug_modes()`:
   ```python
   self.DEBUG_MODES['NEW_MODE'] = 903
   ```
2. Add logic in the appropriate method to handle the new mode
3. Document the new flag in this file

## Notes
- Debug flags are primarily for development and testing
- Production deployments should use flag 0 (default)
- Verbose modes may significantly increase log output
- Legacy mode (900) is useful for comparing implementations