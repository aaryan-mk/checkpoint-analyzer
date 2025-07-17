# Audio Quality Analysis Pipeline

A comprehensive audio and text quality analysis pipeline for Hindi ASR/TTS systems. This tool performs audio quality checks, whistle detection, text quality analysis, and error classification to assess the quality of audio files and their transcriptions.

## Project Structure

```
checkpoint-analyzer/
├── scripts/
│   ├── analyze.py           # Main analysis script
│   └── run_analyzer.sh      # Bash script to run analysis
├── outputs/                 # All analysis outputs (results, statistics, etc.)
├── environment.yml          # Conda environment configuration
└── setup.sh                 # Automated environment setup script
```

## Features

- **Automated Setup**: One-command environment setup with progress tracking
- **Audio Quality Analysis**: Duration, energy, speech content, beep detection, and whistle detection
- **Text Quality Analysis**: WER, CER, BERT score, Levenshtein distance, word ratio
- **Error Classification**: 11 distinct categories for comprehensive error analysis
- **Performance Optimized**: Fast audio filtering with expensive text analysis only when needed
- **Professional Output**: Clean terminal display and CSV statistics export
- **Configurable Parameters**: Tunable thresholds and processing options
- **Environment Management**: Conda-based environment with all dependencies pre-configured

## System Requirements
- **Operating System**: Linux, macOS, or Windows
- **Conda**: Miniconda or Anaconda (required; the setup script handles everything else)
- **CUDA-compatible GPU**: Optional, for BERT scoring acceleration
- **RAM**: 16GB+ recommended for large datasets
- **Storage**: 5GB+ for environment and models

**Note:** Python 3.10+ and all other dependencies are automatically installed by the `setup.sh` script.

## Installation & Setup

### 1. Automated Setup (Recommended)

From the project root (`checkpoint-analyzer/`):

```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create the conda environment
- Install all dependencies
- Verify the installation
- Test the analyzer script

### 2. Activate the Environment

```bash
conda activate audio-analysis
```

## Usage

### Quick Start (Recommended)

1. **Navigate to the scripts directory:**
   ```bash
   cd scripts
   ```
2. **Run the analysis:**
   ```bash
   ./run_analyzer.sh
   ```
   (You may need to run `chmod +x run_analyzer.sh` once to make it executable.)

### Manual Execution

If you want to run the Python script directly for custom arguments:

```bash
cd scripts
python analyze.py \
    --reference_json <input_data.json> \
    --output_json <results.jsonl> \
    --output_csv <statistics.csv> \
    --use_gpu \
    --num_workers 16 \
    --batch_size 124 \
    --duration_threshold 0.2
```

### Configuration and Tuning

For detailed information about all configurable parameters, thresholds, and tuning guidance:

```bash
cd scripts
python analyze.py --help
```

The script includes comprehensive inline documentation for:
- **Audio quality thresholds** (duration, RMS, VAD, beep/whistle detection)
- **Text analysis parameters** (BERTScore, Levenshtein, word ratio thresholds)
- **Processing settings** (GPU usage, worker count, batch size optimization)
- **Performance tuning** recommendations for different system types
- **Error classification** logic and thresholds

All parameters are documented with their default values, valid ranges, and specific tuning guidance for different use cases.

### Script Configuration

- Edit `scripts/run_analyzer.sh` to set:
  - `INPUT_JSON` (input file path)
  - `OUTPUT_JSON`, `OUTPUT_CSV` (output file names, e.g., `../outputs/results.jsonl`, `../outputs/statistics.csv`)
  - `NUM_WORKERS`, `BATCH_SIZE`, `DURATION_THRESHOLD` (processing parameters)

## Scripts Directory

- **`scripts/analyze.py`**: Main Python analysis script (audio/text checks, error classification)
- **`scripts/run_analyzer.sh`**: Bash script to run the analysis with pre-configured settings
- **`scripts/detect_beep.py`**: Standalone beep detection script for testing and validation

**Usage Tips:**
- Always activate the environment before running scripts: `conda activate audio-analysis`
- Always run analysis scripts from within the `scripts/` directory.
- For custom runs, use `python analyze.py --help` for all options.
- Test beep detection independently with `python detect_beep.py` before integration.

## Troubleshooting

- **Conda not found**: Install Miniconda from https://docs.conda.io/en/latest/miniconda.html
- **Environment creation fails**: Check the output from `setup.sh` for errors
- **Script not found**: Make sure you are in the `scripts/` directory
- **Permission denied**: Run `chmod +x scripts/run_analyzer.sh`
- **Python not found**: Activate the environment: `conda activate audio-analysis`
- **Input file not found**: Check the `INPUT_JSON` path in `scripts/run_analyzer.sh`
- **Missing dependencies**: Re-run `./setup.sh` from the project root
- **JSON parsing errors**: Validate your input JSON format

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test with `./setup.sh` and `cd scripts && ./run_analyzer.sh`
5. Submit a pull request

## Error Categories

The pipeline classifies errors into **11 distinct categories** across 5 groups:

### **Audio Quality Issues** (Red) - 5 categories
- `"Noise (low energy)"` - Audio too quiet
- `"Noise (low speech content)"` - Not enough speech detected by VAD
- `"Poor TTS Generation (too short)"` - Audio duration below threshold
- `"Noise (beep detected)"` - High confidence beep detected in first 10ms
- `"Noise (whistling detected)"` - Whistling tones detected

### **Perfect Matches** (Green) - 1 category
- `"Perfect Match"` - Exact match between prediction and ground truth

### **Minor Issues** (Yellow) - 1 category
- `"Minor ASR Error"` - Small differences with high semantic similarity

### **TTS Issues** (Magenta) - 3 categories
- `"TTS Omission"` - TTS missed words
- `"TTS Unintelligible Output"` - Poor semantic match but reasonable length
- `"TTS Hallucination"` - Includes both insertions and major hallucinations

### **Miscellaneous** (White) - 1 category
- `"Processing Error"` - Technical errors during analysis
- `"Miscellaneous"` - Any other unclassified issues

## Audio Quality Pipeline

The audio quality checks are performed in the following order:

1. **Duration Check** - Ensures audio meets minimum length requirement
2. **RMS Energy Check** - Validates audio has sufficient energy
3. **Speech Content (VAD)** - Confirms presence of speech using Voice Activity Detection
4. **Beep Detection** - Fast, early detection of beep artifacts in first 60ms
5. **Whistle Detection** - Full audio analysis for whistling tones

### Beep Detection Features
- **High Sensitivity**: Detects beeps with RMS threshold as low as 0.00005
- **Dual Confidence Levels**: Soft (20ms) and strict (10ms) thresholds
- **Multiple Detection Paths**: 5/6 conditions OR strong combo detection
- **Early Filtering**: Only processes first 60ms for efficiency
- **Strict Classification**: Only flags high confidence (≤10ms) beeps

---

For more details on pipeline logic, error categories, and output formats, see the full documentation sections below in this README.

## Credits

This codebase was built by:
- **svp**
- **amk**
