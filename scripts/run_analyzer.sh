#!/bin/bash

# Configuration
INPUT_JSON="/projects/data/ttsteam/repos/aaryan/checkpoint-analyzer/subset_predictions_all.json"
OUTPUT_JSON="outputs/analysis_results_predictions_all.jsonl"
OUTPUT_CSV="outputs/analysis_statistics_predictions_all.csv"
NUM_WORKERS=16
BATCH_SIZE=124
DURATION_THRESHOLD=0.2  # Minimum duration in seconds for audio to be considered valid

# Check if input file exists
if [ ! -f "$INPUT_JSON" ]; then
    echo "Error: Input file not found: $INPUT_JSON"
    exit 1
fi

# Check if Python and required packages are available
if ! command -v python &> /dev/null; then
    echo "Error: Python is not installed or not in PATH"
    exit 1
fi

# Display configuration
echo "=== Audio Analysis Configuration ==="
echo "Input:        ${INPUT_JSON}"
echo "Output JSON:  ${OUTPUT_JSON}"
echo "Output CSV:   ${OUTPUT_CSV}"
echo "Workers:      ${NUM_WORKERS}"
echo "Batch Size:   ${BATCH_SIZE}"
echo "Duration Threshold: ${DURATION_THRESHOLD}s"
echo "GPU:          Enabled"
echo ""
echo "Analysis Features:"
echo "  • Audio Quality Checks (Duration, Energy, Speech Content)"
echo "  • Whistle Detection"
echo "  • Text Quality Analysis (WER, CER, BERT Score)"
echo "  • Error Classification (12 categories)"
echo "  • Duration-based Statistics"
echo "  • Automatic Miscellaneous categorization"
echo "=================================="

# Run the analyzer
echo "Starting analysis..."
python analyze.py \
    --reference_json ${INPUT_JSON} \
    --output_json ${OUTPUT_JSON} \
    --output_csv ${OUTPUT_CSV} \
    --num_workers ${NUM_WORKERS} \
    --batch_size ${BATCH_SIZE} \
    --duration_threshold ${DURATION_THRESHOLD} \
    --use_gpu

# Check if analysis completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "Analysis completed successfully!"
    echo "Results saved to:"
    echo "  • JSON: ${OUTPUT_JSON}"
    echo "  • CSV:  ${OUTPUT_CSV}"
else
    echo ""
    echo "Analysis failed with exit code $?"
    exit 1
fi 
