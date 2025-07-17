#!/usr/bin/env python3
"""
Audio and Text Quality Analysis Pipeline for Hindi ASR/TTS

Performs audio quality checks (duration, energy, speech, whistle), text quality analysis (WER, CER, BERTScore, Levenshtein, word ratio), and error classification.

CONFIGURABLE PARAMETERS:
- Audio quality thresholds (duration, RMS, VAD, whistle detection)
- Text analysis parameters (BERTScore thresholds, error classification)
- Processing settings (GPU usage, worker count, batch size)
- All parameters are documented inline with their default values and tuning guidance
"""
import warnings
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Tuple
import re
import logging

# Suppress pkg_resources deprecation warning from webrtcvad
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

import numpy as np
import torch
import torchaudio
import webrtcvad
import librosa
from bert_score import BERTScorer
from jiwer import wer, cer
from rich.progress import Progress
from Levenshtein import distance as levenshtein_distance
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from indicnlp.tokenize import indic_tokenize
import string

# Patch for NumPy compatibility (for older libraries)
np.complex = complex
np.float = float

def detect_beep_with_confidence(y, sr, max_duration_sec=0.06, confidence_cutoff_strict=0.010, confidence_cutoff_soft=0.020):
    """
    Detects beep-like sounds within the first 60 milliseconds of an audio signal.
    Returns True if beep is detected, else False.

    CONFIGURABLE PARAMETERS:
        max_duration_sec (float): Maximum duration to analyze for beep (default=0.06s)
            - Increase to catch beeps that occur later in the audio
            - Decrease for faster processing
        confidence_cutoff_strict (float): High confidence if detected before this time (default=0.010s)
            - Lower values = stricter beep detection
            - Higher values = more lenient detection
        confidence_cutoff_soft (float): Soft confidence if detected before this time (default=0.020s)
            - Secondary threshold for borderline cases
            - Should be > confidence_cutoff_strict

    TUNING GUIDANCE:
        - For strict beep filtering: lower confidence_cutoff_strict to 0.005
        - For lenient beep filtering: increase confidence_cutoff_strict to 0.015
        - For different beep types: adjust max_duration_sec based on expected beep length
    """
    try:
        n_samples = int(sr * max_duration_sec)
        y_start = y[:n_samples]
        y_mono = np.mean(y_start, axis=0) if y_start.ndim == 2 else y_start

        hop_length = 256
        frame_length = 1024

        rms = librosa.feature.rms(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]
        centroid = librosa.feature.spectral_centroid(y=y_mono, sr=sr, hop_length=hop_length)[0]
        bandwidth = librosa.feature.spectral_bandwidth(y=y_mono, sr=sr, hop_length=hop_length)[0]
        zcr = librosa.feature.zero_crossing_rate(y=y_mono, frame_length=frame_length, hop_length=hop_length)[0]
        flatness = librosa.feature.spectral_flatness(y=y_mono, hop_length=hop_length)[0]
        rolloff = librosa.feature.spectral_rolloff(y=y_mono, sr=sr, hop_length=hop_length)[0]

        satisfied = []
        for i in range(len(rms)):
            timestamp = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            if timestamp > max_duration_sec:
                break

            conditions = [
                rms[i] > 0.00005,
                centroid[i] > 2200,
                bandwidth[i] < 2500,
                zcr[i] > 0.18,
                flatness[i] > 0.18,
                rolloff[i] > 4500
            ]
            count = sum(conditions)
            strong_combo = rms[i] > 0.00007 and flatness[i] > 0.20
            satisfied.append((count >= 5 or strong_combo, timestamp))

        for i in range(len(satisfied) - 1):
            valid1, ts1 = satisfied[i]
            valid2, ts2 = satisfied[i + 1]
            if valid1 and valid2:
                if ts1 <= confidence_cutoff_soft:
                    return bool(ts1 <= confidence_cutoff_strict)  # Only return True if high confidence (≤0.010s)

        return False
        
    except Exception as e:
        logging.error(f"Error in beep detection: {str(e)}")
        return False

def detect_whistling_final(y, sr, flatness_threshold=0.015, 
                         min_sustain_seconds=0.4, energy_threshold=-80, 
                         rms_threshold_db=0, band_width=300):
    """
    Detects whistling-like tones in an audio signal using spectral flatness and band-limited RMS.
    Scans 2–6 kHz for 2 or more flat, energetic, sustained bands.
    Returns True if whistling is detected, else False.

    CONFIGURABLE PARAMETERS:
        flatness_threshold (float): Lower values make whistle detection stricter (default=0.015)
            - Range: 0.01-0.03
            - Lower = more sensitive to whistling
            - Higher = less sensitive, may miss subtle whistles
        min_sustain_seconds (float): Minimum duration for a whistle band to be considered (default=0.4s)
            - Range: 0.2-1.0 seconds
            - Lower = catch shorter whistles
            - Higher = only catch sustained whistles
        energy_threshold (float): Minimum mean dB for a band to be considered energetic (default=-80dB)
            - Range: -90 to -60 dB
            - Higher = ignore quieter whistles
            - Lower = catch very quiet whistles
        rms_threshold_db (float): Minimum overall RMS dB in 2-6kHz band (default=0)
            - Range: -20 to +10 dB
            - Lower = allow quieter whistles
            - Higher = only detect loud whistles
        band_width (int): Width of frequency bands in Hz (default=300)
            - Range: 200-500 Hz
            - Smaller = finer detection, slower processing
            - Larger = coarser detection, faster processing

    TUNING GUIDANCE:
        - For strict whistle filtering: increase flatness_threshold to 0.02, energy_threshold to -70
        - For lenient whistle filtering: decrease flatness_threshold to 0.01, energy_threshold to -90
        - For different whistle types: adjust band_width based on expected whistle frequency range
    """
    hop_length = 512
    S = librosa.stft(y, hop_length=hop_length)
    S_db = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    freqs = librosa.fft_frequencies(sr=sr)

    # Compute 2–6 kHz RMS energy
    power = np.abs(S) ** 2
    band_idx_all = np.where((freqs >= 2000) & (freqs <= 6000))[0]
    band_power = np.mean(power[band_idx_all, :], axis=0)
    rms_db_band = 10 * np.log10(np.mean(band_power) + 1e-9)
    if rms_db_band < rms_threshold_db:
        return False

    # Split 2–6 kHz into smaller bands
    band_indices = []
    for start in range(2000, 6000, band_width):
        idx = np.where((freqs >= start) & (freqs < start + band_width))[0]
        if len(idx) > 0:
            band_indices.append(idx)

    time_per_frame = hop_length / sr
    frames_required = int(min_sustain_seconds / time_per_frame)

    for i in range(S_db.shape[1] - frames_required):
        flat_bands = 0
        for idx in band_indices:
            band_energy = np.mean(S_db[idx, i:i+frames_required], axis=0)
            if np.std(band_energy) < flatness_threshold and np.mean(band_energy) > energy_threshold:
                flat_bands += 1
            if flat_bands >= 2:
                return True
    return False

class AudioAnalyzer:
    """
    Main class for audio and text quality analysis and error classification.
    Handles audio checks, text metrics, and batch processing.

    KEY FEATURES:
    - Audio quality assessment (duration, energy, speech, noise detection)
    - Text quality analysis (WER, CER, BERTScore, semantic similarity)
    - Error classification into actionable categories
    - Batch processing with configurable parallelism
    - GPU acceleration for BERTScore calculations
    """
    def __init__(self, use_gpu: bool = True, num_workers: int = 16, batch_size: int = 32, duration_threshold: float = 0.2):
        """
        Initialize the analyzer with configuration for GPU, workers, batch size, and duration threshold.

        CONFIGURABLE PARAMETERS:
            use_gpu (bool): Use GPU for BERTScore and torch operations (default=True)
                - Set False for CPU-only processing (slower but no GPU required)
                - Requires CUDA-compatible GPU and torch with CUDA support
            num_workers (int): Number of threads for batch processing (default=16)
                - Range: 1-32 (depends on CPU cores)
                - Increase for faster processing if you have more CPU cores
                - Decrease if system becomes unresponsive
            batch_size (int): Number of items to process per batch (default=32)
                - Range: 8-64
                - Increase for speed, decrease if you run out of memory
                - Larger batches = better GPU utilization but more memory usage
            duration_threshold (float): Minimum audio duration in seconds (default=0.2)
                - Range: 0.1-1.0 seconds
                - Raise to filter out shorter clips (may indicate poor TTS)
                - Lower to include very short audio clips

        PERFORMANCE TUNING:
            - For high-end systems: num_workers=24, batch_size=48
            - For low-end systems: num_workers=4, batch_size=16
            - For memory-constrained systems: batch_size=16, use_gpu=False
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.duration_threshold = duration_threshold
        
        # VAD aggressiveness (0-3); 3 is most aggressive
        # CONFIGURABLE: Change for stricter/looser speech detection
        # - 0: Least aggressive (may miss speech)
        # - 1: Low aggressive
        # - 2: Medium aggressive (default for most use cases)
        # - 3: Most aggressive (may classify noise as speech)
        self.vad = webrtcvad.Vad(3)
        
        # Hindi text normalizer for consistent text processing
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        
        # Multilingual BERT scorer for semantic similarity
        # CONFIGURABLE: Change model_type for different BERT models
        # - "bert-base-multilingual-cased": Good balance of speed/accuracy
        # - "xlm-roberta-base": Better multilingual performance, slower
        # - "distilbert-base-multilingual-cased": Faster, slightly lower accuracy
        self.bert_scorer = BERTScorer(
            lang="hi",
            model_type="bert-base-multilingual-cased",
            device="cuda" if use_gpu else "cpu"
        )

    def check_audio_quality(self, audio_path: str) -> Tuple[bool, str]:
        """
        Perform a series of audio quality checks:
        1. Duration check
        2. RMS energy check
        3. Speech content check (VAD)
        4. Beep detection
        5. Whistle detection
        
        Returns (True, "Pass") if all checks pass, else (False, reason).

        CONFIGURABLE PARAMETERS (hardcoded in function):
            - duration_threshold: See __init__ (default=0.2s)
            - RMS dB threshold: -25dB (line ~200) - change for stricter/looser energy filtering
            - VAD aggressiveness: See __init__ (default=3)
            - speech_ratio threshold: 0.2 (line ~220) - change for stricter/looser speech filtering
            - Whistle detection parameters: See detect_whistling_final function
            - Beep detection parameters: See detect_beep_with_confidence function

        TUNING GUIDANCE:
            - For strict audio filtering: increase RMS threshold to -20dB, speech_ratio to 0.3
            - For lenient audio filtering: decrease RMS threshold to -30dB, speech_ratio to 0.1
            - For different audio types: adjust VAD aggressiveness based on expected speech clarity
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # 1. Duration check
            duration = len(y) / sr
            if duration < self.duration_threshold:
                return False, "Poor TTS Generation (too short)"
            
            # 2. RMS Energy check
            # CONFIGURABLE: Change -25 to adjust energy threshold
            # - Higher values (-20 to -15): stricter filtering, may reject quiet speech
            # - Lower values (-30 to -25): more lenient, may accept noise
            rms = np.sqrt(np.mean(y ** 2))
            rms_db = 20 * np.log10(rms + 1e-9)
            if rms_db < -25:  # CONFIGURABLE: Energy threshold in dB
                return False, "Noise (low energy)"
            
            # 3. Speech content check (VAD)
            audio_16bit = (y * 32768).astype(np.int16)
            frame_duration = 30  # ms (CONFIGURABLE: change for different VAD frame sizes)
            frame_size = int(sr * frame_duration / 1000)
            speech_frames = 0
            total_frames = 0
            for i in range(0, len(audio_16bit) - frame_size, frame_size):
                frame = audio_16bit[i:i + frame_size]
                if self.vad.is_speech(frame.tobytes(), sr):
                    speech_frames += 1
                total_frames += 1
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            
            # CONFIGURABLE: Change 0.2 to adjust speech content threshold
            # - Higher values (0.3-0.5): stricter filtering, requires more speech
            # - Lower values (0.1-0.2): more lenient, accepts audio with less speech
            if speech_ratio < 0.2:  # CONFIGURABLE: Speech ratio threshold
                return False, "Noise (low speech content)"
            
            # 4. Beep check (fast, early detection)
            # CONFIGURABLE: See detect_beep_with_confidence function for parameters
            if detect_beep_with_confidence(y, sr):
                return False, "Noise (beep detected)"
            
            # 5. Whistle check
            # CONFIGURABLE: See detect_whistling_final function for parameters
            if detect_whistling_final(y, sr):
                return False, "Noise (whistling detected)"
            
            return True, "Pass"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

    def analyze_text_quality(self, pred_text: str, ground_truth: str) -> Dict:
        """
        Analyze text quality between prediction and ground truth using:
        - WER (Word Error Rate)
        - CER (Character Error Rate)
        - BERTScore (semantic similarity)
        - Levenshtein distance (edit distance)
        - Word ratio (prediction length vs ground truth)
        
        Returns a dict of all metrics and classification.

        CONFIGURABLE PARAMETERS:
            - BERT model type: See __init__ (default: bert-base-multilingual-cased)
            - Normalization/tokenization logic: Edit normalize_hindi_text function
            - Error classification thresholds: See _classify_error function

        PERFORMANCE NOTES:
            - Early exit for perfect matches (WER=0, CER=0) to avoid expensive BERTScore
            - BERTScore is the most computationally expensive operation
            - Hindi text normalization includes Unicode normalization and punctuation removal
        """
        def normalize_hindi_text(text):
            """
            Normalize Hindi text: Unicode normalization, remove Latin/punctuation, whitespace, tokenize.
            
            CONFIGURABLE: Modify this function to change text preprocessing:
            - Add/remove specific character filtering
            - Change tokenization method
            - Add custom normalization rules
            """
            if not isinstance(text, str):
                text = str(text)
            try:
                normalized = self.normalizer.normalize(text)
                normalized = re.sub(f'[{string.ascii_letters}{string.punctuation}]', '', normalized)
                normalized = ' '.join(normalized.split())
                tokens = indic_tokenize.trivial_tokenize(normalized)
                normalized = ' '.join(tokens)
                return normalized.strip()
            except Exception as e:
                logging.error(f"Error in Hindi normalization: {str(e)}")
                logging.error(f"Original text: {text}")
                return text  # Fallback to original
        try:
            # Normalize and compute WER/CER
            norm_pred = normalize_hindi_text(pred_text)
            norm_ground = normalize_hindi_text(ground_truth)
            word_error_rate = wer(norm_ground, norm_pred)
            char_error_rate = cer(norm_ground, norm_pred)
            
            # Early exit for perfect match (optimization)
            if word_error_rate == 0 and char_error_rate == 0:
                return {
                    "wer": word_error_rate,
                    "cer": char_error_rate,
                    "bert_score": 1.0,
                    "levenshtein": 0,
                    "word_ratio": 1.0,
                    "classification": "Perfect Match"
                }
            
            # Expensive metrics for non-perfect matches
            bert_score = self.bert_scorer.score([pred_text], [ground_truth])[2].mean().item()
            lev_distance = levenshtein_distance(norm_pred, norm_ground)
            pred_words = len(indic_tokenize.trivial_tokenize(norm_pred))
            ground_words = len(indic_tokenize.trivial_tokenize(norm_ground))
            word_ratio = pred_words / ground_words if ground_words > 0 else float('inf')
            classification = self._classify_error(bert_score, lev_distance, word_ratio)
            return {
                "wer": word_error_rate,
                "cer": char_error_rate,
                "bert_score": bert_score,
                "levenshtein": lev_distance,
                "word_ratio": word_ratio,
                "classification": classification
            }
        except Exception as e:
            logging.error(f"Error in text quality analysis: {str(e)}")
            logging.error(f"Ground truth: {ground_truth}")
            logging.error(f"Prediction: {pred_text}")
            raise

    def _classify_error(self, bert_score: float, levenshtein: int, word_ratio: float) -> str:
        """
        Classify error type based on BERTScore, Levenshtein distance, and word ratio.
        Returns a string label for the error category.

        CONFIGURABLE PARAMETERS:
            - BERTScore threshold: 0.8 (line ~280) - change for stricter/looser semantic match
            - Levenshtein thresholds: 10, 100 (lines ~281, 285) - change for stricter/looser edit distance
            - Word ratio threshold: 1.1 (line ~290) - change for stricter/looser word ratio

        CLASSIFICATION LOGIC:
            - Perfect Match: WER=0, CER=0 (handled in analyze_text_quality)
            - Minor ASR Error: High semantic similarity (BERTScore≥0.8) + small edits (Levenshtein<10)
            - TTS Omission: High semantic similarity + word_ratio≤1 (prediction shorter than ground truth)
            - TTS Hallucination: High semantic similarity + word_ratio>1 (prediction longer than ground truth)
            - TTS Unintelligible Output: Low semantic similarity (BERTScore<0.8) + reasonable length
            - TTS Hallucination: Low semantic similarity + excessive length

        TUNING GUIDANCE:
            - For strict classification: increase BERTScore threshold to 0.85, decrease Levenshtein to 5
            - For lenient classification: decrease BERTScore threshold to 0.75, increase Levenshtein to 15
            - For different languages: adjust BERTScore threshold based on language-specific performance
        """
        # CONFIGURABLE: BERTScore threshold for semantic similarity
        # - Higher values (0.85-0.9): stricter semantic matching
        # - Lower values (0.7-0.8): more lenient semantic matching
        if bert_score >= 0.8:  # CONFIGURABLE: Semantic similarity threshold
            # CONFIGURABLE: Levenshtein threshold for minor errors
            # - Lower values (5-10): stricter edit distance
            # - Higher values (15-20): more lenient edit distance
            if levenshtein < 10:  # CONFIGURABLE: Minor error threshold
                return "Minor ASR Error"
            elif levenshtein <= 100:  # CONFIGURABLE: Major error threshold
                if word_ratio > 1:
                    return "TTS Hallucination"  # Merged with TTS Insertion
                else:  # word_ratio <= 1
                    return "TTS Omission"
            else:  # levenshtein > 100
                return "TTS Hallucination"
        else:  # bert_score < 0.8
            # CONFIGURABLE: Word ratio threshold for length-based classification
            # - Lower values (1.0-1.1): stricter length matching
            # - Higher values (1.2-1.5): more lenient length matching
            if word_ratio < 1.1 or levenshtein < 10:  # CONFIGURABLE: Word ratio threshold
                return "TTS Unintelligible Output"
            else:
                return "TTS Hallucination"

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of input items (audio/text pairs).
        Runs audio checks first; if passed, runs text analysis.
        Returns a list of result dicts for each item.

        PROCESSING FLOW:
        1. Audio quality check (duration, energy, speech, noise detection)
        2. If audio passes: Text quality analysis (WER, CER, BERTScore, etc.)
        3. If audio fails: Return immediately with audio issue classification

        CONFIGURABLE PARAMETERS:
            - num_workers: See __init__ (default=16)
            - batch_size: See __init__ (default=32)

        PERFORMANCE NOTES:
            - Uses ThreadPoolExecutor for parallel processing
            - Audio checks are CPU-intensive but fast
            - Text analysis (especially BERTScore) is GPU-intensive and slower
            - Early exit for audio failures saves computation time
        """
        results = []
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            for item in batch:
                try:
                    # STEP 1: Audio quality check FIRST (including whistle)
                    future = executor.submit(self.check_audio_quality, item["audio_filepath"])
                    passed_check, reason = future.result()
                    
                    if not passed_check:
                        # Audio failed - return immediately with audio issue
                        result = {
                            "audio_filepath": item["audio_filepath"],
                            "text": item["text"],
                            "pred_text": item["pred_text"],
                            "duration": item["duration"],
                            "audio_check": reason,
                            "classification": reason,  # Audio issue overrides everything
                            "wer": None, "cer": None, "bert_score": None,
                            "levenshtein": None, "word_ratio": None
                        }
                    else:
                        # Audio passed - now do text analysis
                        text_metrics = self.analyze_text_quality(
                            item["pred_text"], 
                            item["text"]
                        )
                        result = {
                            "audio_filepath": item["audio_filepath"],
                            "text": item["text"],
                            "pred_text": item["pred_text"],
                            "duration": item["duration"],
                            "audio_check": "Pass"
                        }
                        result.update(text_metrics)
                    
                    results.append(result)
                    
                except Exception as e:
                    logging.error(f"Error processing item: {str(e)}")
                    logging.error(f"Item: {item}")
                    results.append({
                        "audio_filepath": item["audio_filepath"],
                        "text": item.get("text", ""),
                        "pred_text": item.get("pred_text", ""),
                        "duration": item.get("duration", 0.0),
                        "audio_check": f"Error: {str(e)}",
                        "classification": "Processing Error"
                    })
        
        return results

def format_duration(seconds):
    """Convert seconds to hours or seconds with appropriate formatting."""
    hours = seconds / 3600.0
    if hours < 1:
        return f"{seconds:.2f}s"
    return f"{hours:.2f}h"

def main():
    """
    Main entry point for the analysis pipeline.
    Handles argument parsing, file I/O, batch processing, and reporting.

    COMMAND-LINE ARGUMENTS (all configurable):
        --reference_json: Input JSONL file with audio_filepath, text, pred_text, duration
        --output_json: Output JSONL file for detailed results
        --output_csv: Output CSV file for summary statistics
        --use_gpu: Enable GPU acceleration for BERTScore (recommended if available)
        --num_workers: Number of parallel threads (default=16)
        --batch_size: Items per batch (default=32)
        --duration_threshold: Minimum audio duration in seconds (default=0.2)

    OUTPUT FILES:
        - JSONL: Detailed results for each audio file with all metrics
        - CSV: Summary statistics grouped by error categories
        - Console: Rich-formatted summary with progress bars and tables

    ERROR HANDLING:
        - Validates input file format and required fields
        - Graceful handling of processing errors
        - Detailed logging for debugging
    """
    parser = argparse.ArgumentParser(description="Audio and Text Quality Analyzer")
    parser.add_argument("--reference_json", type=str, required=True,
                       help="Input JSONL file containing audio_filepath, text, pred_text, duration")
    parser.add_argument("--output_json", type=str, required=True,
                       help="Output JSONL file for detailed results")
    parser.add_argument("--output_csv", type=str, required=True,
                       help="Output CSV file for summary statistics")
    parser.add_argument("--use_gpu", action="store_true",
                       help="Use GPU for BERTScore (recommended if available)")
    parser.add_argument("--num_workers", type=int, default=16,
                       help="Number of threads for parallel processing (default=16)")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for processing (default=32)")
    parser.add_argument("--duration_threshold", type=float, default=0.2,
                       help="Minimum duration in seconds for audio to be considered valid (default=0.2)")
    args = parser.parse_args()

    # Load input data
    try:
        with open(args.reference_json, 'r', encoding='utf-8') as f:
            data = [json.loads(line) for line in f]
    except FileNotFoundError:
        print(f"❌ Error: Input file not found: {args.reference_json}")
        return 1
    except json.JSONDecodeError as e:
        print(f"❌ Error: Invalid JSON in input file: {e}")
        return 1
    except Exception as e:
        print(f"❌ Error reading input file: {e}")
        return 1

    # Validate data structure
    required_keys = ["audio_filepath", "text", "pred_text", "duration"]
    for i, item in enumerate(data):
        missing_keys = [key for key in required_keys if key not in item]
        if missing_keys:
            print(f"❌ Error: Missing required keys {missing_keys} in item {i}")
            return 1

    # Initialize analyzer with command-line parameters
    analyzer = AudioAnalyzer(
        use_gpu=args.use_gpu,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        duration_threshold=args.duration_threshold
    )

    # Process data in batches with progress tracking
    results = []
    with Progress() as progress:
        task = progress.add_task("Analyzing...", total=len(data))
        for i in range(0, len(data), args.batch_size):
            batch = data[i:i + args.batch_size]
            batch_results = analyzer.process_batch(batch)
            results.extend(batch_results)
            progress.update(task, advance=len(batch))

    # Save detailed results to JSONL
    try:
        with open(args.output_json, 'w', encoding='utf-8') as f:
            for result in results:
                json.dump(result, f, ensure_ascii=False)
                f.write('\n')
    except Exception as e:
        print(f"❌ Error writing output JSON: {e}")
        return 1

    # Define error/reporting categories for summary
    # CONFIGURABLE: Modify these categories to change the summary grouping
    category_groups = {
        "Audio Quality Issues": {
            "color": "red",
            "categories": ["Noise (low energy)", "Noise (low speech content)", 
                          "Poor TTS Generation (too short)", "Noise (beep detected)", "Noise (whistling detected)"],
            "description": "Issues detected in audio before ASR processing"
        },
        "Perfect Matches": {
            "color": "green",
            "categories": ["Perfect Match"],
            "description": "Exact matches between prediction and ground truth"
        },
        "Minor Issues": {
            "color": "yellow",
            "categories": ["Minor ASR Error"],
            "description": "Small differences with high semantic similarity"
        },
        "TTS Issues": {
            "color": "magenta",
            "categories": ["TTS Omission", "TTS Unintelligible Output", "TTS Hallucination"],
            "description": "Major TTS-related problems"
        },
        "Miscellaneous": {
            "color": "white",
            "categories": ["Processing Error", "Miscellaneous"],
            "description": "Other issues and edge cases"
        }
    }

    # Compute statistics for each category
    total_files = len(results)
    durations = {}  # Track total duration for each category
    total_duration = 0.0
    
    for result in results:
        key = result.get("classification", result.get("audio_check", "Unknown"))
        # Check if classification matches any existing category
        found_in_categories = False
        for group_info in category_groups.values():
            if key in group_info["categories"]:
                found_in_categories = True
                break
        # If not found in any category, put it in Miscellaneous
        if not found_in_categories and key not in ["Processing Error"]:
            key = "Miscellaneous"
        # Get duration from the original data
        try:
            for item in data:
                if item["audio_filepath"] == result["audio_filepath"]:
                    duration = float(item.get("duration", 0.0))
                    durations[key] = durations.get(key, 0.0) + duration
                    total_duration += duration
                    break
        except (KeyError, ValueError, TypeError) as e:
            logging.error(f"Error processing duration for {result['audio_filepath']}: {e}")
            durations[key] = durations.get(key, 0.0) + 0.0
            total_duration += 0.0
    
    # Prepare data for both display and CSV
    stats_data = []
    
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.layout import Layout
    
    console = Console()
    
    # Print summary header
    total_hours = total_duration / 3600.0
    console.print("\n")
    console.print(Panel(
        f"[bold white]Audio Analysis Results Summary[/]\n" +
        f"[dim]Total Files: {total_files} | Total Duration: {format_duration(total_duration)} ({total_hours:.2f} hours)[/]",
        border_style="blue",
        expand=False,
        padding=(1, 2)
    ))
    console.print("\n")
    
    # Process and display each category group
    for group, info in category_groups.items():
        group_stats = []
        group_duration = sum(durations.get(cat, 0.0) for cat in info["categories"])
        group_duration_percentage = (group_duration / total_duration) * 100 if total_duration > 0 else 0
        
        if group_duration > 0:
            # Create table for this group
            table = Table(
                show_header=True,
                header_style=f"bold {info['color']}",
                border_style=info['color'],
                title=f"[bold {info['color']}]{group}[/]",
                title_style=f"bold {info['color']}",
                caption=f"[dim]{info['description']}[/]",
                caption_style=f"dim {info['color']}",
                padding=(0, 1),
                show_edge=True,
                show_lines=True
            )
            # Add columns
            table.add_column("Category", style=f"{info['color']}", width=30)
            table.add_column("Duration", justify="right", style=f"bold {info['color']}", width=15)
            table.add_column("Percentage", justify="right", style=info['color'], width=12)
            # Add rows
            for cat in info["categories"]:
                if cat in durations:
                    duration = durations.get(cat, 0.0)
                    duration_percentage = (duration / total_duration) * 100 if total_duration > 0 else 0
                    table.add_row(
                        cat,
                        format_duration(duration),
                        f"{duration_percentage:.1f}%"
                    )
                    group_stats.append({
                        "Group": group,
                        "Category": cat,
                        "Duration": format_duration(duration),
                        "Duration_Percentage": f"{duration_percentage:.1f}%"
                    })
            # Add table to layout
            console.print(table)
            console.print(f"[dim]{group}: {format_duration(group_duration)} ({group_duration_percentage:.1f}% of total)[/]")
            console.print("\n")
        stats_data.extend(group_stats)
    
    # Export statistics to CSV
    import csv
    try:
        with open(args.output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=["Group", "Category", "Duration", "Duration_Percentage"])
            writer.writeheader()
            writer.writerows(stats_data)
        console.print(Panel(
            f"[bold green]Analysis Complete[/]\n" +
            f"[dim]Results exported to:[/]\n" +
            f"[dim]• JSON: {args.output_json}[/]\n" +
            f"[dim]• CSV:  {args.output_csv}[/]",
            border_style="green",
            expand=False,
            padding=(1, 2)
        ))
    except Exception as e:
        print(f"Error writing output CSV: {e}")
        return 1

    return 0

if __name__ == "__main__":
    main() 
