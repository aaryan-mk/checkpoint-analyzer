#!/usr/bin/env python3
"""
Audio and Text Quality Analysis Pipeline for Hindi ASR/TTS

Performs audio quality checks (duration, energy, speech, whistle), text quality analysis (WER, CER, BERTScore, Levenshtein, word ratio), and error classification.
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

    Parameters (modify these thresholds as needed):
        max_duration_sec (float): Maximum duration to analyze for beep (default=0.06s)
        confidence_cutoff_strict (float): High confidence if detected before this time (default=0.010s)
        confidence_cutoff_soft (float): Soft confidence if detected before this time (default=0.020s)
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

    Parameters:
        flatness_threshold (float): Lower values make whistle detection stricter (default=0.015). Increase to be more lenient.
        min_sustain_seconds (float): Minimum duration for a whistle band to be considered (default=0.4s). Decrease to catch shorter whistles.
        energy_threshold (float): Minimum mean dB for a band to be considered energetic (default=-80dB). Raise to ignore quieter whistles.
        rms_threshold_db (float): Minimum overall RMS dB in 2-6kHz band (default=0). Lower to allow quieter whistles.
        band_width (int): Width of frequency bands in Hz (default=300). Increase for coarser detection.
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
    """
    def __init__(self, use_gpu: bool = True, num_workers: int = 16, batch_size: int = 32, duration_threshold: float = 0.2):
        """
        Initialize the analyzer with configuration for GPU, workers, batch size, and duration threshold.

        Parameters:
            use_gpu (bool): Use GPU for BERTScore and torch operations (default=True). Set False for CPU-only.
            num_workers (int): Number of threads for batch processing (default=16). Increase for faster processing if you have more CPU cores.
            batch_size (int): Number of items to process per batch (default=32). Increase for speed, decrease if you run out of memory.
            duration_threshold (float): Minimum audio duration in seconds (default=0.2). Raise to filter out shorter clips.
        """
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.duration_threshold = duration_threshold
        self.vad = webrtcvad.Vad(3)  # VAD aggressiveness (0-3); 3 is most aggressive. Change for stricter/looser speech detection.
        
        # Hindi text normalizer
        self.normalizer = IndicNormalizerFactory().get_normalizer("hi")
        
        # Multilingual BERT scorer for semantic similarity
        self.bert_scorer = BERTScorer(
            lang="hi",
            model_type="bert-base-multilingual-cased",
            device="cuda" if use_gpu else "cpu"
        )

    def check_audio_quality(self, audio_path: str) -> Tuple[bool, str]:
        """
        Perform a series of audio quality checks:
        1. Duration
        2. RMS energy
        3. Speech content (VAD)
        4. Whistle detection
        Returns (True, "Pass") if all checks pass, else (False, reason).
        
        Tunable parameters:
            - duration_threshold (see __init__)
            - RMS dB threshold (hardcoded: -25dB, change below if needed)
            - VAD aggressiveness (see __init__)
            - speech_ratio threshold (hardcoded: 0.2, change below if needed)
            - Whistle detection parameters (see detect_whistling_final)
        """
        try:
            # Load audio file
            y, sr = librosa.load(audio_path, sr=None)
            
            # 1. Duration check
            duration = len(y) / sr
            if duration < self.duration_threshold:
                return False, "Poor TTS Generation (too short)"
            
            # 2. RMS Energy check
            rms = np.sqrt(np.mean(y ** 2))
            rms_db = 20 * np.log10(rms + 1e-9)
            if rms_db < -25:  # Change this threshold for stricter/looser energy filtering
                return False, "Noise (low energy)"
            
            # 3. Speech content check (VAD)
            audio_16bit = (y * 32768).astype(np.int16)
            frame_duration = 30  # ms (change for different VAD frame sizes)
            frame_size = int(sr * frame_duration / 1000)
            speech_frames = 0
            total_frames = 0
            for i in range(0, len(audio_16bit) - frame_size, frame_size):
                frame = audio_16bit[i:i + frame_size]
                if self.vad.is_speech(frame.tobytes(), sr):
                    speech_frames += 1
                total_frames += 1
            speech_ratio = speech_frames / total_frames if total_frames > 0 else 0
            if speech_ratio < 0.2:  # Change this threshold for stricter/looser speech filtering
                return False, "Noise (low speech content)"
            
            # 4. Beep check (fast, early detection)
            if detect_beep_with_confidence(y, sr):  # See function above for tunable params
                return False, "Noise (beep detected)"
            
            # 5. Whistle check
            if detect_whistling_final(y, sr):  # See function above for tunable params
                return False, "Noise (whistling detected)"
            
            return True, "Pass"
            
        except Exception as e:
            return False, f"Error: {str(e)}"

    def analyze_text_quality(self, pred_text: str, ground_truth: str) -> Dict:
        """
        Analyze text quality between prediction and ground truth using:
        - WER, CER
        - BERTScore (semantic similarity)
        - Levenshtein distance
        - Word ratio
        Returns a dict of all metrics and classification.
        
        Tunable parameters:
            - BERT model type (see __init__)
            - Normalization/tokenization logic (edit normalize_hindi_text)
        """
        def normalize_hindi_text(text):
            """Normalize Hindi text: Unicode normalization, remove Latin/punctuation, whitespace, tokenize."""
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
            # Early exit for perfect match
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
        
        Tunable parameters:
            - BERTScore threshold (0.8, see below)
            - Levenshtein thresholds (10, 100, see below)
            - Word ratio threshold (1.1, see below)
        """
        if bert_score >= 0.8:  # Change this threshold for stricter/looser semantic match
            if levenshtein < 10:  # Change for stricter/looser edit distance
                return "Minor ASR Error"
            elif levenshtein <= 100:
                if word_ratio > 1:
                    return "TTS Hallucination"  # Merged with TTS Insertion
                else:  # word_ratio <= 1
                    return "TTS Omission"
            else:  # levenshtein > 100
                return "TTS Hallucination"
        else:  # bert_score < 0.8
            if word_ratio < 1.1 or levenshtein < 10:  # Change 1.1 for stricter/looser word ratio
                return "TTS Unintelligible Output"
            else:
                return "TTS Hallucination"

    def process_batch(self, batch: List[Dict]) -> List[Dict]:
        """
        Process a batch of input items (audio/text pairs).
        Runs audio checks first; if passed, runs text analysis.
        Returns a list of result dicts for each item.
        
        Tunable parameters:
            - num_workers, batch_size (see __init__)
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
    
    Tunable parameters:
        - All command-line arguments (see below)
    """
    parser = argparse.ArgumentParser(description="Audio and Text Quality Analyzer")
    parser.add_argument("--reference_json", type=str, required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_csv", type=str, required=True)
    parser.add_argument("--use_gpu", action="store_true")  # Use GPU for BERTScore (recommended if available)
    parser.add_argument("--num_workers", type=int, default=16)  # Number of threads for parallel processing
    parser.add_argument("--batch_size", type=int, default=32)  # Batch size for processing
    parser.add_argument("--duration_threshold", type=float, default=0.2, help="Minimum duration in seconds for audio to be considered valid.")  # Filter out short audio
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

    analyzer = AudioAnalyzer(
        use_gpu=args.use_gpu,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        duration_threshold=args.duration_threshold
    )

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
