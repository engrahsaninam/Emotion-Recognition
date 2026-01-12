"""
Modern Speech Emotion Recognition using Wav2Vec2

This module provides state-of-the-art speech emotion recognition using
pre-trained Wav2Vec2 models from HuggingFace. It replaces the legacy
MFCC-based TensorFlow model with a transformer-based approach.

Supported Models:
- superb/wav2vec2-large-superb-er (default, trained on IEMOCAP)
- ehcalabres/wav2vec2-lg-xlsr-en-speech-emotion-recognition (RAVDESS + CREMA-D)
"""

import csv
import os
import subprocess
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torchaudio
import matplotlib.pyplot as plt
from PyQt5.QtCore import QObject, pyqtSignal


# Emotion labels mapping
EMOTION_LABELS = {
    0: 'angry',
    1: 'happy', 
    2: 'neutral',
    3: 'sad',
    # Extended labels for some models
    4: 'fear',
    5: 'surprise',
    6: 'disgust'
}

# SUPERB model labels (IEMOCAP dataset)
SUPERB_LABELS = ['neutral', 'happy', 'sad', 'angry']

# Color mapping for visualization
COLOR_DICT = {
    "neutral": "grey",
    "positive": "green",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "negative": "red",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown"
}


def get_subprocess_kwargs():
    """Get platform-specific subprocess kwargs to hide console window on Windows."""
    kwargs = {}
    if sys.platform == 'win32':
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs['startupinfo'] = startupinfo
    return kwargs


def get_ffmpeg_path():
    """Get the ffmpeg executable path based on platform."""
    if sys.platform == 'win32':
        bundled_path = os.path.join('bin', 'ffmpeg.exe')
        if os.path.exists(bundled_path):
            return os.path.abspath(bundled_path)
    return 'ffmpeg'


class Wav2VecEmotionRecognizer(QObject):
    """
    Modern Speech Emotion Recognition using Wav2Vec2.
    
    This class provides emotion recognition from audio using pre-trained
    transformer models, offering significantly better accuracy than
    traditional MFCC-based approaches.
    """
    
    completed_signal = pyqtSignal(bool)
    path_signals = pyqtSignal(list)
    progress_signal = pyqtSignal(int, int)  # current, total

    def __init__(self, model_name: str = "superb/wav2vec2-large-superb-er"):
        super().__init__()
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.device = None
        self.labels = SUPERB_LABELS
        
        # Audio settings
        self.audio_file = None
        self.trim = 10.0  # Default segment length in seconds
        self.process_all = False
        self.start_time = 0
        self.end_time = 0
        
        # Cross-platform paths
        self.output_base_dir = os.path.join('Output', 'Audio')
        
    def load_models(self):
        """Load the Wav2Vec2 model and processor from HuggingFace."""
        print(f'Loading Wav2Vec2 model: {self.model_name}')
        
        from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2FeatureExtractor
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Using device: {self.device}')
        
        # Load model and processor
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(self.model_name)
        self.model = Wav2Vec2ForSequenceClassification.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()
        
        # Get labels from model config if available
        if hasattr(self.model.config, 'id2label'):
            self.labels = [self.model.config.id2label[i] for i in range(len(self.model.config.id2label))]
        
        print(f'Model loaded successfully. Labels: {self.labels}')
        self.starttime = datetime.now()

    def predict_emotion(self, audio_path: str) -> Tuple[str, Dict[str, float], np.ndarray]:
        """
        Predict emotion from an audio file.
        
        Args:
            audio_path: Path to the audio file
            
        Returns:
            Tuple of (dominant_emotion, emotion_scores, raw_predictions)
        """
        import librosa
        
        # Load audio with librosa (more reliable than torchaudio)
        waveform_np, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
        waveform = torch.tensor(waveform_np).unsqueeze(0)
        
        # Prepare input
        inputs = self.processor(
            waveform.squeeze().numpy(),
            sampling_rate=16000,
            return_tensors="pt",
            padding=True
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            predictions = torch.nn.functional.softmax(logits, dim=-1)
        
        # Get results
        predictions_np = predictions.cpu().numpy()[0]
        predicted_idx = predictions_np.argmax()
        dominant_emotion = self.labels[predicted_idx]
        
        # Create emotion scores dict
        emotion_scores = {
            self.labels[i]: float(predictions_np[i]) * 100 
            for i in range(len(self.labels))
        }
        
        return dominant_emotion, emotion_scores, predictions_np

    def analyze_audio_segment(self, audio_path: str, output_dir: str) -> Tuple[np.ndarray, str]:
        """
        Analyze a single audio segment and generate visualizations.
        
        Args:
            audio_path: Path to the audio file
            output_dir: Directory to save visualizations
            
        Returns:
            Tuple of (predictions_array, dominant_emotion)
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get prediction
        dominant_emotion, emotion_scores, predictions = self.predict_emotion(audio_path)
        
        # Generate waveform visualization
        self._plot_waveform(audio_path, output_dir)
        
        # Generate emotion polar chart
        melspec_path = self._plot_emotion_polar(predictions, output_dir)
        
        # Emit paths signal
        self.path_signals.emit([
            os.path.join(output_dir, "waveform.png"),
            os.path.join(output_dir, "spectrogram.png"),
            melspec_path
        ])
        
        return predictions, dominant_emotion

    def _plot_waveform(self, audio_path: str, output_dir: str):
        """Generate waveform and spectrogram visualizations."""
        import librosa
        import librosa.display
        
        y, sr = librosa.load(audio_path, sr=16000)
        
        # Waveform
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.set_facecolor('#d1d1e0')
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_title('Waveform')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "waveform.png"))
        plt.close()
        
        # Spectrogram
        fig, ax = plt.subplots(figsize=(10, 2))
        fig.set_facecolor('#d1d1e0')
        D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='hz', ax=ax)
        ax.set_title('Spectrogram')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "spectrogram.png"))
        plt.close()

    def _plot_emotion_polar(self, predictions: np.ndarray, output_dir: str) -> str:
        """Generate polar chart visualization of emotion predictions."""
        N = len(predictions)
        ind = predictions.argmax()
        
        color = COLOR_DICT.get(self.labels[ind], "grey")
        sector_colors = [COLOR_DICT.get(label, "grey") for label in self.labels]
        
        fig = plt.figure(figsize=(5, 5))
        fig.set_facecolor("#d1d1e0")
        ax = plt.subplot(111, polar=True)
        
        theta = np.linspace(0.0, 2 * np.pi, N, endpoint=False)
        
        for sector in range(N):
            radii = np.zeros_like(predictions)
            radii[sector] = predictions[sector] * 10
            width = np.pi / 1.8 * predictions
            c = sector_colors[sector]
            ax.bar(theta, radii, width=width, bottom=0.0, color=c, alpha=0.25)
        
        angles = [i / float(N) * 2 * np.pi for i in range(N)]
        angles += angles[:1]
        
        data = list(predictions)
        data += data[:1]
        plt.polar(angles, data, color=color, linewidth=2)
        plt.fill(angles, data, facecolor=color, alpha=0.25)
        
        ax.spines['polar'].set_color('lightgrey')
        ax.set_theta_offset(np.pi / 3)
        ax.set_theta_direction(-1)
        plt.xticks(theta, self.labels)
        ax.set_rlabel_position(0)
        plt.yticks([0, .25, .5, .75, 1], color="grey", size=8)
        
        title = f"Detected emotion: {self.labels[ind]} - {predictions[ind] * 100:.2f}%"
        plt.suptitle(title, color="darkblue", size=10)
        plt.ylim(0, 1)
        plt.subplots_adjust(top=0.75)
        
        melspec_path = os.path.join(output_dir, "emotion_polar.png")
        plt.savefig(melspec_path)
        plt.close()
        
        return melspec_path

    def add_to_csv(self, loop_count: int, dir_path: str, clip_name: str, 
                   dominant: str, confidence: np.ndarray, start_time: float, end_time: float):
        """Add analysis results to CSV file."""
        os.makedirs(dir_path, exist_ok=True)
        
        # Create headers based on available labels
        base_headers = ['clip_name', 'dominant']
        emotion_headers = [label for label in self.labels]
        time_headers = ['start_time', 'end_time']
        headers = base_headers + emotion_headers + time_headers
        
        output_file = os.path.join(dir_path, "output.csv")
        
        if not os.path.exists(output_file) or loop_count == 0:
            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        with open(output_file, 'a', newline="") as f:
            writer = csv.writer(f)
            confidence_values = [float(c) * 100 for c in confidence]
            output_array = [clip_name, dominant] + confidence_values + [start_time, end_time]
            writer.writerow(output_array)

    def calculate_overall_score_index(self, csv_file_path: str) -> Tuple[float, int, int]:
        """Calculate overall sentiment score from CSV results."""
        df = pd.read_csv(csv_file_path, usecols=["dominant"])
        positive_sum = df[df["dominant"] == "happy"].shape[0]
        negative_sum = df[df["dominant"].isin(["sad", "angry", "fear"])].shape[0]
        
        if positive_sum + negative_sum == 0:
            return 0, 0, 0
        
        score_index = (positive_sum - negative_sum) / (positive_sum + negative_sum)
        return score_index, positive_sum, negative_sum

    def add_overall_score_index_to_csv(self, dir_path: str, score_index: float, 
                                        positive_count: int, negative_count: int):
        """Save overall score to CSV file."""
        output_file = os.path.join(dir_path, "output_score.csv")
        with open(output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["positive_count", "negative_count", "overall_score_index"])
            writer.writerow([positive_count, negative_count, score_index])

    def run_ffmpeg(self, input_file: str, output_file: str, start_time: float, end_time: float):
        """Run ffmpeg command in a cross-platform way."""
        ffmpeg = get_ffmpeg_path()
        command = [
            ffmpeg,
            '-hide_banner', '-loglevel', 'error',
            '-y', '-i', input_file,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-ar', '16000',  # Resample to 16kHz for Wav2Vec2
            '-ac', '1',      # Convert to mono
            output_file
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=os.getcwd(), **get_subprocess_kwargs())

    def get_output_folder(self) -> str:
        """Get the output folder path for current audio file."""
        return os.path.join(self.output_base_dir, self.get_audiofile_name(False))

    def get_audiofile_name(self, ext: bool = True) -> str:
        """Get the audio file name with or without extension."""
        if ext:
            return os.path.basename(self.audio_file)
        else:
            return os.path.splitext(os.path.basename(self.audio_file))[0]

    def load_parameters_from_ui(self, ui):
        """Load parameters from UI elements."""
        self.audio_file = ui.audio_link.text()
        self.trim = ui.trim_audio_duration.value()
        self.process_all = ui.process_all_audio.isChecked()
        self.start_time = ui.audio_start_time.value()
        self.end_time = ui.audio_end_time.value()

    def audio_main(self):
        """Main processing function for audio emotion analysis."""
        if not self.audio_file or self.audio_file.strip() == '':
            return
        
        print(f'Processing audio file: {self.audio_file}')
        
        # Get audio duration
        import librosa
        length_in_seconds = librosa.get_duration(path=self.audio_file)
        
        if self.trim > length_in_seconds:
            self.trim = length_in_seconds
        
        total_segments = max(1, int(length_in_seconds / self.trim))
        
        output_dir = self.get_output_folder()
        os.makedirs(output_dir, exist_ok=True)
        
        if self.process_all:
            for i in range(total_segments):
                self.progress_signal.emit(i + 1, total_segments)
                
                trimmed_dir = os.path.join(output_dir, f"clip_{i}")
                os.makedirs(trimmed_dir, exist_ok=True)
                
                self.start_time = i * self.trim
                if i == total_segments - 1:
                    self.end_time = length_in_seconds
                else:
                    self.end_time = self.start_time + self.trim
                
                print(f"Processing segment {i+1}/{total_segments}: {self.start_time:.1f}s to {self.end_time:.1f}s")
                audio_file_trimmed = os.path.join(trimmed_dir, f"{self.get_audiofile_name(False)}_{i}.wav")
                
                self.run_ffmpeg(self.audio_file, audio_file_trimmed, self.start_time, self.end_time)
                
                predictions, dominant = self.analyze_audio_segment(audio_file_trimmed, trimmed_dir)
                csv_dir = os.path.join(output_dir, "_result")
                self.add_to_csv(i, csv_dir, audio_file_trimmed, dominant, predictions, 
                               self.start_time, self.end_time)
            
            # Calculate overall score
            csv_path = os.path.join(csv_dir, "output.csv")
            score_index, positive_sum, negative_sum = self.calculate_overall_score_index(csv_path)
            self.add_overall_score_index_to_csv(csv_dir, score_index, positive_sum, negative_sum)
            self.completed_signal.emit(True)
        
        else:
            # Process single segment
            if self.end_time > length_in_seconds:
                self.end_time = length_in_seconds
            if self.start_time < 0:
                self.start_time = 0
            if self.start_time > self.end_time:
                self.start_time, self.end_time = self.end_time, self.start_time
            if self.start_time == self.end_time:
                if self.start_time == 0:
                    self.end_time = min(self.trim, length_in_seconds)
                else:
                    self.start_time = max(0, length_in_seconds - self.trim)
            
            print(f"Processing: {self.start_time:.1f}s to {self.end_time:.1f}s")
            clip_dir = os.path.join(output_dir, f"clip_{self.start_time}_{self.end_time}")
            os.makedirs(clip_dir, exist_ok=True)
            
            audio_file_trimmed = os.path.join(
                clip_dir,
                f"{self.get_audiofile_name(False)}_{self.start_time}_{self.end_time}.wav"
            )
            
            self.run_ffmpeg(self.audio_file, audio_file_trimmed, self.start_time, self.end_time)
            
            predictions, dominant = self.analyze_audio_segment(audio_file_trimmed, clip_dir)
            csv_dir = os.path.join(clip_dir, "_result")
            self.add_to_csv(0, csv_dir, audio_file_trimmed, dominant, predictions,
                           self.start_time, self.end_time)
            
            score_index, positive_sum, negative_sum = self.calculate_overall_score_index(
                os.path.join(csv_dir, "output.csv")
            )
            self.add_overall_score_index_to_csv(csv_dir, score_index, positive_sum, negative_sum)
            self.completed_signal.emit(True)


# Convenience function for standalone usage
def analyze_audio_file(audio_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Analyze emotions in an audio file.
    
    Args:
        audio_path: Path to the audio file
        output_dir: Optional output directory for results
        
    Returns:
        Dictionary with emotion analysis results
    """
    recognizer = Wav2VecEmotionRecognizer()
    recognizer.load_models()
    
    dominant, scores, predictions = recognizer.predict_emotion(audio_path)
    
    result = {
        "dominant_emotion": dominant,
        "emotion_scores": scores,
        "confidence": float(predictions.max()) * 100
    }
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        recognizer.analyze_audio_segment(audio_path, output_dir)
        result["output_dir"] = output_dir
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze emotions in audio files")
    parser.add_argument("audio_file", help="Path to audio file")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    
    args = parser.parse_args()
    
    result = analyze_audio_file(args.audio_file, args.output)
    print(f"\nResults:")
    print(f"  Dominant Emotion: {result['dominant_emotion']}")
    print(f"  Confidence: {result['confidence']:.2f}%")
    print(f"  All Scores: {result['emotion_scores']}")

