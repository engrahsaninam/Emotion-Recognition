import csv
import os
import subprocess
import sys
import time
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal
from librosa.display import specshow
from librosa import stft, load, amplitude_to_db, feature, get_duration
from Backend.Audio.melspec import plot_colored_polar


def get_subprocess_kwargs():
    """Get platform-specific subprocess kwargs to hide console window on Windows."""
    kwargs = {}
    if sys.platform == 'win32':
        # Only use STARTUPINFO on Windows
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0  # SW_HIDE = 0
        kwargs['startupinfo'] = startupinfo
    return kwargs


def get_ffmpeg_path():
    """Get the ffmpeg executable path based on platform."""
    if sys.platform == 'win32':
        # Check for bundled ffmpeg on Windows
        bundled_path = os.path.join('bin', 'ffmpeg.exe')
        if os.path.exists(bundled_path):
            return os.path.abspath(bundled_path)
    # Fall back to system ffmpeg
    return 'ffmpeg'


class AudioAnalyzer(QObject):
    completed_signal = pyqtSignal(bool)
    path_signals = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.model = None
        self.audio_file = None
        self.trim = None
        self.process_all = False
        self.start_time = 0
        self.end_time = 0
        
        # Cross-platform paths
        self.models_dir = os.path.join('Backend', 'Audio', 'models')
        self.output_base_dir = os.path.join('Output', 'Audio')
        
        self.CAT6 = ['fear', 'angry', 'neutral', 'happy', 'sad', 'surprise']
        self.COLOR_DICT = {
            "neutral": "grey",
            "positive": "green",
            "happy": "green",
            "surprise": "orange",
            "fear": "purple",
            "negative": "red",
            "angry": "red",
            "sad": "lightblue"
        }

    def load_models(self):
        from tensorflow.python.keras.models import load_model
        model_path = os.path.join(self.models_dir, 'model3.h5')
        self.model = load_model(model_path)
        self.starttime = datetime.now()

    def log_file(self, txt=None):
        with open("log.txt", "a") as f:
            datetoday = datetime.now().strftime("%d/%m/%Y %H:%M:%S")
            f.write(f"{txt} - {datetoday};\n")

    def add_overall_score_index_to_csv(self, dir_path, overall_score_index, positive_count, negative_count):
        output_file = os.path.join(dir_path, "output_score.csv")
        with open(output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["positive_count", "negative_count", "overall_score_index"])
            writer.writerow([positive_count, negative_count, overall_score_index])

    def calculate_overall_score_index(self, csv_file_path):
        df = pd.read_csv(csv_file_path, usecols=["dominant"])
        positive_sum = df[df["dominant"] == "happy"].shape[0]
        negative_sum = df[df["dominant"].isin(["sad", "angry", "fear"])].shape[0]
        if positive_sum + negative_sum == 0:
            return 0, 0, 0
        score_index = (positive_sum - negative_sum) / (positive_sum + negative_sum)
        return score_index, positive_sum, negative_sum

    def add_to_csv(self, loop_count, dir_path, clip_name, dominant, confidence, start_time, end_time):
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
        headers = ['clip_name', "dominant", 'fear', 'angry', 'neutral', 'happy', 'sad', 'surprise', 'start_time', 'end_time']
        output_file = os.path.join(dir_path, "output.csv")

        if not os.path.exists(output_file) or loop_count == 0:
            with open(output_file, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)

        with open(output_file, 'a', newline="") as f:
            writer = csv.writer(f)
            confidence_values = [each_conf.max() * 100 for each_conf in confidence]
            output_array = [
                clip_name, dominant,
                confidence_values[0], confidence_values[1], confidence_values[2],
                confidence_values[3], confidence_values[4], confidence_values[5],
                start_time, end_time
            ]
            writer.writerow(output_array)

    def get_melspec(self, audio, path):
        y, sr = load(audio, sr=44100)
        X = stft(y)
        Xdb = amplitude_to_db(abs(X))
        img = np.stack((Xdb,) * 3, -1)
        img = img.astype(np.uint8)
        import cv2
        grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        grayImage = cv2.resize(grayImage, (224, 224))
        rgbImage = np.repeat(grayImage[..., np.newaxis], 3, -1)
        cv2.imwrite(os.path.join(path, "Mel-log-spectrogram.png"), rgbImage)
        return rgbImage, Xdb

    def get_mfccs(self, audio, limit):
        y, sr = load(audio)
        a = feature.mfcc(y=y, sr=sr, n_mfcc=40)
        if a.shape[1] > limit:
            mfccs = a[:, :limit]
        elif a.shape[1] < limit:
            mfccs = np.zeros((a.shape[0], limit))
            mfccs[:, :a.shape[1]] = a
        else:
            mfccs = a
        return mfccs

    def get_title(self, predictions, categories=None):
        if categories is None:
            categories = self.CAT6
        title = f"Detected emotion: {categories[predictions.argmax()]} - {predictions.max() * 100:.2f}%"
        return title

    def color_dict(self):
        return self.COLOR_DICT

    def main(self, audio_file, output_dir):
        try:
            wav, sr = load(audio_file, sr=44100)
            Xdb = self.get_melspec(audio_file, output_dir)[1]
            mfccs = feature.mfcc(y=wav, sr=sr)

            fig = plt.figure(figsize=(10, 2))
            fig.set_facecolor('#d1d1e0')
            plt.title("MFCCs")
            specshow(mfccs, sr=sr, x_axis='time')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.savefig(os.path.join(output_dir, "MFCCs.png"))
            plt.close()

            fig2 = plt.figure(figsize=(10, 2))
            fig2.set_facecolor('#d1d1e0')
            plt.title("Mel-log-spectrogram")
            specshow(Xdb, sr=sr, x_axis='time', y_axis='hz')
            plt.gca().axes.get_yaxis().set_visible(False)
            plt.gca().axes.spines["right"].set_visible(False)
            plt.gca().axes.spines["left"].set_visible(False)
            plt.gca().axes.spines["top"].set_visible(False)
            plt.savefig(os.path.join(output_dir, "Mel-log-spectrogram.png"))
            plt.close()

            mfccs = self.get_mfccs(audio_file, self.model.input_shape[-1])
            mfccs = mfccs.reshape(1, *mfccs.shape)
            pred = self.model.predict(mfccs)[0]
            txt = "MFCCs\n" + self.get_title(pred, self.CAT6)

            fig3 = plt.figure(figsize=(5, 5))
            COLORS = self.color_dict()

            melspec_path = plot_colored_polar(fig3, predictions=pred, categories=self.CAT6, title=txt, colors=COLORS, path=output_dir)
            dominant_arg = self.CAT6[pred.argmax()]
            plt.close()
            time.sleep(2)
            self.path_signals.emit([
                os.path.join(output_dir, "MFCCs.png"),
                os.path.join(output_dir, "Mel-log-spectrogram.png"),
                melspec_path
            ])

            return pred, dominant_arg
        except Exception as e:
            print(e)
            return None, None

    def load_paramters_from_ui(self, ui):
        self.audio_file = ui.audio_link.text()
        self.trim = ui.trim_audio_duration.value()
        self.process_all = ui.process_all_audio.isChecked()
        self.start_time = ui.audio_start_time.value()
        self.end_time = ui.audio_end_time.value()

    def get_output_folder(self):
        return os.path.join(self.output_base_dir, self.get_audiofile_name(False))

    def get_audiofile_name(self, ext=True):
        if ext:
            return os.path.basename(self.audio_file)
        else:
            return os.path.splitext(os.path.basename(self.audio_file))[0]

    def run_ffmpeg(self, input_file, output_file, start_time, end_time):
        """Run ffmpeg command in a cross-platform way."""
        ffmpeg = get_ffmpeg_path()
        command = [
            ffmpeg,
            '-hide_banner', '-loglevel', 'error',
            '-y', '-i', input_file,
            '-ss', str(start_time),
            '-to', str(end_time),
            '-c', 'copy',
            output_file
        ]
        print(f"Running: {' '.join(command)}")
        subprocess.run(command, cwd=os.getcwd(), **get_subprocess_kwargs())

    def audio_main(self):
        if not self.audio_file or self.audio_file.strip() == '':
            return

        print(f'path of audio file is {self.audio_file}')
        length_in_seconds = get_duration(path=self.audio_file)

        if self.trim > length_in_seconds:
            self.trim = length_in_seconds

        total_segments = int(length_in_seconds / self.trim)

        output_dir = self.get_output_folder()
        os.makedirs(output_dir, exist_ok=True)

        if self.process_all:
            for i in range(total_segments):
                trimmed_dir = os.path.join(output_dir, f"clip_{i}")
                os.makedirs(trimmed_dir, exist_ok=True)
                
                self.start_time = i * self.trim
                if i == total_segments - 1:
                    self.end_time = length_in_seconds
                else:
                    self.end_time = self.start_time + self.trim
                
                print(f"trimming {self.audio_file} from {self.start_time} to {self.end_time}")
                audio_file_trimmed = os.path.join(trimmed_dir, f"{self.get_audiofile_name(False)}_{i}.wav")

                self.run_ffmpeg(self.audio_file, audio_file_trimmed, self.start_time, self.end_time)

                pred, dominant_arg = self.main(audio_file_trimmed, trimmed_dir)
                csv_dir = os.path.join(output_dir, "_result")
                self.add_to_csv(i, csv_dir, audio_file_trimmed, dominant_arg, pred, self.start_time, self.end_time)

            score_index, positive_sum, negative_sum = self.calculate_overall_score_index(
                os.path.join(csv_dir, "output.csv")
            )
            self.add_overall_score_index_to_csv(csv_dir, score_index, positive_sum, negative_sum)
            self.completed_signal.emit(True)

        else:
            if self.end_time > length_in_seconds:
                self.end_time = length_in_seconds
            if self.start_time < 0:
                self.start_time = 0

            if self.start_time > self.end_time:
                self.start_time, self.end_time = self.end_time, self.start_time
            if self.start_time == self.end_time:
                if self.start_time == 0:
                    self.end_time = self.trim
                if self.end_time == length_in_seconds:
                    self.start_time = length_in_seconds - self.trim

            print(f"trimming {self.audio_file} from {self.start_time} to {self.end_time}")
            clip_dir = os.path.join(output_dir, f"clip_{self.start_time}_{self.end_time}")
            os.makedirs(clip_dir, exist_ok=True)
            
            audio_file_trimmed = os.path.join(
                clip_dir,
                f"{self.get_audiofile_name(False)}_{self.start_time}_{self.end_time}.wav"
            )

            self.run_ffmpeg(self.audio_file, audio_file_trimmed, self.start_time, self.end_time)

            pred, dominant_arg = self.main(audio_file_trimmed, clip_dir)
            csv_dir = os.path.join(clip_dir, "_result")
            self.add_to_csv(0, csv_dir, audio_file_trimmed, dominant_arg, pred, self.start_time, self.end_time)
            score_index, positive_sum, negative_sum = self.calculate_overall_score_index(
                os.path.join(csv_dir, "output.csv")
            )
            self.add_overall_score_index_to_csv(csv_dir, score_index, positive_sum, negative_sum)
