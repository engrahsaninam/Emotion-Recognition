import csv
import os
import time

import pandas as pd
import yt_dlp

from PyQt5.QtCore import QObject, pyqtSignal


# YouTube video URL
# url = "https://www.youtube.com/watch?v=1OPYwTn1n-M&list=PPSV"

class VideoAnalyzer(QObject):
    download_progress = pyqtSignal(int)
    resized_image = pyqtSignal(object)
    emotions_dict = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.video_url = None
        self.delay = None
        self.video_title = None
        self.video_duration = None
        
        # Cross-platform paths
        self.input_dir = os.path.join('Input', 'Video')
        self.output_dir = os.path.join('Output', 'Video')

    def set_video_url(self, url):
        self.video_url = url

    def set_delay(self, delay=30):
        self.delay = delay

    def download_video(self):
        try:
            # Clear existing dir
            import shutil
            if os.path.exists(self.input_dir):
                shutil.rmtree(self.input_dir)
            os.makedirs(self.input_dir, exist_ok=True)

            self.download_progress.emit(2)
            
            ydl_opts = {
                'outtmpl': os.path.join(self.input_dir, '%(title)s.%(ext)s'),
                'progress_hooks': [self.update_progress],
                'continuedl': False,
                'format': 'best[ext=mp4]/best',
                'quiet': True,
                'no_warnings': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.video_url])
            
            time.sleep(2)
            downloaded_list = [f for f in os.listdir(self.input_dir)]
            if downloaded_list:
                self.video_title = os.path.basename(downloaded_list[0])
                print('After download')
                print(self.video_title)
                self.download_progress.emit(1111)
        except Exception as e:
            print('Error in Downloading')
            print(e)
            self.download_progress.emit(-1)

    def update_progress(self, progress_dict):
        if progress_dict['status'] == 'downloading':
            total_bytes = progress_dict.get('total_bytes') or progress_dict.get('total_bytes_estimate', 0)
            if total_bytes > 0:
                progress_percent = progress_dict.get('downloaded_bytes', 0) / total_bytes * 100
                if int(progress_percent) < 100:
                    self.download_progress.emit(int(progress_percent))
                print(f"Downloaded {progress_percent:.2f}%")
        elif progress_dict['status'] == 'finished':
            self.download_progress.emit(100)

    def get_video_duration(self):
        return self.video_duration

    def calculate_video_duration(self):
        print('Calculating Video Duration')
        from moviepy.video.io.VideoFileClip import VideoFileClip
        clip = VideoFileClip(self.get_input_path())
        self.video_duration = clip.duration
        clip.close()

    def get_input_path(self, ext=True):
        return os.path.join(self.input_dir, self.get_video_title(ext))

    def get_output_path(self, ext=True):
        if ext:
            return os.path.join(self.output_dir, f'{self.get_video_title(False)}_without_audio.mp4')
        else:
            return os.path.join(self.output_dir, f'{self.get_video_title(False)}_without_audio')

    def get_video_title(self, ext=True):
        if ext:
            video_title = self.video_title
        else:
            video_title = os.path.splitext(self.video_title)[0]
        print(video_title)
        return video_title

    def clean_output_path(self):
        output_path = self.get_output_path()
        if os.path.exists(output_path):
            os.remove(output_path)

    def cut_video(self, start_time, end_time):
        target_path = os.path.join(
            os.path.dirname(self.get_input_path()),
            f'{self.get_video_title(False)}_{start_time}_{end_time}.mp4'
        )
        from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
        ffmpeg_extract_subclip(self.get_input_path(), start_time, end_time, targetname=target_path)
        self.video_title = os.path.basename(target_path)
        self.video_duration = end_time - start_time

    def detect_faces(self, emotion):
        output_emotions = []
        frameCount = 0
        dominant_emotion = "neutral"
        import cv2
        
        # Create a VideoCapture object and set the start position
        cap = cv2.VideoCapture(self.get_input_path())
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frameSize = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        second_wait = int(self.delay)
        frames_wait = int(fps * second_wait)

        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        output_path = self.get_output_path()
        out = cv2.VideoWriter(output_path, fourcc, fps, frameSize)

        while True:
            start = time.time()
            try:
                success, image = cap.read()
            except Exception as e:
                print(e)

            if not success:
                print('failed frame')
                break

            frameCount += 1

            outputE, region, confidence, emotionsDict = emotion(image)
            tempDict = {}

            if outputE != "Not Detected":
                output_emotions.append(outputE)
                try:
                    emotionsDict.pop(outputE)
                    tempDict.update({outputE: 0})
                except Exception as error:
                    pass
                try:
                    emotionsDict.pop(dominant_emotion)
                    tempDict.update({dominant_emotion: confidence})
                except:
                    pass

                if len(output_emotions) > frames_wait:
                    dominant_emotion = max(set(output_emotions[-20:]), key=output_emotions[-20:].count)
                    output_emotions = []
                
                for emotionname, confidencee in emotionsDict.items():
                    tempDict.update({emotionname: confidencee})
                self.emotions_dict.emit(tempDict)
                self.add_to_csv(dominant_emotion,
                                [tempDict['sad'], tempDict['angry'], tempDict['surprise'], tempDict['fear'],
                                 tempDict['happy'], tempDict['neutral']])
            
            if region != 0:
                x, y, w, h = region.values()
                image = cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            end = time.time()
            totalTime = end - start
            try:
                current_fps = 1 / totalTime
            except:
                current_fps = fps

            cv2.putText(image, f'FPS: {int(current_fps)}', (200, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            out.write(cv2.resize(image, frameSize))

            # Resize the image using the new height and width
            signal_image = cv2.resize(image, (700, 400))
            self.resized_image.emit(signal_image)

        cap.release()
        out.release()

        score_index, positive_sum, negative_sum = self.calculate_overall_score_index()
        self.add_overall_score_index_to_csv(score_index, positive_sum, negative_sum)

    def add_overall_score_index_to_csv(self, overall_score_index, positive_count, negative_count):
        output_file = f"{self.get_output_path(ext=False)}-overall_score.csv"
        with open(output_file, 'w', newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["positive_count", "negative_count", "overall_score_index"])
            writer.writerow([positive_count, negative_count, overall_score_index])

    def calculate_overall_score_index(self):
        csv_path = self.get_output_path(False) + '.csv'
        df = pd.read_csv(csv_path)
        positive_sum = df["happy"].sum()
        negative_sum = df["sad"].sum()
        negative_sum += df["angry"].sum()
        negative_sum += df["fear"].sum()
        if positive_sum + negative_sum == 0:
            print(0)
            return 0, positive_sum, negative_sum
        score_index = (positive_sum - negative_sum) / (positive_sum + negative_sum)
        return score_index, positive_sum, negative_sum

    def add_to_csv(self, dominant_emotion, percentages):
        headers = ['clip_name', "dominant", 'sad', 'angry', 'surprise', 'fear', 'happy', 'neutral']
        csv_path = self.get_output_path(False) + '.csv'
        if not os.path.exists(csv_path):
            with open(csv_path, 'w', newline="") as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        with open(csv_path, 'a', newline="") as f:
            writer = csv.writer(f)
            output_array = [self.get_video_title(False), dominant_emotion, percentages[0], percentages[1],
                            percentages[2], percentages[3], percentages[4], percentages[5]]
            writer.writerow(output_array)
