import os
import PyPDF2
import pandas as pd
from PyQt5.QtCore import QObject, pyqtSignal


class TextAnalyzer(QObject):
    completed_signal = pyqtSignal(bool)

    def __init__(self):
        super(TextAnalyzer, self).__init__()
        self.finbert = None
        self.tokenizer = None
        self.nlp = None
        self.file_path_pdf = None
        self.start_line = 0
        self.end_line = 0
        self.paragraph_limit = 0
        self.process_all_paragraphs = False
        self.file_name = None
        
        # Cross-platform paths
        self.output_base_dir = os.path.join('Output', 'Text')

    def load_models(self):
        print('Loading Text Models')
        from transformers import BertTokenizer, BertForSequenceClassification
        from transformers import pipeline
        self.finbert = BertForSequenceClassification.from_pretrained('yiyanghkust/finbert-tone', num_labels=3)
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.nlp = pipeline("sentiment-analysis", model=self.finbert, tokenizer=self.tokenizer)
        print('Loaded Text Models')

    def calculate_overall_score_index(self, csv_path):
        df = pd.read_csv(csv_path)
        positive_sum = df[df["dominant_emotion"] == "Positive"].shape[0]
        negative_sum = df[df["dominant_emotion"] == "Negative"].shape[0]
        if positive_sum + negative_sum == 0:
            return 0, positive_sum, negative_sum
        score_index = (positive_sum - negative_sum) / (positive_sum + negative_sum)
        return score_index, positive_sum, negative_sum

    def save_overall_score_index_to_csv(self, folder_path, score_index, positive_sum, negative_sum):
        df = pd.DataFrame({
            "score_index": [score_index],
            "positive_sum": [positive_sum],
            "negative_sum": [negative_sum]
        })
        csv_path = os.path.join(folder_path, "overall_score_index.csv")
        df.to_csv(csv_path, index=False)

    def pdf_to_txt(self, file_path, output_path, name):
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            output_file = os.path.join(output_path, f"{name}.txt")
            with open(output_file, "a", newline="", encoding='utf-8') as txt_file:
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    txt_file.write(page.extract_text())

    def txt_to_paragraphs(self, file_path, paragraph_folder):
        output_emotion = {
            "dominant_emotion": [],
            "probability": []
        }

        start_line = self.start_line
        end_line = self.end_line

        if start_line < 0:
            start_line = 0
        if end_line < 0:
            end_line = 0
        if self.paragraph_limit < 0:
            self.paragraph_limit = 0
        if start_line > end_line:
            start_line, end_line = end_line, start_line

        with open(f"{file_path}.txt", "r", encoding='utf-8') as f:
            lines = f.readlines()
            if end_line > len(lines):
                end_line = len(lines)
            if self.process_all_paragraphs:
                start_line = 0
                end_line = len(lines)
            lines = lines[start_line:end_line]
            text = "".join(lines)

            paragraphs = text.split('.\n')
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) > int(self.paragraph_limit):
                    paragraph_file = os.path.join(paragraph_folder, f"{i}.txt")
                    with open(paragraph_file, "w", newline="", encoding='utf-8') as pf:
                        pf.write(paragraph)
                    
                    words = paragraph.split()
                    chunked_words = [words[j:j + 250] for j in range(0, len(words), 250)]
                    for chunk in chunked_words:
                        chunk_text = " ".join(chunk)
                        results = self.nlp(chunk_text)
                        output_emotion["dominant_emotion"].append(results[0]["label"])
                        output_emotion["probability"].append(results[0]["score"])

        return output_emotion

    def get_output_folder(self):
        return os.path.join(self.output_base_dir, self.get_input_file_name(False))

    def get_input_file_name(self, ext=True):
        if ext:
            return self.file_name
        else:
            return os.path.splitext(self.file_name)[0]

    def main(self, ui):
        # Wait for model to load
        while self.nlp is None:
            pass

        print('executing text main')
        self.file_path_pdf = ui.text_file_path.text()
        if self.file_path_pdf in [None, '']:
            return
        
        self.start_line = ui.text_start_line.value()
        self.end_line = ui.text_end_line.value()
        self.paragraph_limit = ui.text_paragraph_limit.value()
        self.process_all_paragraphs = ui.text_all_paragraphs.isChecked()

        name = os.path.splitext(os.path.basename(self.file_path_pdf))[0]
        name = str(name)
        
        if self.process_all_paragraphs:
            self.file_name = f"{name}-all-paragraphs"
        else:
            self.file_name = f"{name}-{self.start_line}-{self.end_line}"

        file_folder = self.get_output_folder()
        os.makedirs(file_folder, exist_ok=True)
        
        file_path = os.path.join(file_folder, self.get_input_file_name(False))
        txt_file_path = f"{file_path}.txt"
        
        if os.path.exists(txt_file_path):
            os.remove(txt_file_path)
        
        # Load your PDF
        self.pdf_to_txt(self.file_path_pdf, self.get_output_folder(), self.get_input_file_name(False))

        paragraph_folder = os.path.join(file_folder, "paragraph")
        os.makedirs(paragraph_folder, exist_ok=True)
        
        output_emotion_dict = self.txt_to_paragraphs(file_path, paragraph_folder)
        print(output_emotion_dict)
        
        df = pd.DataFrame(output_emotion_dict)
        df.to_csv(f"{file_path}_output.csv", index=False)
        
        score_index, positive_sum, negative_sum = self.calculate_overall_score_index(f"{file_path}_output.csv")
        self.save_overall_score_index_to_csv(file_folder, score_index, positive_sum, negative_sum)
        self.completed_signal.emit(True)
