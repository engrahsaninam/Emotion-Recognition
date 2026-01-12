"""
Advanced Text Analysis Module

This module provides enhanced text analysis capabilities including:
- Sentiment analysis with FinBERT
- Aspect-based sentiment extraction
- Named entity recognition (NER)
- Financial entity extraction (companies, tickers, etc.)
- Batch processing for large documents
"""

import os
import re
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

import pandas as pd
import PyPDF2
from PyQt5.QtCore import QObject, pyqtSignal


@dataclass
class SentimentResult:
    """Container for sentiment analysis results."""
    text: str
    sentiment: str  # Positive, Negative, Neutral
    confidence: float
    aspects: List[Dict[str, Any]] = field(default_factory=list)
    entities: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DocumentAnalysisResult:
    """Container for full document analysis results."""
    paragraphs: List[SentimentResult]
    overall_sentiment: str
    score_index: float
    positive_count: int
    negative_count: int
    neutral_count: int
    entities_summary: Dict[str, List[str]]
    key_aspects: List[Dict[str, Any]]


class AdvancedTextAnalyzer(QObject):
    """
    Advanced text analysis with aspect-based sentiment and NER.
    """
    
    completed_signal = pyqtSignal(bool)
    progress_signal = pyqtSignal(int, int)  # current, total

    def __init__(self):
        super().__init__()
        self.finbert = None
        self.tokenizer = None
        self.sentiment_pipeline = None
        self.ner_pipeline = None
        
        # Analysis parameters
        self.file_path_pdf = None
        self.start_line = 0
        self.end_line = 0
        self.paragraph_limit = 50
        self.process_all_paragraphs = False
        self.file_name = None
        
        # Cross-platform paths
        self.output_base_dir = os.path.join('Output', 'Text')
        
        # Financial keywords for aspect extraction
        self.financial_aspects = [
            'revenue', 'profit', 'earnings', 'growth', 'margin', 'debt',
            'cash flow', 'dividend', 'stock', 'share', 'investment',
            'market', 'sales', 'cost', 'expense', 'income', 'loss',
            'forecast', 'outlook', 'guidance', 'performance', 'risk'
        ]

    def load_models(self):
        """Load all required NLP models."""
        print('Loading Advanced Text Models...')
        
        from transformers import (
            BertTokenizer, 
            BertForSequenceClassification,
            pipeline,
            AutoModelForTokenClassification,
            AutoTokenizer
        )
        
        # Load FinBERT for sentiment
        print('  Loading FinBERT sentiment model...')
        self.finbert = BertForSequenceClassification.from_pretrained(
            'yiyanghkust/finbert-tone', 
            num_labels=3
        )
        self.tokenizer = BertTokenizer.from_pretrained('yiyanghkust/finbert-tone')
        self.sentiment_pipeline = pipeline(
            "sentiment-analysis", 
            model=self.finbert, 
            tokenizer=self.tokenizer
        )
        
        # Load NER model
        print('  Loading NER model...')
        try:
            self.ner_pipeline = pipeline(
                "ner",
                model="dbmdz/bert-large-cased-finetuned-conll03-english",
                aggregation_strategy="simple"
            )
        except Exception as e:
            print(f'  NER model loading failed: {e}. Using basic NER.')
            self.ner_pipeline = None
        
        print('Advanced Text Models loaded successfully!')

    def analyze_sentiment(self, text: str) -> Tuple[str, float]:
        """
        Analyze sentiment of a text chunk.
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (sentiment_label, confidence)
        """
        if not text.strip():
            return "Neutral", 0.0
        
        # Truncate if too long (BERT max 512 tokens)
        words = text.split()
        if len(words) > 400:
            text = ' '.join(words[:400])
        
        try:
            result = self.sentiment_pipeline(text)[0]
            return result["label"], result["score"]
        except Exception as e:
            print(f"Sentiment analysis error: {e}")
            return "Neutral", 0.0

    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract named entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of entity dictionaries
        """
        entities = []
        
        if self.ner_pipeline:
            try:
                # Truncate if too long
                words = text.split()
                if len(words) > 400:
                    text = ' '.join(words[:400])
                
                ner_results = self.ner_pipeline(text)
                
                for entity in ner_results:
                    entities.append({
                        "text": entity["word"],
                        "type": entity["entity_group"],
                        "confidence": entity["score"],
                        "start": entity["start"],
                        "end": entity["end"]
                    })
            except Exception as e:
                print(f"NER error: {e}")
        
        # Add financial entity patterns
        entities.extend(self._extract_financial_entities(text))
        
        return entities

    def _extract_financial_entities(self, text: str) -> List[Dict[str, Any]]:
        """Extract financial entities using regex patterns."""
        entities = []
        
        # Stock ticker pattern (e.g., $AAPL, NASDAQ:GOOG)
        ticker_pattern = r'\$[A-Z]{1,5}|(?:NYSE|NASDAQ|AMEX):\s*[A-Z]{1,5}'
        for match in re.finditer(ticker_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "TICKER",
                "confidence": 1.0,
                "start": match.start(),
                "end": match.end()
            })
        
        # Money pattern (e.g., $10M, $5.2 billion)
        money_pattern = r'\$[\d,.]+\s*(?:million|billion|M|B|K)?|\d+(?:\.\d+)?\s*(?:million|billion)\s*(?:dollars)?'
        for match in re.finditer(money_pattern, text, re.IGNORECASE):
            entities.append({
                "text": match.group(),
                "type": "MONEY",
                "confidence": 0.9,
                "start": match.start(),
                "end": match.end()
            })
        
        # Percentage pattern
        pct_pattern = r'-?\d+(?:\.\d+)?%'
        for match in re.finditer(pct_pattern, text):
            entities.append({
                "text": match.group(),
                "type": "PERCENTAGE",
                "confidence": 1.0,
                "start": match.start(),
                "end": match.end()
            })
        
        return entities

    def extract_aspects(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract aspect-based sentiment from text.
        
        Args:
            text: Input text
            
        Returns:
            List of aspect sentiment dictionaries
        """
        aspects = []
        text_lower = text.lower()
        
        for aspect in self.financial_aspects:
            if aspect in text_lower:
                # Find sentences containing the aspect
                sentences = text.split('.')
                for sentence in sentences:
                    if aspect in sentence.lower():
                        sentiment, confidence = self.analyze_sentiment(sentence)
                        aspects.append({
                            "aspect": aspect,
                            "sentence": sentence.strip(),
                            "sentiment": sentiment,
                            "confidence": confidence
                        })
                        break  # One aspect per keyword
        
        return aspects

    def analyze_paragraph(self, text: str) -> SentimentResult:
        """
        Perform full analysis on a paragraph.
        
        Args:
            text: Paragraph text
            
        Returns:
            SentimentResult with all analysis
        """
        sentiment, confidence = self.analyze_sentiment(text)
        entities = self.extract_entities(text)
        aspects = self.extract_aspects(text)
        
        return SentimentResult(
            text=text,
            sentiment=sentiment,
            confidence=confidence,
            aspects=aspects,
            entities=entities
        )

    def analyze_document(self, paragraphs: List[str]) -> DocumentAnalysisResult:
        """
        Analyze a full document.
        
        Args:
            paragraphs: List of paragraph texts
            
        Returns:
            DocumentAnalysisResult with full analysis
        """
        results = []
        all_entities = {}
        all_aspects = []
        
        for i, para in enumerate(paragraphs):
            if len(para.strip()) > self.paragraph_limit:
                self.progress_signal.emit(i + 1, len(paragraphs))
                result = self.analyze_paragraph(para)
                results.append(result)
                
                # Aggregate entities
                for entity in result.entities:
                    entity_type = entity["type"]
                    if entity_type not in all_entities:
                        all_entities[entity_type] = []
                    if entity["text"] not in all_entities[entity_type]:
                        all_entities[entity_type].append(entity["text"])
                
                # Aggregate aspects
                all_aspects.extend(result.aspects)
        
        # Calculate overall sentiment
        positive_count = sum(1 for r in results if r.sentiment == "Positive")
        negative_count = sum(1 for r in results if r.sentiment == "Negative")
        neutral_count = sum(1 for r in results if r.sentiment == "Neutral")
        
        total = positive_count + negative_count + neutral_count
        if total == 0:
            score_index = 0
            overall_sentiment = "Neutral"
        else:
            score_index = (positive_count - negative_count) / total
            if score_index > 0.1:
                overall_sentiment = "Positive"
            elif score_index < -0.1:
                overall_sentiment = "Negative"
            else:
                overall_sentiment = "Neutral"
        
        return DocumentAnalysisResult(
            paragraphs=results,
            overall_sentiment=overall_sentiment,
            score_index=score_index,
            positive_count=positive_count,
            negative_count=negative_count,
            neutral_count=neutral_count,
            entities_summary=all_entities,
            key_aspects=all_aspects[:20]  # Top 20 aspects
        )

    def pdf_to_paragraphs(self, file_path: str) -> List[str]:
        """
        Extract paragraphs from a PDF file.
        
        Args:
            file_path: Path to PDF file
            
        Returns:
            List of paragraph strings
        """
        paragraphs = []
        
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            full_text = ""
            
            for page in reader.pages:
                full_text += page.extract_text() + "\n"
        
        # Split into paragraphs
        raw_paragraphs = full_text.split('\n\n')
        
        for para in raw_paragraphs:
            cleaned = para.strip()
            if len(cleaned) > self.paragraph_limit:
                paragraphs.append(cleaned)
        
        return paragraphs

    def save_results(self, result: DocumentAnalysisResult, output_folder: str):
        """
        Save analysis results to files.
        
        Args:
            result: DocumentAnalysisResult to save
            output_folder: Output directory
        """
        os.makedirs(output_folder, exist_ok=True)
        
        # Save paragraph-level results
        para_data = []
        for r in result.paragraphs:
            para_data.append({
                "text": r.text[:200] + "..." if len(r.text) > 200 else r.text,
                "sentiment": r.sentiment,
                "confidence": r.confidence,
                "num_entities": len(r.entities),
                "num_aspects": len(r.aspects)
            })
        
        para_df = pd.DataFrame(para_data)
        para_df.to_csv(os.path.join(output_folder, "paragraph_analysis.csv"), index=False)
        
        # Save overall results
        overall_data = {
            "overall_sentiment": [result.overall_sentiment],
            "score_index": [result.score_index],
            "positive_count": [result.positive_count],
            "negative_count": [result.negative_count],
            "neutral_count": [result.neutral_count]
        }
        overall_df = pd.DataFrame(overall_data)
        overall_df.to_csv(os.path.join(output_folder, "overall_score_index.csv"), index=False)
        
        # Save entities
        if result.entities_summary:
            entities_data = []
            for entity_type, entities in result.entities_summary.items():
                for entity in entities:
                    entities_data.append({"type": entity_type, "entity": entity})
            
            if entities_data:
                entities_df = pd.DataFrame(entities_data)
                entities_df.to_csv(os.path.join(output_folder, "entities.csv"), index=False)
        
        # Save aspects
        if result.key_aspects:
            aspects_df = pd.DataFrame(result.key_aspects)
            aspects_df.to_csv(os.path.join(output_folder, "aspects.csv"), index=False)

    def get_output_folder(self) -> str:
        """Get output folder path."""
        return os.path.join(self.output_base_dir, self.get_input_file_name(False))

    def get_input_file_name(self, ext: bool = True) -> str:
        """Get input file name."""
        if ext:
            return self.file_name
        return os.path.splitext(self.file_name)[0]

    def main(self, ui=None, file_path: Optional[str] = None):
        """
        Main analysis function.
        
        Args:
            ui: Optional UI object for parameter loading
            file_path: Optional direct file path
        """
        # Wait for model
        while self.sentiment_pipeline is None:
            pass
        
        print('Executing advanced text analysis...')
        
        # Get parameters
        if ui:
            self.file_path_pdf = ui.text_file_path.text()
            self.start_line = ui.text_start_line.value()
            self.end_line = ui.text_end_line.value()
            self.paragraph_limit = ui.text_paragraph_limit.value()
            self.process_all_paragraphs = ui.text_all_paragraphs.isChecked()
        elif file_path:
            self.file_path_pdf = file_path
            self.process_all_paragraphs = True
        
        if not self.file_path_pdf:
            return
        
        # Set file name
        name = os.path.splitext(os.path.basename(self.file_path_pdf))[0]
        if self.process_all_paragraphs:
            self.file_name = f"{name}-all-paragraphs"
        else:
            self.file_name = f"{name}-{self.start_line}-{self.end_line}"
        
        # Extract paragraphs
        paragraphs = self.pdf_to_paragraphs(self.file_path_pdf)
        
        if not self.process_all_paragraphs:
            paragraphs = paragraphs[self.start_line:self.end_line]
        
        print(f"Analyzing {len(paragraphs)} paragraphs...")
        
        # Analyze
        result = self.analyze_document(paragraphs)
        
        # Save results
        output_folder = self.get_output_folder()
        self.save_results(result, output_folder)
        
        print(f"Analysis complete! Results saved to {output_folder}")
        print(f"  Overall sentiment: {result.overall_sentiment}")
        print(f"  Score index: {result.score_index:.2f}")
        print(f"  Positive: {result.positive_count}, Negative: {result.negative_count}, Neutral: {result.neutral_count}")
        
        self.completed_signal.emit(True)
        return result


def analyze_text_file(file_path: str, output_dir: Optional[str] = None) -> Dict:
    """
    Convenience function for analyzing a text/PDF file.
    
    Args:
        file_path: Path to the file
        output_dir: Optional output directory
        
    Returns:
        Analysis results dictionary
    """
    analyzer = AdvancedTextAnalyzer()
    analyzer.load_models()
    
    if output_dir:
        analyzer.output_base_dir = output_dir
    
    result = analyzer.main(file_path=file_path)
    
    return {
        "overall_sentiment": result.overall_sentiment,
        "score_index": result.score_index,
        "positive_count": result.positive_count,
        "negative_count": result.negative_count,
        "neutral_count": result.neutral_count,
        "entities": result.entities_summary,
        "key_aspects": result.key_aspects
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze text/PDF documents")
    parser.add_argument("file", help="Path to PDF file")
    parser.add_argument("-o", "--output", help="Output directory", default=None)
    
    args = parser.parse_args()
    
    result = analyze_text_file(args.file, args.output)
    
    print("\nAnalysis Results:")
    print(f"  Overall Sentiment: {result['overall_sentiment']}")
    print(f"  Score Index: {result['score_index']:.2f}")
    print(f"  Entities Found: {sum(len(v) for v in result['entities'].values())}")
    print(f"  Aspects Analyzed: {len(result['key_aspects'])}")

