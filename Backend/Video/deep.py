"""
Modern Face Emotion Detection Module

This module provides facial emotion detection using DeepFace with
optional MediaPipe face mesh for enhanced landmark detection.

Features:
- Multiple detector backends (opencv, mtcnn, retinaface, mediapipe)
- Multi-face detection and tracking
- Optional liveness detection
- Enhanced accuracy with latest DeepFace models
"""

import os
from typing import Dict, List, Optional, Tuple, Any
import numpy as np


# Emotion color mapping
EMOTION_COLORS = {
    "neutral": "grey",
    "happy": "green",
    "surprise": "orange",
    "fear": "purple",
    "angry": "red",
    "sad": "lightblue",
    "disgust": "brown"
}


class FaceEmotionDetector:
    """
    Enhanced face emotion detection with multiple backend support.
    """
    
    def __init__(self, detector_backend: str = "opencv", use_mediapipe_mesh: bool = False):
        """
        Initialize the face emotion detector.
        
        Args:
            detector_backend: Face detection backend 
                Options: 'opencv', 'mtcnn', 'retinaface', 'mediapipe', 'ssd', 'dlib'
            use_mediapipe_mesh: Whether to use MediaPipe Face Mesh for landmarks
        """
        self.detector_backend = detector_backend
        self.use_mediapipe_mesh = use_mediapipe_mesh
        self.face_mesh = None
        self.deepface = None
        
    def initialize(self):
        """Initialize models (lazy loading)."""
        if self.use_mediapipe_mesh:
            try:
                import mediapipe as mp
                self.mp_face_mesh = mp.solutions.face_mesh
                self.face_mesh = self.mp_face_mesh.FaceMesh(
                    max_num_faces=5,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                self.mp_drawing = mp.solutions.drawing_utils
                self.mp_drawing_styles = mp.solutions.drawing_styles
                print("MediaPipe Face Mesh initialized")
            except ImportError:
                print("MediaPipe not available, falling back to DeepFace only")
                self.use_mediapipe_mesh = False
    
    def detect_emotion(self, frame: np.ndarray) -> Tuple[str, Dict, float, Dict[str, float]]:
        """
        Detect emotion from a video frame.
        
        Args:
            frame: BGR image frame (numpy array)
            
        Returns:
            Tuple of (dominant_emotion, region, confidence, all_emotions)
        """
        try:
            from deepface import DeepFace
            
            # Analyze with DeepFace
            result = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Handle list result (multiple faces)
            if isinstance(result, list):
                result = result[0]
            
            dominant_emotion = result["dominant_emotion"]
            confidence = float(result["emotion"][dominant_emotion])
            # Convert numpy floats to Python floats for JSON serialization
            emotions = {k: float(v) for k, v in result["emotion"].items()}
            region = result.get("region", {})
            
            return dominant_emotion, region, confidence, emotions
            
        except Exception as e:
            print(f"Emotion detection error: {e}")
            return "Not Detected", {}, 0, {}
    
    def detect_emotions_multi_face(self, frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect emotions for all faces in a frame.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of detection results for each face
        """
        try:
            from deepface import DeepFace
            
            results = DeepFace.analyze(
                img_path=frame,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend=self.detector_backend,
                silent=True
            )
            
            # Ensure list format
            if not isinstance(results, list):
                results = [results]
            
            detections = []
            for result in results:
                detections.append({
                    "dominant_emotion": result["dominant_emotion"],
                    "confidence": float(result["emotion"][result["dominant_emotion"]]),
                    "emotions": {k: float(v) for k, v in result["emotion"].items()},
                    "region": result.get("region", {})
                })
            
            return detections
            
        except Exception as e:
            print(f"Multi-face detection error: {e}")
            return []
    
    def get_face_landmarks(self, frame: np.ndarray) -> List[Dict]:
        """
        Get face landmarks using MediaPipe Face Mesh.
        
        Args:
            frame: BGR image frame
            
        Returns:
            List of landmark dictionaries for each detected face
        """
        if not self.use_mediapipe_mesh or self.face_mesh is None:
            return []
        
        import cv2
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        landmarks_list = []
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = {
                    "landmarks": [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark],
                    "num_landmarks": len(face_landmarks.landmark)
                }
                landmarks_list.append(landmarks)
        
        return landmarks_list
    
    def draw_face_mesh(self, frame: np.ndarray, draw_contours: bool = True) -> np.ndarray:
        """
        Draw face mesh landmarks on frame.
        
        Args:
            frame: BGR image frame
            draw_contours: Whether to draw face contours
            
        Returns:
            Frame with landmarks drawn
        """
        if not self.use_mediapipe_mesh or self.face_mesh is None:
            return frame
        
        import cv2
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                if draw_contours:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
                    )
                else:
                    self.mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_tesselation_style()
                    )
        
        return frame
    
    def check_liveness(self, frame: np.ndarray) -> Tuple[bool, float]:
        """
        Basic liveness detection to prevent photo spoofing.
        
        This is a simple implementation based on face mesh depth variation.
        For production use, consider more sophisticated methods.
        
        Args:
            frame: BGR image frame
            
        Returns:
            Tuple of (is_live, confidence)
        """
        if not self.use_mediapipe_mesh:
            return True, 0.5  # Cannot determine without face mesh
        
        landmarks_list = self.get_face_landmarks(frame)
        
        if not landmarks_list:
            return False, 0.0
        
        # Check depth variation in z-coordinates
        landmarks = landmarks_list[0]["landmarks"]
        z_values = [lm[2] for lm in landmarks]
        z_variation = np.std(z_values)
        
        # Threshold for 3D face (real faces have more z-variation)
        is_live = z_variation > 0.01
        confidence = min(z_variation * 50, 1.0)
        
        return is_live, confidence
    
    def close(self):
        """Release resources."""
        if self.face_mesh:
            self.face_mesh.close()


# Global detector instance for backward compatibility
_detector = None


def get_detector(detector_backend: str = "opencv", use_mediapipe: bool = False) -> FaceEmotionDetector:
    """Get or create the global detector instance."""
    global _detector
    if _detector is None:
        _detector = FaceEmotionDetector(detector_backend, use_mediapipe)
        _detector.initialize()
    return _detector


def emotion(frame: np.ndarray, detector_backend: str = "opencv") -> Tuple[str, Dict, float, Dict[str, float]]:
    """
    Legacy function for backward compatibility.
    
    Detect emotion from a video frame using DeepFace.
    
    Args:
        frame: BGR image frame (numpy array)
        detector_backend: Face detection backend
        
    Returns:
        Tuple of (output_emotion, region, confidence, emotions_dict)
    """
    try:
        from deepface import DeepFace
        
        result = DeepFace.analyze(
            img_path=frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend=detector_backend,
            silent=True
        )[0]
        
        output_emotion = result["dominant_emotion"]
        confidence = float(result["emotion"][output_emotion])
        # Convert numpy floats to Python floats for JSON serialization
        emotions = {k: float(v) for k, v in result["emotion"].items()}
        region = result.get("region", 0)
        
        return output_emotion, region, confidence, emotions
        
    except Exception as e:
        print(f"Emotion detection error: {e}")
        return "Not Detected", 0, 0, {}


def emotion_multi_face(frame: np.ndarray, detector_backend: str = "opencv") -> List[Dict]:
    """
    Detect emotions for multiple faces in a frame.
    
    Args:
        frame: BGR image frame
        detector_backend: Face detection backend
        
    Returns:
        List of emotion detection results for each face
    """
    detector = get_detector(detector_backend)
    return detector.detect_emotions_multi_face(frame)


if __name__ == "__main__":
    import cv2
    
    # Test with webcam
    cap = cv2.VideoCapture(0)
    detector = FaceEmotionDetector(detector_backend="opencv", use_mediapipe_mesh=True)
    detector.initialize()
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect emotion
        dominant, region, confidence, emotions = detector.detect_emotion(frame)
        
        # Draw results
        if region:
            x, y, w, h = region.get('x', 0), region.get('y', 0), region.get('w', 0), region.get('h', 0)
            if w > 0 and h > 0:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, f"{dominant}: {confidence:.1f}%", (x, y - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw face mesh
        frame = detector.draw_face_mesh(frame)
        
        cv2.imshow('Emotion Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
