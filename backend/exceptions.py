"""Custom exceptions for the wellness analyzer"""


class WellnessAnalysisError(Exception):
    """Base exception for wellness analysis errors"""
    pass


class FaceDetectionError(WellnessAnalysisError):
    """Raised when face detection fails"""
    pass


class ImageProcessingError(WellnessAnalysisError):
    """Raised when image processing fails"""
    pass


class CalibrationError(WellnessAnalysisError):
    """Raised when calibration fails"""
    pass


class AnalysisError(WellnessAnalysisError):
    """Raised when analysis fails"""
    pass


class InvalidImageError(ImageProcessingError):
    """Raised when image is invalid or cannot be decoded"""
    pass


class InsufficientDataError(AnalysisError):
    """Raised when there's insufficient data for analysis"""
    pass

