# algorithms.py - Backward Compatibility Wrapper
# This file now redirects to modularized components.

from config_manager import config
from signal_processing import OneEuroFilter, PositionEstimator
from scene_detection import (
    QuickSceneDetector, 
    SceneBoundaryHandler, 
    SceneSegmenter, 
    SceneTypeDetector, 
    SceneAnchorSelector
)
from action_generation import ActionPointGenerator, ScriptPostProcessor
from tracking import (
    OpticalFlowEstimator,
    ROIDetector,
    DynamicROITracker,
    YoloPoseTracker,
    ContactPointTracker,
    DualAnchorTracker,
    MotionExtractor,
    DEVICE,
    RESIZE_WIDTH,
    YOLO_AVAILABLE,
    AffineResult
)

# Constants used by main.py
# (They are already imported above from tracking.py)

class VideoAnalyzer:
    """
    Legacy VideoAnalyzer if needed, but usually classes are used directly.
    """
    pass
