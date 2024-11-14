# Ultralytics YOLO 🚀, AGPL-3.0 license
"""Module defines the base classes and structures for object tracking in YOLO."""

from collections import OrderedDict

import numpy as np


class TrackState:
    """
    Enumeration class representing the possible states of an object being tracked.
    跟踪状态类
    Attributes:
        New (int): State when the object is newly detected.
        Tracked (int): State when the object is successfully tracked in subsequent frames.
        Lost (int): State when the object is no longer tracked.
        Removed (int): State when the object is removed from tracking.

    Examples:
        >>> state = TrackState.New
        >>> if state == TrackState.New:
        >>>     print("Object is newly detected.")
    """

    New = 0
    Tracked = 1
    Lost = 2
    Removed = 3


class BaseTrack:
    """
    Base class for object tracking, providing foundational attributes and methods.
    基础目标跟踪类,提供基础属性和方法
    Attributes:
        _count (int): Class-level counter for unique track IDs.
        track_id (int): Unique identifier for the track.
        is_activated (bool): Flag indicating whether the track is currently active.
        state (TrackState): Current state of the track.
        history (OrderedDict): Ordered history of the track's states.
        features (List): List of features extracted from the object for tracking.
        curr_feature (Any): The current feature of the object being tracked.
        score (float): The confidence score of the tracking.
        start_frame (int): The frame number where tracking started.
        frame_id (int): The most recent frame ID processed by the track.
        time_since_update (int): Frames passed since the last update.
        location (Tuple): The location of the object in the context of multi-camera tracking.

    Methods:
        end_frame: Returns the ID of the last frame where the object was tracked.
        next_id: Increments and returns the next global track ID.
        activate: Abstract method to activate the track.
        predict: Abstract method to predict the next state of the track.
        update: Abstract method to update the track with new data.
        mark_lost: Marks the track as lost.
        mark_removed: Marks the track as removed.
        reset_id: Resets the global track ID counter.

    Examples:
        Initialize a new track and mark it as lost:
        >>> track = BaseTrack()
        >>> track.mark_lost()
        >>> print(track.state)  # Output: 2 (TrackState.Lost)
    """

    _count = 0

    def __init__(self):
        """
        Initializes a new track with a unique ID and foundational tracking attributes.

        Examples:
            Initialize a new track
            >>> track = BaseTrack()
            >>> print(track.track_id)
            0
        """
        self.track_id = 0#跟踪id号
        self.is_activated = False#是否激活
        self.state = TrackState.New#新的跟踪状态
        self.history = OrderedDict()
        self.features = []#保留的特征图
        self.curr_feature = None#当前特征
        self.score = 0#分数
        self.start_frame = 0#起始帧
        self.frame_id = 0#帧id号
        self.time_since_update = 0#更新时间
        self.location = (np.inf, np.inf)#位置

    @property
    def end_frame(self):#返回跟踪的最后帧id
        """Returns the ID of the most recent frame where the object was tracked."""
        return self.frame_id

    @staticmethod
    def next_id():#返回全局id号计数
        """Increment and return the next unique global track ID for object tracking."""
        BaseTrack._count += 1
        return BaseTrack._count

    def activate(self, *args):
        """Activates the track with provided arguments, initializing necessary attributes for tracking."""
        raise NotImplementedError

    def predict(self):
        """Predicts the next state of the track based on the current state and tracking model."""
        raise NotImplementedError

    def update(self, *args, **kwargs):
        """Updates the track with new observations and data, modifying its state and attributes accordingly."""
        raise NotImplementedError

    def mark_lost(self):#标记为丢失
        """Marks the track as lost by updating its state to TrackState.Lost."""
        self.state = TrackState.Lost

    def mark_removed(self):#标记为移除
        """Marks the track as removed by setting its state to TrackState.Removed."""
        self.state = TrackState.Removed

    @staticmethod
    def reset_id():
        """Reset the global track ID counter to its initial value."""#全局id归零
        BaseTrack._count = 0
