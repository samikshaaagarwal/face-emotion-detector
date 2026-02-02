import numpy as np
from collections import Counter

def smooth_predictions(history, window=7):
    """
    Smooth predictions using a moving window majority vote.
    """
    if len(history) < window:
        windowed = history
    else:
        windowed = history[-window:]
    most_common = Counter(windowed).most_common(1)[0][0]
    return most_common
