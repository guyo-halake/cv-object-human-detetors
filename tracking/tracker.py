from collections import defaultdict

class TrajectoryTracker:
    def __init__(self, max_history=30):
        self.history = defaultdict(list)
        self.max_history = max_history

    def update(self, track_id, bbox):
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        
        self.history[track_id].append((cx, cy))
        if len(self.history[track_id]) > self.max_history:
            self.history[track_id].pop(0)
            
    def get_trajectory(self, track_id):
        return self.history.get(track_id, [])
