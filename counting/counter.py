class Counter:
    def __init__(self, class_names):
        self.class_names = class_names
        self.counted_ids = set()
        
        self.global_counts = {name: 0 for name in class_names.values()}
        self.in_counts = {name: 0 for name in class_names.values()}
        self.out_counts = {name: 0 for name in class_names.values()}
        
        self.line = None
        self.previous_sides = {}

    def set_line(self, p1, p2):
        self.line = (p1, p2)

    def _get_side(self, point):
        if not self.line:
            return 0
        p1, p2 = self.line
        # Cross product to determine side of the line
        return (p2[0] - p1[0]) * (point[1] - p1[1]) - (p2[1] - p1[1]) * (point[0] - p1[0])

    def update(self, track_id, class_id, bbox):
        class_name = self.class_names.get(class_id, "unknown")
        
        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        center = (cx, cy)

        if track_id not in self.counted_ids:
            self.counted_ids.add(track_id)
            if class_name in self.global_counts:
                self.global_counts[class_name] += 1
            else:
                self.global_counts[class_name] = 1
            
        if self.line:
            current_side = self._get_side(center)
            if track_id in self.previous_sides:
                prev_side = self.previous_sides[track_id]
                
                # Check for crossing (sides have different signs)
                if prev_side * current_side < 0:
                    if class_name not in self.in_counts:
                        self.in_counts[class_name] = 0
                        self.out_counts[class_name] = 0
                        
                    if current_side > 0:
                        self.in_counts[class_name] += 1
                    else:
                        self.out_counts[class_name] += 1
                        
            self.previous_sides[track_id] = current_side

    def get_counts(self):
        return {
            "global": self.global_counts,
            "in": self.in_counts,
            "out": self.out_counts
        }
