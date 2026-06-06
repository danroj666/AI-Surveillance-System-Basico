# ============================================
# AI Surveillance System - Tracker de objetos
# ============================================

from scipy.spatial import distance as dist
from collections import OrderedDict
import numpy as np

class CentroidTracker:
    def __init__(self, max_disappeared=30, max_distance=100):
        self.next_id = 0
        self.objects = OrderedDict()        # ID -> centroide
        self.disappeared = OrderedDict()    # ID -> frames desaparecido
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance

    def register(self, centroid):
        self.objects[self.next_id] = centroid
        self.disappeared[self.next_id] = 0
        self.next_id += 1

    def deregister(self, object_id):
        del self.objects[object_id]
        del self.disappeared[object_id]
        return object_id

    def update(self, detections):
        disappeared_ids = []

        # Sin detecciones: incrementar desaparecidos
        if len(detections) == 0:
            for object_id in list(self.disappeared.keys()):
                self.disappeared[object_id] += 1
                if self.disappeared[object_id] > self.max_disappeared:
                    disappeared_ids.append(self.deregister(object_id))
            return self.objects, disappeared_ids

        # Calcular centroides de detecciones nuevas
        input_centroids = np.array([
            ((x1 + x2) // 2, (y1 + y2) // 2)
            for (x1, y1, x2, y2) in detections
        ])

        # Sin objetos registrados: registrar todos
        if len(self.objects) == 0:
            for c in input_centroids:
                self.register(c)
            return self.objects, disappeared_ids

        # Asociar detecciones con objetos existentes
        object_ids = list(self.objects.keys())
        object_centroids = list(self.objects.values())

        D = dist.cdist(np.array(object_centroids), input_centroids)
        rows = D.min(axis=1).argsort()
        cols = D.argmin(axis=1)[rows]

        used_rows, used_cols = set(), set()

        for (row, col) in zip(rows, cols):
            if row in used_rows or col in used_cols:
                continue
            if D[row, col] > self.max_distance:
                continue
            object_id = object_ids[row]
            self.objects[object_id] = input_centroids[col]
            self.disappeared[object_id] = 0
            used_rows.add(row)
            used_cols.add(col)

        unused_rows = set(range(D.shape[0])) - used_rows
        unused_cols = set(range(D.shape[1])) - used_cols

        for row in unused_rows:
            object_id = object_ids[row]
            self.disappeared[object_id] += 1
            if self.disappeared[object_id] > self.max_disappeared:
                disappeared_ids.append(self.deregister(object_id))

        for col in unused_cols:
            self.register(input_centroids[col])

        return self.objects, disappeared_ids