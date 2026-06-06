# ============================================
# AI Surveillance System - Main
# ============================================

import cv2
import torch
import datetime
import threading
import time
from ultralytics import YOLO
from config import *
from detector.tracker import CentroidTracker
from data.database import init_db, log_event, log_capture

# ── Estado global ────────────────────────────────────────────
current_persons   = 0
total_entries     = 0
total_exits       = 0
fps               = 0
tracked_ids       = set()
frame_shared      = None
lock              = threading.Lock()

# ── Inicialización ───────────────────────────────────────────
print("[INFO] Iniciando AI Surveillance System...")
init_db()

device = DEVICE if torch.cuda.is_available() and USE_GPU else "cpu"
print(f"[INFO] Usando dispositivo: {device.upper()}")

model = YOLO("yolov8n.pt")
model.to(device)
print("[INFO] Modelo YOLO cargado")

tracker = CentroidTracker(
    max_disappeared=MAX_DISAPPEARED,
    max_distance=MAX_DISTANCE
)

# ── Funciones auxiliares ─────────────────────────────────────
def draw_ui(frame, objects):
    global current_persons

    current_persons = len(objects)
    h, w = frame.shape[:2]

    # Línea de conteo
    cv2.line(frame, (0, COUNTING_LINE_Y), (w, COUNTING_LINE_Y), (0, 255, 255), 2)
    cv2.putText(frame, "LINEA DE CONTEO", (10, COUNTING_LINE_Y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    # Dibujar cada persona tracked
    for person_id, centroid in objects.items():
        cx, cy = centroid
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)
        cv2.putText(frame, f"Persona #{person_id}", (cx - 40, cy - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Panel de stats (esquina superior izquierda)
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (280, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    stats = [
        f"Personas: {current_persons}",
        f"Entradas: {total_entries}",
        f"Salidas:  {total_exits}",
        f"FPS:      {fps:.1f}",
        f"GPU:      {'Activa' if device == 'cuda' else 'CPU'}",
    ]
    for i, text in enumerate(stats):
        cv2.putText(frame, text, (10, 25 + i * 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)

    return frame

def save_capture(frame, person_id):
    if not SAVE_CAPTURES:
        return
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{CAPTURES_PATH}persona_{person_id}_{ts}.jpg"
    cv2.imwrite(filename, frame)
    log_capture(filename, person_id)

# ── Loop principal ───────────────────────────────────────────
def surveillance_loop():
    global fps, frame_shared, total_entries, total_exits, tracked_ids

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print("[ERROR] No se pudo abrir la cámara")
        return

    print("[INFO] Cámara iniciada — presiona Q para salir")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Detección YOLO
        results = model(frame, classes=CLASSES_TO_DETECT,
                        conf=CONFIDENCE_THRESHOLD, verbose=False)

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                detections.append((x1, y1, x2, y2))

                # Dibujar bounding box
                conf = float(box.conf[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
                cv2.putText(frame, f"Persona {conf:.0%}", (x1, y1 - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 2)

        # Tracking
        objects, disappeared_ids = tracker.update(detections)

        # Detectar entradas y salidas
        current_ids = set(objects.keys())

        new_ids = current_ids - tracked_ids
        for pid in new_ids:
            total_entries += 1
            log_event("detected", pid, "Persona detectada")
            save_capture(frame, pid)
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Persona detectada — ID: {pid}")

        for pid in disappeared_ids:
            if pid in tracked_ids:
                total_exits += 1
                log_event("exited", pid, "Persona salió")
                print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] Persona salió     — ID: {pid}")

        tracked_ids = current_ids

        # UI
        frame = draw_ui(frame, objects)

        # FPS
        now = time.time()
        fps = 1.0 / (now - prev_time + 1e-6)
        prev_time = now

        # Compartir frame para dashboard
        with lock:
            frame_shared = frame.copy()

        cv2.imshow("AI Surveillance System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Sistema detenido")

if __name__ == "__main__":
    surveillance_loop()