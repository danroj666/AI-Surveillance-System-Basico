# ============================================
# AI Surveillance System - Configuración
# ============================================

# Cámara
CAMERA_INDEX = 1          # 0 = cámara principal
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Detección
CONFIDENCE_THRESHOLD = 0.5   # Confianza mínima (0-1)
CLASSES_TO_DETECT = [0]      # 0 = persona en YOLO

# Tracking
MAX_DISAPPEARED = 30         # Frames antes de eliminar un ID
MAX_DISTANCE = 100           # Distancia máxima para mismo objeto

# Línea de conteo (posición Y en la imagen)
COUNTING_LINE_Y = 360        # Mitad de 720px

# Base de datos
DB_PATH = "data/surveillance.db"

# Capturas
CAPTURES_PATH = "captures/"
SAVE_CAPTURES = True

# Dashboard web
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
SECRET_KEY = "surveillance_2026"

# GPU
USE_GPU = True
DEVICE = "cuda"              # "cuda" o "cpu"