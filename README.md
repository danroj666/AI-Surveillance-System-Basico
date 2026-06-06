# AI Surveillance System

Sistema de videovigilancia con IA que detecta y rastrea personas en tiempo real usando la GPU.

## ¿Qué hace?

- Detecta personas frente a la cámara
- Le asigna un ID único a cada una y las sigue por la escena
- Cuenta entradas y salidas
- Guarda capturas automáticamente
- Registra todos los eventos en una base de datos
- Muestra estadísticas en un panel web en `http://localhost:5000`

## Por qué la GPU marca la diferencia

Una CPU procesa las cosas en secuencia, una tras otra. Una GPU tiene miles de núcleos pequeños que trabajan en paralelo al mismo tiempo.

Detectar personas en video significa analizar millones de píxeles por segundo. Si eso corre en CPU, el sistema no pasa de 5–8 FPS y el video se ve entrecortado. Al mover ese cálculo a la GPU con CUDA, el mismo proceso corre a 30–45 FPS en tiempo real.

CUDA es la tecnología de NVIDIA que permite programar la GPU directamente para ese tipo de cómputo. En este proyecto se usa para correr el modelo de detección YOLOv8, que es una red neuronal que analiza cada frame del video y decide dónde están las personas y con qué confianza.

Sin GPU esto no sería videovigilancia en tiempo real, sería un slideshow con detecciones.

## Tecnologías

| Tecnología | ¿Por qué se usó? |
|---|---|
| **NVIDIA CUDA** | Permite correr la IA en la GPU en paralelo, logrando detección en tiempo real |
| **PyTorch** | Framework de deep learning que aprovecha CUDA para ejecutar la red neuronal |
| **YOLOv8** | Modelo de detección más rápido y preciso disponible para video en vivo |
| **OpenCV** | Captura el video y dibuja los cuadros y etiquetas sobre la imagen |
| **SciPy** | Calcula distancias entre detecciones para identificar si una persona es la misma entre frames |
| **Flask** | Levanta el servidor web del dashboard de estadísticas |
| **SQLite** | Guarda eventos localmente sin necesitar un servidor de base de datos externo |

## Hardware probado

- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- FPS promedio: 30–45 @ 720p
