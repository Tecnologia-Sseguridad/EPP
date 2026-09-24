# Validación inicial — 16 de septiembre de 2026

## Comprobado

- Entorno `.venv` con Python 3.12.10; NumPy 2.2.6, OpenCV 4.12.0.88 y Pillow 11.3.0 instalados. `pip check`: sin incompatibilidades.
- YuNet y SFace descargados de OpenCV Zoo; SHA256 verificados; revisión y licencias conservadas en `models/`.
- Carga y calentamiento de ambos modelos completados.
- Seis pruebas automatizadas ejecutadas y aprobadas: búsqueda, margen entre identidades distintas, continuidad, descarte de frames pendientes, persistencia/reemplazo y vectores inválidos.
- Se añadieron dos pruebas de clasificación de evaluaciones; la solicitud para ejecutar la suite final de ocho pruebas fue rechazada por el usuario. Estas dos pruebas nuevas quedan sin ejecutar.
- Interfaz iniciada con cámara y cerrada automáticamente sin errores en una prueba de cinco segundos. Se generaron métricas locales.

## Mediciones breves (no equivalen a una prueba de estabilidad)

| Prueba | Captura | Análisis mediana | Análisis p95 | Rostros válidos |
|---|---:|---:|---:|---:|
| Imagen negra 1280×720, 3 s | No aplica | 18.1 ms | 21.2 ms | 0 |
| Cámara DirectShow 1280×720, 12 s | 7.4 FPS | 17.3 ms | 22.9 ms | 0 |
| Cámara DirectShow 640×480, 10 s | 15.0 FPS | 26.2 ms | 35.9 ms | 0 |

La prueba de UI a 720p informó una entrega-a-vista p95 de 48.5 ms para 18 frames. Esta medición empieza después de la lectura del dispositivo y termina al preparar la vista; no es latencia física cámara-pantalla. Media Foundation no entregó frames dentro de una prueba de 10 segundos; no se concluye que sea incompatible para todos los casos.

A 640×480 el detector recibe más píxeles verticales que al reducir 720p a 640×360; por ello estos tiempos no son una comparación pura de rendimiento por resolución. La captura mejoró al bajar resolución. No se ha establecido si el límite proviene del dispositivo, exposición, controlador o formato de captura. El inicio de la aplicación se configuró a 640×480 usando esta evidencia.

## Aún no demostrado

No hubo rostros detectados en las capturas de estas mediciones. Por tanto, **no hay todavía evidencia de precisión, tiempo real de extracción facial sobre una persona, ni éxito del enrolamiento con la cámara**. Esos pasos requieren que el usuario se coloque frente al dispositivo y haga las pruebas guiadas. El calentamiento de SFace con una imagen vacía solo verifica que ejecuta.

Pendientes: enrolamiento real, conocidos y desconocidos, calibración de umbrales, condiciones variadas, sesión de una hora, recuperación de cámara, objetivo de 25–30 FPS y prueba externa de latencia. No existe detección de suplantación ni integración con puertas.
