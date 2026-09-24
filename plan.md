# Plan de desarrollo: detector y reconocimiento facial local

Actualizado: 16 de septiembre de 2026.

## Objetivo y alcance

Construir un sistema que mantenga video fluido mientras detecta y reconoce rostros, con latencia y errores medidos en el equipo de destino. El primer prototipo trabaja con una cámara y una persona por acceso. No prometer latencia cero ni precisión universal.

Distinguir detección (localizar una cara), reconocimiento (comparar identidad) y autorización (decidir un acceso). El prototipo implementa las dos primeras y una confirmación temporal experimental; no controla puertas ni detecta suplantación por fotografías o pantallas.

## Arquitectura para evitar acumulación de retraso

Cámara -> captura continua -> un único frame reciente compartido.
- La interfaz obtiene ese frame sin esperar al reconocimiento.
- Un trabajador independiente analiza el último frame a una frecuencia limitada inicialmente a 10 Hz.
- No se encolan frames: el nuevo reemplaza al anterior.
- Los modelos se cargan y calientan una vez.
- La captura del prototipo inicia a 640x480 solicitando 30 FPS: la cámara probada entregó aproximadamente 15 FPS frente a 7.4 FPS a 1280x720. Resolución y FPS son configurables; la meta sigue pendiente. Ver VALIDACION.md.
- El detector procesa hasta 640 píxeles de ancho. La alineación y extracción usan la imagen original.
- Los resultados incluyen secuencia y tiempo de captura. Se invalidan visualmente a los 500 ms.
- En esta etapa se muestran dos vistas: video continuo y último frame analizado con sus propios recuadros. No se proyectan coordenadas atrasadas sobre el video nuevo.
- El parámetro de buffer de OpenCV es un intento: hay que verificar el backend de captura y la cámara real.

## Stack inicial y criterios de elección

- Python 3.12 en `.venv`, dependencias fijadas en `requirements.txt`.
- OpenCV 4.12: captura, YuNet 2023mar y SFace 2021dec mediante OpenCV DNN en CPU.
- NumPy: búsqueda exacta por similitud coseno sobre vectores normalizados de 128 dimensiones. Se toma la mejor plantilla por identidad y se compara con la segunda identidad distinta.
- SQLite: persistencia de plantillas con versión del modelo; carga en memoria durante el reconocimiento.
- Tkinter y Pillow: interfaz local de laboratorio, sin servidor web ni dependencia de internet durante el uso.
- DeepFace/Facenet512 y ONNX Runtime quedan como alternativas a comparar si el rendimiento o los errores lo justifican. No asumir que instalar ONNX Runtime cambia el backend de DeepFace.
- FAISS se incorpora cuando las mediciones con el tamaño real de la galería lo justifiquen; no prometer búsquedas de 2 ms sin benchmark.
- Los modelos se descargan del repositorio oficial, se fijan a una revisión y se verifican con los SHA256 de Git LFS. Se conservan manifiesto y licencias.

## Fase 1: prototipo medible (implementada; validación de campo pendiente)

Entregables:
- `app.py`: dos vistas de cámara, métricas y estado explícito si no hay resultado fresco.
- `engine.py`: captura independiente, inferencia, filtros de calidad, plantillas y confirmación temporal.
- Enrolamiento con ocho capturas separadas al menos 700 ms, un solo rostro y control básico de tamaño, iluminación y desenfoque.
- Reemplazo atómico del registro al completar las ocho capturas; cancelación o timeout no dejan registro parcial.
- Confirmación con tres observaciones consecutivas compatibles. Interrupciones, falta de calidad y ausencia de coincidencia reinician la confirmación. No equivale a seguimiento robusto ni prueba de presencia real.
- Evaluaciones de tres segundos etiquetadas por el operador: conocido o desconocido; exportación CSV separada de las plantillas.
- `benchmark.py`: medición con cámara o imagen sintética; separa claramente análisis sin rostro y reconocimiento efectivo.
- Pruebas de continuidad, búsqueda, vectores inválidos y persistencia.

Criterio de salida: ejecutar con una cámara real; registrar resolución y FPS efectivos, inferencia p50/p95, edades de resultados y entrega del frame a la UI. La métrica entrega-a-UI empieza después de `read()` y no mide los buffers internos ni el refresco físico de pantalla.

## Fase 2: calibración y precisión

1. Registrar varias personas con muestras útiles; volver en otra sesión para evaluar, sin reutilizar imágenes de enrolamiento.
2. Probar personas registradas y no registradas, iluminación normal/contraluz, movimiento, lentes, distancias y dos personas a la vez.
3. Separar calibración y prueba final. Ajustar umbral y margen solo sobre calibración; congelarlos antes de probar.
4. Reportar identificación correcta, falsa aceptación de desconocidos, falso rechazo, identidad incorrecta y pruebas sin muestra válida, con sus denominadores.
5. El umbral inicial 0.45 y margen 0.08 son hipótesis de trabajo; la similitud no es un porcentaje de certeza. La consistencia entre frames tampoco implica ensayos independientes.
6. Medir errores por intento y persona, con tamaño de muestra e incertidumbre. Cero errores en pocos intentos no acredita 99.9% de precisión.
7. Comparar otros modelos sobre el mismo conjunto únicamente si existe una necesidad demostrada.

## Fase 3: rendimiento sostenido y robustez

Metas propuestas, condicionadas al hardware y aún no garantizadas:

| Métrica | Meta |
|---|---|
| Video visible | 25-30 FPS sostenidos |
| Latencia física cámara-pantalla | p95 <= 150 ms, medida con referencia externa |
| Rostro de calidad disponible a decisión | p95 <= 500 ms |
| Estabilidad | Una hora sin crecimiento continuo de memoria ni retraso |
| Precisión | Objetivos acordados según uso, validados con datos independientes |

Pruebas pendientes: sesión de una hora, CPU/RAM, cambios de luz, desconexión/reconexión, cámara ocupada, permisos, base dañada y galería del tamaño esperado. Ajustar hilos, frecuencia y resolución con evidencia. Para múltiples cámaras o rostros, rediseñar presupuesto de inferencia y seguimiento antes de ampliar el alcance.

## Fase 4: producto de control de acceso

Pendiente antes de conectar una puerta:
- Detección y validación de ataques con fotografías/pantallas y elección de sensores si corresponde.
- Seguimiento robusto, invalidación de identidad, reglas de autorización y comportamiento seguro ante fallos.
- Persistencia asíncrona de eventos con cola acotada, deduplicación, alertas y recuperación de errores.
- Administración autenticada, eliminación de registros, protección/cifrado de plantillas, respaldo y retención definida.
- Separar monitor y administración si se desarrolla una interfaz web; el video continuo debe tener transporte adecuado.
- Instalador reproducible, modelos incluidos, prueba offline en equipo limpio y procedimiento de actualización.

## Costos, licencias y comercialización

El funcionamiento local elimina la necesidad de una API facial de pago, pero no implica costo total cero ni utilidad neta: considerar hardware, desarrollo, soporte, mantenimiento y actualización. Revisar por separado las licencias de código y pesos de cada modelo seleccionado. La estrategia comercial puede incluir instalación, licencia local y soporte; la fecha de producción depende de superar validaciones, no de un plazo fijo de 15 días.

## Referencias técnicas

- OpenCV: https://docs.opencv.org/4.12.0/d0/dd4/tutorial_dnn_face.html
- YuNet y licencia: https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet
- SFace y licencia: https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface
- Similitud coseno en FAISS: https://github.com/facebookresearch/faiss/wiki/MetricType-and-distances

