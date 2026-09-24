# Detector facial — prototipo local

Interfaz de escritorio para ver la cámara, registrar personas y comprobar reconocimiento y rendimiento. Usa Python 3.12, OpenCV YuNet/SFace, NumPy, SQLite y Tkinter. Los modelos se ejecutan con OpenCV DNN en CPU. No requiere internet después de instalar y descargar los modelos.

## Ejecutar

Abre **`iniciar.bat`** con doble clic, o desde PowerShell en esta carpeta:

```powershell
.\.venv\Scripts\python.exe app.py
```

Si no aparece imagen, cierra otras aplicaciones que usen la cámara y prueba:

```powershell
.\.venv\Scripts\python.exe app.py --camera 1
.\.venv\Scripts\python.exe app.py --backend msmf
```

La configuración inicial es **640×480**: en la prueba de este equipo entregó aproximadamente 15 FPS, frente a 7.4 FPS a 720p. No se alcanzaron todavía los 25–30 FPS objetivo. Si quieres comparar detalle y fluidez:

```powershell
.\.venv\Scripts\python.exe app.py --width 1280 --height 720
.\.venv\Scripts\python.exe benchmark.py --width 640 --height 480 --seconds 30
```

Una mejor iluminación puede ayudar si la cámara reduce su frecuencia por exposición, pero la causa no está confirmada. Acércate más a 640×480 para cumplir el tamaño mínimo de rostro. Media Foundation no entregó imágenes durante la prueba breve; el método predeterminado es DirectShow.

Para una cámara IP, usa `--backend auto --camera "URL"`. El prototipo prioriza USB; el comportamiento de red y reconexión aún necesita validación. No publiques URLs con contraseñas. El índice y los controles de captura dependen del dispositivo.

## Primera prueba

1. Mantén una sola persona visible, buena luz y la cara completa, aproximadamente frontal.
2. Escribe un nombre y pulsa **Registrar / reemplazar**. Se toman ocho muestras útiles separadas en el tiempo. Cambia ligeramente el ángulo, sin girar demasiado. El registro vence a los 45 segundos si no logra completarse.
3. Aléjate y vuelve a entrar. Se mostrará **Coincidencia estable** después de tres observaciones consecutivas compatibles.
4. En **Identidad real**, selecciona tu nombre y pulsa **Probar durante 3 segundos**. Repite bajo distintas condiciones y en otra sesión.
5. Para medir falsas aceptaciones, otra persona que NO esté registrada debe hacer la prueba seleccionando **(Desconocido)**. Seleccionar desconocido mientras aparece alguien registrado no es una prueba válida.

La cámara principal muestra el frame más reciente sin esperar a la IA. El panel de análisis muestra recuadros sobre la imagen realmente procesada; se actualiza inicialmente a un máximo de 10 Hz. Una edad mayor a 500 ms invalida visualmente el resultado. No hay seguimiento por movimiento ni superposición predictiva en la vista principal.

## Interpretar los resultados

- **Similitud**: comparación coseno con la mejor plantilla. No es probabilidad ni porcentaje de certeza.
- **Umbral**: mínimo de similitud; predeterminado experimental `0.45`.
- **Margen**: separación mínima con la segunda identidad distinta; predeterminado `0.08`. Con solo una persona registrada no hay segundo candidato.
- **Identificación correcta**: durante el intento se confirmó la identidad declarada sin confirmar otra.
- **Falsa aceptación**: una persona desconocida recibió una identidad registrada.
- **Falso rechazo**: una persona registrada tuvo muestras válidas pero no obtuvo confirmación.
- **Identidad incorrecta**: se confirmó a otra persona durante un intento de alguien registrado; se cuenta aunque también haya una coincidencia correcta.
- **Sin muestra válida**: no hubo un rostro utilizable; se reporta aparte, no como acierto.

Los intentos duran tres segundos y las observaciones dentro de uno están correlacionadas. El operador declara la identidad real: etiquetas equivocadas invalidan la evaluación. Los resultados son exploratorios; usa personas y sesiones variadas, un conjunto para calibrar y otro reservado para evaluar.

```powershell
.\.venv\Scripts\python.exe app.py --threshold 0.50 --margin 0.10 --hz 10
```

Subir el umbral puede reducir falsas aceptaciones y aumentar rechazos. No cambiarlo usando el conjunto reservado para la prueba final.

## Rendimiento y límites de medición

La interfaz informa FPS de captura/vista, p95 del tiempo de análisis y edad del resultado. Entrega-a-vista empieza al regresar `camera.read()` y acaba tras preparar la imagen para Tkinter; **no mide la latencia física completa cámara-pantalla**. Los percentiles corresponden a ventanas recientes. Los buffers de hardware/controlador pueden agregar retraso no visible en esa métrica.

```powershell
.\.venv\Scripts\python.exe benchmark.py --seconds 30
.\.venv\Scripts\python.exe benchmark.py --synthetic --seconds 5
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

El benchmark sintético verifica carga/ejecución del detector sobre una imagen negra y calentamiento del reconocedor, pero no mide precisión ni reconocimiento de una persona. El benchmark de cámara informa cuántas imágenes realmente produjeron un embedding. Para evaluar estabilidad, mantener la aplicación una hora y observar también CPU y memoria en el Administrador de tareas.

## Archivos locales

- `data/people.sqlite3`: nombres y plantillas faciales (128 dimensiones, modelo versionado). No se guardan fotos ni video. Las plantillas sí son datos biométricos y este prototipo las guarda **sin cifrado**.
- `data/evaluacion_FECHA.csv`: intentos y resultados de esta sesión.
- `data/rendimiento_FECHA.json`: métricas recientes; se actualizan cada cinco segundos y al cerrar.
- `data/benchmark_*.json`: última ejecución de cada modo del benchmark.
- `models/manifest.json`: revisión de OpenCV Zoo y SHA256 de los modelos.
- `models/*.LICENSE`: licencias de los modelos descargados.

El prototipo no incorpora autorización de acceso, apertura de puertas, detección de fotos/pantallas, administración autenticada ni borrado desde la interfaz. Para borrar todos los registros de laboratorio, cierra la aplicación y elimina `data/people.sqlite3` y sus archivos auxiliares `-wal`/`-shm` si existen. No elimines archivos mientras la aplicación está abierta.

## Reinstalar desde cero

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe setup_models.py
.\.venv\Scripts\python.exe app.py
```

La descarga inicial requiere aproximadamente 39 MB para los modelos, además de los paquetes. `setup_models.py` resuelve una revisión del repositorio oficial, valida los SHA256 de Git LFS y conserva las licencias. La aplicación no descarga nada al arrancar.

## Sistema integrado y prueba de vida

`sistema_final` incorpora una prueba de vida pasiva MiniFASNet antes de confirmar una identidad o aceptar muestras de enrolamiento. Utiliza un ensamble V2 + V1SE en ONNX y acumula evidencia temporal: un resultado aislado no confirma ni rechaza. Los estados dudosos se muestran como inconclusos y no habilitan la identidad.

```powershell
.\sistema_final\iniciar.bat
```

Esta protección reduce ataques básicos con fotografías y pantallas, pero una cámara RGB y un modelo pasivo no garantizan resistencia a todos los ataques. Antes de usarlo para control de acceso deben realizarse pruebas locales con personas reales, fotos impresas, teléfonos y videos, midiendo falsas aceptaciones y falsos rechazos. Los umbrales se encuentran en `sistema_final/config.json` y no deben ajustarse usando el mismo conjunto reservado para la evaluación final.

Hoja de ruta y pendientes: [plan.md](plan.md).
