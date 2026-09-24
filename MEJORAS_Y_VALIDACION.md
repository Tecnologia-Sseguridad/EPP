# Puesto de reconocimiento y EPP

Abrir desde `abrir_sistema.bat` o ejecutar `venv\Scripts\python.exe -m sistema_final.main`.
El `app.py` de la raíz continúa siendo el laboratorio facial original.
Los cambios de esta entrega están en **EPP Original**, no en la carpeta hermana EPP.

## Interfaz

- Navegación lateral: Reconocimiento, Protección EPP y Administración.
- Vídeo independiente del análisis, encuadre monocromo opcional y resultados fuera de la imagen.
- Iconos SVG locales, sin librerías web, descargas ni servicios externos.
- Personas: búsqueda, selección múltiple, enrolamiento y eliminación confirmada.
- Registros: fechas locales, filtros, páginas de 100, selección por casillas, evidencia,
  eliminación por selección y CSV de selección o página. Exportar no borra registros.
- Configuración: cinco EPP configurables, cámara, resolución, FPS y silencio de voz.
  La cámara requiere reiniciar. La configuración conserva los demás valores existentes.
- Diagnóstico exportable sin identidades, fotos, vectores faciales ni URL de cámara.
- Captura manual y pantalla completa (F11 / Escape).

## Correcciones de decisión

El modelo EPP predice sobre cada imagen completa. Los objetos no necesitan obtener
un ID de seguimiento para aparecer. La continuidad espacial se aplica a la persona.
Las decisiones se basan en observaciones temporales recientes, no en cuántas veces
la interfaz dibuja un resultado. Un borde inferior recortado ya no bloquea todo el equipo.

La evidencia positiva requiere al menos 3 fotogramas y 0,30 segundos; la negativa,
4 fotogramas. Debe predominar dentro de una ventana de 2 segundos. Un resultado
confirmado tolera una pérdida breve y deja de confirmar después de 2,5 segundos sin
renovación. Estos valores son criterios iniciales de operación, no umbrales calibrados
para una tasa de error certificada.

Para guantes se requieren dos cajas distintas en cada observación positiva.
Una mano moviéndose de lado no puede acreditar las dos. Una mano desnuda contradice
el cumplimiento aunque también haya un guante. Sin pose o identificación de cada mano,
la geometría no garantiza que las cajas sean dos manos: hay que validar los ejemplos reales.

Los cambios de presencia y las ausencias reinician la evaluación. El seguimiento
espacial no identifica personas; no garantiza distinguir un reemplazo instantáneo
en la misma posición. No se emite un resultado completo cuando la persona está ausente
o el análisis está vencido. El módulo EPP por sí solo no autoriza un acceso físico.

El facial conserva YuNet, SFace y MiniFASNet, pero no reinicia la confirmación por
leer dos veces el mismo fotograma. La prueba de vida se reinicia si cambia claramente
el vector facial o existe una interrupción de observación. Desactivar MiniFASNet ya
no presenta una prueba de vida ficticiamente aprobada.
Una observación actual sospechosa tampoco se aprueba por una mayoría de observaciones
reales anteriores; se exige volver a confirmar.

## Medir el funcionamiento real

Las pruebas unitarias verifican lógica, no precisión de los modelos en una cámara.
No se ha medido ni certificado un 99 %. Tampoco se entrenaron modelos nuevos.

Preparar vídeos cortos con una sola condición real constante en cada clip: casco
puesto/ausente, chaleco puesto/ausente, ambos guantes, uno, ninguno; diferentes personas,
distancias, luz frontal/contraluz, giros y manos junto al cuerpo. Los clips de cambio
de equipo deben separarse por condición para que su etiqueta sea correcta.

Crear un archivo JSON junto a los vídeos, por ejemplo:

```json
[
  {"video": "equipo_completo.mp4", "expected": {"casco": "si", "chaleco": "si", "guantes": "si"}},
  {"video": "sin_casco.mp4", "expected": {"casco": "no", "chaleco": "si", "guantes": "si"}}
]
```

Ejecutar:

```text
venv\Scripts\python.exe validar_epp.py pruebas.json --output evaluacion_epp.json
```

El informe distingue falsos cumplimientos, falsas faltas, abstenciones y cobertura.
Una abstención no cuenta como acierto. Los fotogramas correlacionados no prueban
una precisión poblacional: separar personas y sesiones de ajuste de las de validación.
Si el modelo no ve los guantes, guardar esos casos para mejorar el conjunto de entrenamiento;
mantener un estado antiguo en verde no soluciona ese fallo.

Para facial, comprobar personas enroladas y desconocidas por separado, más fotografías
y pantallas para prueba de vida. Utilizar el laboratorio y las instrucciones de VALIDACION.md.

## Verificaciones automatizadas

```text
venv\Scripts\python.exe -m unittest discover -s sistema_final/tests -v
venv\Scripts\python.exe -m unittest discover -s tests -v
```

Los tests de interfaz usan servicios simulados y no abren la cámara ni modifican datos reales.
La herramienta tools/preview_workspace.py permite revisar las pantallas con esos mismos datos.
