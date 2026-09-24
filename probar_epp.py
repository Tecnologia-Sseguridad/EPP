"""Verificación EPP por persona con ByteTrack y evidencia temporal."""
from collections import deque
from dataclasses import dataclass, field

import cv2
from ultralytics import YOLO


MODELO = "models/ppe_yolo26n.pt"
CONFIANZA_MINIMA = 0.28
IOU_NMS = 0.45
VENTANA = 15
FRAMES_PERSONA = 2
FRAMES_DE_GRACIA = 5
FRAMES_PARA_RECONECTAR = 45
SUAVIZADO_CAJA = 0.45

# Elementos exigidos por la decisión principal. Se pueden ampliar más adelante.
REQUERIDOS = ("casco", "chaleco", "guantes")
REGLAS = {
    "casco": {"positivo": "helmet", "negativo": "no_helmet", "zona": "cabeza"},
    "chaleco": {"positivo": "vest", "negativo": "none", "zona": "torso"},
    "guantes": {"positivo": "gloves", "negativo": "no_gloves", "zona": "manos"},
    "antiparras": {"positivo": "goggles", "negativo": "no_goggle", "zona": "cabeza"},
    "botas": {"positivo": "boots", "negativo": "no_boots", "zona": "pies"},
}
NOMBRES_ES = {
    "helmet": "casco", "gloves": "guantes", "vest": "chaleco",
    "boots": "botas", "goggles": "antiparras", "none": "sin chaleco",
    "Person": "persona", "no_helmet": "sin casco",
    "no_goggle": "sin antiparras", "no_gloves": "sin guantes",
    "no_boots": "sin botas",
}
CLASES_NEGATIVAS = {"none", "no_helmet", "no_goggle", "no_gloves", "no_boots"}


@dataclass
class Persona:
    caja: list[float]
    confianza: float
    identificador: int
    aciertos: int = 1
    ausente: int = 0
    evidencia: dict = field(default_factory=lambda: {
        nombre: deque(maxlen=VENTANA) for nombre in REGLAS
    })
    confianzas: dict = field(default_factory=lambda: {
        nombre: deque(maxlen=VENTANA) for nombre in REGLAS
    })
    visible: dict = field(default_factory=lambda: {
        nombre: False for nombre in REGLAS
    })


def area(caja):
    return max(1.0, caja[2] - caja[0]) * max(1.0, caja[3] - caja[1])


def interseccion(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def quitar_duplicados(detecciones, limite=0.65):
    """Elimina cajas anidadas repetidas de la misma clase."""
    elegidas = []
    for det in sorted(detecciones, key=lambda d: d["confianza"], reverse=True):
        repetida = any(
            det["nombre"] != "Person"
            and det["nombre"] == otra["nombre"]
            and interseccion(det["caja"], otra["caja"]) /
            min(area(det["caja"]), area(otra["caja"])) >= limite
            for otra in elegidas
        )
        if not repetida:
            elegidas.append(det)
    return elegidas


def zona_relativa(caja_epp, caja_persona):
    cx = (caja_epp[0] + caja_epp[2]) / 2
    cy = (caja_epp[1] + caja_epp[3]) / 2
    x1, y1, x2, y2 = caja_persona
    return ((cx - x1) / max(1.0, x2 - x1), (cy - y1) / max(1.0, y2 - y1))


def compatible_con_zona(caja_epp, caja_persona, zona):
    rx, ry = zona_relativa(caja_epp, caja_persona)
    if not (-0.12 <= rx <= 1.12 and -0.08 <= ry <= 1.08):
        return False
    limites = {
        "cabeza": (-0.08, 0.38),
        "torso": (0.12, 0.78),
        "manos": (0.08, 0.92),
        "pies": (0.62, 1.08),
    }
    minimo, maximo = limites[zona]
    return minimo <= ry <= maximo


def asignar_a_personas(epp, personas):
    """Asigna cada caja EPP a una única persona usando inclusión, zona y distancia."""
    asignadas = {track_id: {} for track_id in personas}
    por_clase = {
        regla["positivo"]: (elemento, regla["zona"])
        for elemento, regla in REGLAS.items()
    }
    por_clase.update({
        regla["negativo"]: (elemento, regla["zona"])
        for elemento, regla in REGLAS.items()
    })

    for det in epp:
        if det["nombre"] not in por_clase:
            continue
        elemento, zona = por_clase[det["nombre"]]
        mejor = None
        mejor_puntaje = -10.0
        for track_id, persona in personas.items():
            if not compatible_con_zona(det["caja"], persona.caja, zona):
                continue
            cobertura = interseccion(det["caja"], persona.caja) / area(det["caja"])
            rx, ry = zona_relativa(det["caja"], persona.caja)
            distancia = ((rx - 0.5) ** 2 + (ry - 0.45) ** 2) ** 0.5
            puntaje = cobertura - 0.20 * distancia
            if puntaje > mejor_puntaje:
                mejor, mejor_puntaje = track_id, puntaje
        if mejor is not None and mejor_puntaje >= 0.35:
            anterior = asignadas[mejor].get(det["nombre"])
            if anterior is None or det["confianza"] > anterior["confianza"]:
                asignadas[mejor][det["nombre"]] = det
    return asignadas


def distancia_reconexion(caja_nueva, persona):
    """Distancia normalizada para recuperar un ID perdido cerca de su última zona."""
    cx_n = (caja_nueva[0] + caja_nueva[2]) / 2
    cy_n = (caja_nueva[1] + caja_nueva[3]) / 2
    cx_a = (persona.caja[0] + persona.caja[2]) / 2
    cy_a = (persona.caja[1] + persona.caja[3]) / 2
    diagonal = max(1.0, ((persona.caja[2] - persona.caja[0]) ** 2
                         + (persona.caja[3] - persona.caja[1]) ** 2) ** 0.5)
    razon_area = area(caja_nueva) / area(persona.caja)
    if not 0.35 <= razon_area <= 2.85:
        return None
    return ((cx_n - cx_a) ** 2 + (cy_n - cy_a) ** 2) ** 0.5 / diagonal


def zona_visible(caja, zona, forma):
    """Evita afirmar ausencia cuando la región corporal está fuera del encuadre."""
    alto, ancho = forma[:2]
    x1, y1, x2, y2 = caja
    h = max(1.0, y2 - y1)
    if zona == "cabeza":
        return y1 > 4 and y1 + 0.35 * h < alto - 4
    if zona == "torso":
        return y1 + 0.72 * h < alto - 4
    if zona == "manos":
        return x1 > 4 and x2 < ancho - 4 and y1 + 0.88 * h < alto - 4
    if zona == "pies":
        return y2 < alto - 4
    return False


def agregar_evidencia(persona, detecciones, forma):
    for elemento, regla in REGLAS.items():
        persona.visible[elemento] = zona_visible(persona.caja, regla["zona"], forma)
        if not persona.visible[elemento]:
            persona.evidencia[elemento].append(None)
            persona.confianzas[elemento].append(None)
            continue
        positiva = detecciones.get(regla["positivo"])
        negativa = detecciones.get(regla["negativo"])
        if positiva and negativa:
            positiva, negativa = (
                (positiva, None) if positiva["confianza"] >= negativa["confianza"]
                else (None, negativa)
            )
        if positiva:
            persona.evidencia[elemento].append(1)
            persona.confianzas[elemento].append(positiva["confianza"])
        elif negativa:
            persona.evidencia[elemento].append(-1)
            persona.confianzas[elemento].append(negativa["confianza"])
        else:
            # Una zona visible sin detección aporta evidencia débil, no un NO inmediato.
            persona.evidencia[elemento].append(0)
            persona.confianzas[elemento].append(None)


def decidir(persona, elemento):
    if not persona.visible[elemento]:
        return "no visible"
    valores = [v for v in persona.evidencia[elemento] if v is not None]
    if not valores:
        return "verificando"
    positivos = sum(v == 1 for v in valores)
    negativos = sum(v == -1 for v in valores)
    muestras = len(valores)
    promedio = positivos / muestras

    # Confirmación temprana: dos positivos consistentes bastan para responder rápido.
    if positivos >= 2 and promedio >= 0.60:
        return "si"
    evidencia_explicita = positivos + negativos
    if negativos >= 2 and evidencia_explicita and negativos / evidencia_explicita >= 0.67:
        return "no"

    # Para casco y chaleco la ausencia sostenida también es evidencia. En guantes
    # esperamos la clase explícita no_gloves porque las manos pueden estar ocultas.
    if elemento in {"casco", "chaleco"} and muestras >= 7 and promedio <= 0.20:
        return "no"
    return "verificando"


def color_estado(estado):
    return {
        "si": (65, 205, 85), "no": (45, 45, 235),
        "verificando": (40, 190, 255), "no visible": (150, 150, 150),
    }[estado]


def dibujar_panel(frame, persona):
    x1, y1, x2, y2 = [int(round(v)) for v in persona.caja]
    alto, ancho = frame.shape[:2]
    x1, x2 = max(0, x1), min(ancho - 1, x2)
    y1, y2 = max(0, y1), min(alto - 1, y2)
    decisiones = {elemento: decidir(persona, elemento) for elemento in REQUERIDOS}
    estados = list(decisiones.values())
    general = ("INCOMPLETO" if "no" in estados else
               "COMPLETO" if all(e == "si" for e in estados) else "VERIFICANDO")
    color_general = ((65, 205, 85) if general == "COMPLETO" else
                     (45, 45, 235) if general == "INCOMPLETO" else (40, 190, 255))
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_general, 2, cv2.LINE_AA)

    lineas = [(f"Persona #{persona.identificador}", color_general)]
    for elemento in REQUERIDOS:
        estado = decisiones[elemento]
        marca = {"si": "[OK]", "no": "[X]", "verificando": "[...]", "no visible": "[?]"}[estado]
        lineas.append((f"{marca} {elemento.title()}", color_estado(estado)))
    lineas.append((f"Estado: {general}", color_general))

    panel_ancho, linea_alto = 205, 23
    panel_alto = len(lineas) * linea_alto + 8
    px = min(max(0, x1), max(0, ancho - panel_ancho))
    py = y1 - panel_alto - 4 if y1 >= panel_alto + 4 else min(alto - panel_alto, y1 + 4)
    overlay = frame.copy()
    cv2.rectangle(overlay, (px, py), (px + panel_ancho, py + panel_alto),
                  (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.78, frame, 0.22, 0, frame)
    for indice, (texto, color) in enumerate(lineas):
        cv2.putText(frame, texto, (px + 7, py + 20 + indice * linea_alto),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.51, color, 1, cv2.LINE_AA)


def dibujar_debug(frame, detecciones):
    for det in detecciones:
        x1, y1, x2, y2 = map(int, det["caja"])
        color = (45, 45, 235) if det["nombre"] in CLASES_NEGATIVAS else (220, 150, 40)
        texto = f'{NOMBRES_ES.get(det["nombre"], det["nombre"])} {det["confianza"]:.2f}'
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        cv2.putText(frame, texto, (x1, max(14, y1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)


print("Cargando modelo EPP...")
model = YOLO(MODELO)
nombres_originales = dict(model.names)
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara")

personas = {}
siguiente_persona = 1
debug = False
print("Cámara iniciada. 'd': depuración, 'q': salir.")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Error: no se puede recibir video de la cámara.")
            break
        resultado = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=CONFIANZA_MINIMA, iou=IOU_NMS, verbose=False,
        )[0]
        detecciones = []
        if resultado.boxes is not None:
            cajas = resultado.boxes.xyxy.cpu().tolist()
            confianzas = resultado.boxes.conf.cpu().tolist()
            clases = resultado.boxes.cls.int().cpu().tolist()
            ids = (resultado.boxes.id.int().cpu().tolist()
                   if resultado.boxes.id is not None else [None] * len(cajas))
            for caja, confianza, clase, track_id in zip(cajas, confianzas, clases, ids):
                detecciones.append({
                    "caja": caja, "confianza": float(confianza),
                    "clase": clase, "nombre": nombres_originales[clase],
                    "track_id": track_id,
                })
        detecciones = quitar_duplicados(detecciones)
        for persona in personas.values():
            persona.ausente += 1

        for det in detecciones:
            if det["nombre"] != "Person" or det["track_id"] is None:
                continue
            track_id = det["track_id"]
            persona = personas.get(track_id)
            if persona is None:
                candidatos = []
                for id_anterior, perdida in personas.items():
                    if perdida.ausente <= 0:
                        continue
                    distancia = distancia_reconexion(det["caja"], perdida)
                    if distancia is not None and distancia <= 0.65:
                        candidatos.append((distancia, id_anterior, perdida))
                if candidatos:
                    _, id_anterior, persona = min(candidatos, key=lambda fila: fila[0])
                    del personas[id_anterior]
                    personas[track_id] = persona
                else:
                    persona = Persona(det["caja"], det["confianza"], siguiente_persona)
                    siguiente_persona += 1
                    personas[track_id] = persona
            persona.caja = [
                anterior * (1 - SUAVIZADO_CAJA) + nuevo * SUAVIZADO_CAJA
                for anterior, nuevo in zip(persona.caja, det["caja"])
            ]
            persona.confianza = persona.confianza * 0.6 + det["confianza"] * 0.4
            persona.aciertos = min(100, persona.aciertos + 1)
            persona.ausente = 0

        personas = {
            track_id: persona for track_id, persona in personas.items()
            if persona.ausente <= FRAMES_PARA_RECONECTAR
        }
        cajas_epp = [det for det in detecciones if det["nombre"] != "Person"]
        personas_activas = {
            track_id: persona for track_id, persona in personas.items()
            if persona.ausente == 0
        }
        asignadas = asignar_a_personas(cajas_epp, personas_activas)
        for track_id, persona in personas.items():
            if persona.ausente == 0:
                agregar_evidencia(persona, asignadas.get(track_id, {}), frame.shape)
            if persona.aciertos >= FRAMES_PERSONA and persona.ausente <= FRAMES_DE_GRACIA:
                dibujar_panel(frame, persona)

        if debug:
            dibujar_debug(frame, detecciones)
        cv2.imshow("Verificador EPP por persona", frame)
        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord("d"):
            debug = not debug
finally:
    cap.release()
    cv2.destroyAllWindows()
    print("Programa finalizado correctamente.")
