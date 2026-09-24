"""Asistente de voz local para resultados confirmados del sistema."""
import queue
import threading

import pyttsx3


LABELS = {"casco": "casco", "chaleco": "chaleco reflectante", "guantes": "guantes"}


def epp_message(status, decisions, required):
    required = [name for name in required if name in decisions]
    missing = [LABELS.get(name, name) for name in required if decisions[name] == "no"]
    if status == "COMPLETO":
        return "Acceso autorizado. Equipo de protección personal completo. Puede continuar."
    if missing and len(missing) == len(required):
        return "Atención. Persona detectada sin equipo de protección personal. Por favor, equípese antes de continuar."
    if missing:
        items = missing[0] if len(missing) == 1 else ", ".join(missing[:-1]) + " y " + missing[-1]
        verb = "Falta" if len(missing) == 1 else "Faltan"
        return f"Atención. Equipo de protección personal incompleto. {verb} {items}. Por favor, complete su equipamiento."
    return "Atención. No fue posible confirmar el equipo de protección personal."


class VoiceAssistant(threading.Thread):
    """Cola no bloqueante; inicializa SAPI dentro de su propio hilo."""
    def __init__(self, stop, enabled=True, voice_name="Sabina", rate=165):
        super().__init__(daemon=True, name="voice-assistant")
        self.stop = stop
        self.enabled = enabled
        self.voice_name = voice_name.casefold()
        self.rate = rate
        self.messages = queue.Queue(maxsize=3)
        self.status = "En espera" if enabled else "Desactivada"

    def announce_epp(self, status, decisions, required):
        if not self.enabled:
            return
        message = epp_message(status, decisions, required)
        try:
            self.messages.put_nowait(message)
        except queue.Full:
            # Una alerta antigua pierde valor; se conserva la que ya está sonando.
            try:
                self.messages.get_nowait()
            except queue.Empty:
                pass
            try:
                self.messages.put_nowait(message)
            except queue.Full:
                pass

    def run(self):
        if not self.enabled:
            return
        engine = None
        try:
            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            selected = next((voice for voice in voices if self.voice_name in voice.name.casefold()), None)
            if selected is None:
                raise RuntimeError("No se encontró la voz Microsoft Sabina")
            engine.setProperty("voice", selected.id)
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", 1.0)
            self.status = "Sabina lista"
            while not self.stop.is_set():
                try:
                    message = self.messages.get(timeout=0.1)
                except queue.Empty:
                    continue
                self.status = "Hablando"
                engine.say(message)
                engine.runAndWait()
                self.status = "Sabina lista"
        except Exception as error:
            self.status = f"Error de voz: {error}"
        finally:
            if engine is not None:
                engine.stop()
