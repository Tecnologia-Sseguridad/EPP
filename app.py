"""Desktop prototype: live camera, enrolment, and labelled evaluation trials."""
import argparse
import csv
from collections import deque
from datetime import datetime, timezone
import json
from pathlib import Path
import queue
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox

import cv2
import numpy as np
from PIL import Image, ImageTk

from engine import Camera, Worker, ROOT, classify_trial


class App:
    def __init__(self, root, args):
        self.root = root
        root.title("Detector facial · Laboratorio local")
        root.geometry("1240x850")
        root.minsize(1000, 760)
        self.stop = threading.Event()
        backend = {"dshow": cv2.CAP_DSHOW, "msmf": cv2.CAP_MSMF, "auto": cv2.CAP_ANY}[args.backend]
        source = int(args.camera) if args.camera.isdigit() else args.camera
        self.camera = Camera(source, backend, self.stop, args.width, args.height, args.fps)
        self.worker = Worker(self.camera, self.stop, args.threshold, args.margin, args.hz)
        self.trials = []
        self.active_trial = None
        self.last_sequence = -1
        self.last_analysis = -1
        self.render_times = deque(maxlen=120)
        self.ages = deque(maxlen=1000)
        self.inferences = deque(maxlen=1000)
        self.last_export = time.perf_counter()
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TLabel", font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI", 19, "bold"))
        outer = ttk.Frame(root, padding=16)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Detector facial · Prototipo", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Procesamiento local | Una persona por vez | Reconocimiento experimental, sin apertura de puertas").pack(anchor="w", pady=(0, 12))
        views = ttk.Frame(outer)
        views.pack(fill="x")
        left = ttk.LabelFrame(views, text="Cámara en vivo · independiente de la IA", padding=8)
        left.pack(side="left", fill="both", expand=True)
        self.live = ttk.Label(left, text="Esperando cámara...", anchor="center")
        self.live.pack(fill="both", expand=True)
        right = ttk.LabelFrame(views, text="Última imagen analizada · recuadros sincronizados", padding=8)
        right.pack(side="left", fill="both", expand=True, padx=(12, 0))
        self.analyzed = ttk.Label(right, text="Cargando modelos...", anchor="center")
        self.analyzed.pack(fill="both", expand=True)
        self.result = ttk.Label(outer, text="Preparando...", font=("Segoe UI", 15, "bold"))
        self.result.pack(anchor="w", pady=(14, 4))
        self.metrics = ttk.Label(outer, text="", wraplength=1180)
        self.metrics.pack(anchor="w")
        self.status = ttk.Label(outer, text="", wraplength=1180)
        self.status.pack(anchor="w", pady=4)
        register = ttk.LabelFrame(outer, text="1. Enrolar · 8 capturas de calidad, separadas en el tiempo", padding=10)
        register.pack(fill="x", pady=8)
        ttk.Label(register, text="Nombre o identificador:").pack(side="left")
        self.name = ttk.Entry(register, width=25)
        self.name.pack(side="left", padx=8)
        self.enroll_button = ttk.Button(register, text="Registrar / reemplazar", command=self.enroll)
        self.enroll_button.pack(side="left")
        ttk.Button(register, text="Cancelar", command=lambda: self.command("cancel", "")).pack(side="left", padx=8)
        self.people = ttk.Label(outer, text="Personas registradas: ninguna", wraplength=1180)
        self.people.pack(anchor="w")
        evaluate = ttk.LabelFrame(outer, text="2. Evaluar · vuelve a entrar en escena y declara quién está frente a la cámara", padding=10)
        evaluate.pack(fill="x", pady=8)
        ttk.Label(evaluate, text="Identidad real:").pack(side="left")
        self.expected = ttk.Combobox(evaluate, values=["(Desconocido)"], state="readonly", width=24)
        self.expected.set("(Desconocido)")
        self.expected.pack(side="left", padx=8)
        self.trial_button = ttk.Button(evaluate, text="Probar durante 3 segundos", command=self.start_trial)
        self.trial_button.pack(side="left")
        self.summary = ttk.Label(outer, text="Sin pruebas. La similitud NO es un porcentaje de certeza.", wraplength=1180)
        self.summary.pack(anchor="w", pady=4)
        ttk.Label(outer, text="Resultados: data/evaluacion_*.csv y data/rendimiento_*.json. No se guardan fotos ni video.\n"
                  "Las plantillas faciales se guardan localmente sin cifrado en este prototipo. Sin detección de fotos/pantallas.", wraplength=1180).pack(anchor="w", pady=8)
        self.camera.start()
        self.worker.start()
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(20, self.tick)

    def command(self, action, name):
        try:
            self.worker.commands.put_nowait((action, name))
        except queue.Full:
            messagebox.showinfo("Espera", "Hay una operación pendiente.")

    def enroll(self):
        name = self.name.get().strip()
        if not name or len(name) > 80 or name == "(Desconocido)":
            messagebox.showinfo("Nombre", "Escribe un nombre de 1 a 80 caracteres distinto de (Desconocido).")
            return
        if self.active_trial:
            return
        if name in self.worker.names and not messagebox.askyesno("Reemplazar registro", f"¿Reemplazar las 8 muestras de {name}?"):
            return
        self.command("enroll", name)

    def start_trial(self):
        if self.worker.enrolling or not self.worker.names:
            messagebox.showinfo("Evaluación", "Primero termina un registro. Luego aléjate y vuelve para usar capturas nuevas.")
            return
        now = time.perf_counter()
        self.active_trial = {"started": now, "expected": self.expected.get(), "accepted": set(), "observations": 0, "best_score": -1.0}
        self.command("reset", "")
        self.trial_button.configure(state="disabled")
        self.enroll_button.configure(state="disabled")
        self.summary.configure(text="Prueba en curso: mantente frente a la cámara durante 3 segundos...")

    def finish_trial(self):
        trial = self.active_trial
        expected = trial["expected"]
        accepted = trial["accepted"]
        outcome = classify_trial(expected, accepted, trial["observations"])
        row = {"utc": datetime.now(timezone.utc).isoformat(), "real": expected,
               "aceptadas": " | ".join(sorted(accepted)), "resultado": outcome,
               "muestras_validas": trial["observations"], "similitud_max": round(trial["best_score"], 4),
               "umbral": self.worker.threshold, "margen": self.worker.margin}
        self.trials.append(row)
        path = ROOT / "data" / f"evaluacion_{self.run_id}.csv"
        path.parent.mkdir(exist_ok=True)
        exists = path.exists()
        with path.open("a", newline="", encoding="utf-8-sig") as output:
            writer = csv.DictWriter(output, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        known = [r for r in self.trials if r["real"] != "(Desconocido)" and r["resultado"] != "sin_muestra_valida"]
        unknown = [r for r in self.trials if r["real"] == "(Desconocido)" and r["resultado"] != "sin_muestra_valida"]
        false_accept = sum(r["resultado"] == "falsa_aceptacion" for r in unknown)
        false_reject = sum(r["resultado"] == "falso_rechazo" for r in known)
        wrong = sum(r["resultado"] == "identidad_incorrecta" for r in known)
        correct = sum(r["resultado"] == "identificacion_correcta" for r in known)
        invalid = sum(r["resultado"] == "sin_muestra_valida" for r in self.trials)
        self.summary.configure(text=f"Última: {outcome.replace('_', ' ')}. Conocidos: {correct}/{len(known)} correctos, "
            f"{false_reject} rechazos, {wrong} identidades incorrectas. Desconocidos: {false_accept}/{len(unknown)} falsas aceptaciones. "
            f"Sin muestra válida: {invalid}. Son pruebas exploratorias, no una certificación de precisión.")
        self.active_trial = None
        self.trial_button.configure(state="normal")
        self.enroll_button.configure(state="normal")

    @staticmethod
    def show(label, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((565, 318))
        photo = ImageTk.PhotoImage(image)
        label.configure(image=photo, text="")
        label.image = photo

    def tick(self):
        if self.stop.is_set():
            return
        now = time.perf_counter()
        frame = self.camera.latest.get()
        if frame and now - frame.captured < 0.5:
            if frame.sequence != self.last_sequence:
                self.show(self.live, frame.image)
                self.last_sequence = frame.sequence
                self.render_times.append(now)
                self.ages.append((time.perf_counter() - frame.captured) * 1000)
        else:
            self.live.configure(image="", text=self.camera.status if frame is None else "Sin imágenes recientes")
            self.live.image = None
        result = self.worker.latest.get()
        fresh = result is not None and now - result["captured"] < 0.5 and frame is not None and now - frame.captured < 0.5
        if fresh:
            if result["sequence"] != self.last_analysis:
                self.show(self.analyzed, result["preview"])
                self.last_analysis = result["sequence"]
                self.inferences.append(result["inference_ms"])
                trial = self.active_trial
                if trial and result["captured"] >= trial["started"] and not result["enrolling"] and result["valid"]:
                    trial["observations"] += 1
                    trial["best_score"] = max(trial["best_score"], result["score"])
                    if result["identity"]:
                        trial["accepted"].add(result["identity"])
            if result["enrolling"]:
                text = "Capturando muestras de registro..."
            elif result["identity"]:
                text = f"Coincidencia estable: {result['identity']}"
            elif result["candidate"]:
                text = "Verificando consistencia..."
            elif result["valid"]:
                text = "Sin coincidencia suficiente / desconocido"
            else:
                text = result["quality"]
            self.result.configure(text=f"{text}   |   Similitud: {result['score']:.3f}")
        else:
            self.result.configure(text="Sin resultado reciente · identidad no confirmada")
            self.analyzed.configure(image="", text="Esperando análisis reciente...")
            self.analyzed.image = None
            self.last_analysis = -1
        ui_fps = (len(self.render_times) - 1) / (self.render_times[-1] - self.render_times[0]) if len(self.render_times) > 1 and now - self.render_times[-1] < 1 else 0
        p95 = float(np.percentile(self.ages, 95)) if self.ages else 0
        inference = result["p95_ms"] if fresh else 0
        age = (now - result["captured"]) * 1000 if fresh else 0
        self.metrics.configure(text=f"Captura: {self.camera.fps if frame else 0:.1f} FPS | Vista: {ui_fps:.1f} FPS | "
            f"Entrega a vista p95: {p95:.0f} ms | Análisis p95: {inference:.0f} ms | Edad del resultado: {age:.0f} ms\n"
            f"Umbral: {self.worker.threshold:.2f} | Margen: {self.worker.margin:.2f} | La entrega a vista excluye el buffer interno de la cámara y el refresco físico de pantalla.")
        self.status.configure(text=f"{self.camera.status} · {self.worker.status}" + (f" · {result['quality']}" if fresh else ""))
        names = self.worker.names
        self.people.configure(text="Personas registradas: " + (", ".join(names) if names else "ninguna"))
        self.expected.configure(values=["(Desconocido)"] + names)
        if self.active_trial and now - self.active_trial["started"] >= 3:
            self.finish_trial()
        if now - self.last_export > 5:
            self.export_metrics()
            self.last_export = now
        self.root.after(15, self.tick)

    def export_metrics(self):
        def stats(values):
            return {"n": len(values), "median_ms": float(np.median(values)), "p95_ms": float(np.percentile(values, 95))} if values else None
        folder = ROOT / "data"
        folder.mkdir(exist_ok=True)
        report = {"utc": datetime.now(timezone.utc).isoformat(), "capture_fps": self.camera.fps,
                  "delivery_to_ui": stats(self.ages), "analysis": stats(self.inferences),
                  "note": "Ventanas recientes; entrega a UI no equivale a latencia física cámara-pantalla.",
                  "threshold": self.worker.threshold, "margin": self.worker.margin}
        target = folder / f"rendimiento_{self.run_id}.json"
        temporary = target.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        temporary.replace(target)

    def close(self):
        self.stop.set()
        self.export_metrics()
        self.worker.join(timeout=2)
        self.camera.join(timeout=2)
        self.root.destroy()


def main():
    parser = argparse.ArgumentParser(description="Prototipo local de reconocimiento facial")
    parser.add_argument("--camera", default="0", help="Índice USB o URL de cámara")
    parser.add_argument("--backend", choices=["dshow", "msmf", "auto"], default="dshow")
    parser.add_argument("--threshold", type=float, default=0.45)
    parser.add_argument("--margin", type=float, default=0.08)
    parser.add_argument("--hz", type=float, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--smoke-test", action="store_true", help="Cerrar interfaz después de 5 segundos")
    args = parser.parse_args()
    if not -1 <= args.threshold <= 1 or not 0 <= args.margin <= 2 or not 1 <= args.hz <= 30:
        parser.error("Umbral [-1,1], margen [0,2], frecuencia [1,30]")
    if min(args.width, args.height, args.fps) <= 0:
        parser.error("Resolución y FPS deben ser positivos")
    root = tk.Tk()
    app = App(root, args)
    if args.smoke_test:
        root.after(5000, app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
