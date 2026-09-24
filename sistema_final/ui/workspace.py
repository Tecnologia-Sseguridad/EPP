"""Operational desktop views; processing and storage remain in their services."""
import csv
import ctypes
import json
import queue
import time
import tkinter as tk
from collections import deque
from datetime import date, datetime, timedelta
from tkinter import filedialog, messagebox, ttk

import cv2
from PIL import Image, ImageTk

from .main_window import MainWindow as ExistingActions
from .icons import Icons
from sistema_final.core.configuration import ROOT, save_config
from sistema_final.core.database import (
    list_people, delete_person, get_required_epp, query_events, record_event,
)

STATE_LABELS = {"si": "Confirmado", "no": "Falta", "verificando": "Por confirmar",
                "no visible": "Sin evidencia reciente", "fuera de encuadre": "Revisar encuadre"}
ITEM_ICONS = {"casco": "helmet", "chaleco": "vest", "guantes": "glove",
              "antiparras": "shield", "botas": "shield"}


class MainWindow(ExistingActions):
    def __init__(self, root, camera, facial, epp, stop, config, voice=None):
        self.root, self.camera, self.facial, self.epp = root, camera, facial, epp
        self.stop, self.config, self.voice = stop, config, voice
        self.last_sequences = {"facial": -1, "epp": -1, "admin": -1}
        self.last_camera_sequence = -1
        self.was_enrolling = False
        self.filtered_events = []
        self.event_state = {k: {"key": None, "time": 0., "seen": 0.} for k in ("facial", "epp")}
        self.voice_epp_state = {"announced": None, "pending": None, "since": 0., "seen": 0.}
        self._presence_id = None
        self._pending_record = None
        self._pending_since = 0.
        self._facial_pending = None
        self._facial_since = 0.
        self._page = 0
        self._has_more = False
        self._camera_times = deque(maxlen=60)
        self.overlay = tk.BooleanVar(value=True)
        self.overlay_buttons = []
        self.voice_enabled = tk.BooleanVar(value=config.get("voice_enabled", True))
        self.fullscreen = False
        self.maximized = False
        self._normal_geometry = "1320x850+80+60"
        self._drag_origin = None
        self.icons = Icons(root)
        root.title("SSeguridad | Control y verificación")
        root.geometry(self._normal_geometry)
        root.minsize(1080, 700)
        root.overrideredirect(True)
        self._styles()
        border = tk.Frame(root, background="#17191d", padx=1, pady=1)
        border.pack(fill="both", expand=True)
        self._build_titlebar(border)
        shell = ttk.Frame(border)
        shell.pack(fill="both", expand=True)
        rail = ttk.Frame(shell, padding=(16, 24), width=210, style="Rail.TFrame")
        rail.pack(side="left", fill="y")
        rail.pack_propagate(False)
        ttk.Label(rail, text="SSEGURIDAD", style="Brand.TLabel").pack(anchor="w")
        ttk.Label(rail, text="Control y verificación", style="Rail.TLabel").pack(anchor="w", pady=(3, 30))
        self.nav_buttons = []
        for index, (label, icon) in enumerate((
                ("Reconocimiento", "face"), ("Protección EPP", "shield"), ("Administración", "users"))):
            button = ttk.Button(rail, text="  " + label,
                                image=self.icons.get(icon, 19, "#e9eaec"), compound="left",
                                style="Nav.TButton", command=lambda i=index: self.tabs.select(i))
            button.pack(fill="x", pady=3)
            self.nav_buttons.append(button)
        ttk.Button(rail, text="Pantalla completa", style="RailAction.TButton",
                   command=self.toggle_fullscreen).pack(side="bottom", fill="x")
        ttk.Label(rail, text="F11  Pantalla completa\nEsc  Volver", style="Rail.TLabel").pack(side="bottom", pady=16)
        content = ttk.Frame(shell, padding=(30, 24), style="Workspace.TFrame")
        content.pack(side="left", fill="both", expand=True)
        content.columnconfigure(0, weight=1)
        content.rowconfigure(0, weight=1)
        self.tabs = ttk.Notebook(content, style="Workspace.TNotebook")
        self.tabs.grid(row=0, column=0, sticky="nsew")
        self.facial_tab, self.epp_tab, self.admin_tab = (ttk.Frame(self.tabs) for _ in range(3))
        for page in (self.facial_tab, self.epp_tab, self.admin_tab):
            self.tabs.add(page)
        self._build_facial()
        self._build_epp()
        self._build_admin()
        footer = ttk.Frame(content, padding=(0, 12, 0, 0))
        footer.grid(row=1, column=0, sticky="ew")
        self.system_status = ttk.Label(footer, text="Iniciando servicios", style="Muted.TLabel")
        self.system_status.pack(side="left")
        self.performance = ttk.Label(footer, style="Muted.TLabel")
        self.performance.pack(side="right")
        self.tabs.bind("<<NotebookTabChanged>>", self._tab_changed)
        root.bind("<F11>", lambda e: self.toggle_fullscreen())
        root.bind("<Escape>", lambda e: self.leave_fullscreen())
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(30, self._show_in_taskbar)
        self._tab_changed()
        root.after(25, self.tick)

    def _styles(self):
        self.root.configure(background="#17191d")
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure(".", font=("Segoe UI", 10), background="#f2f3f5", foreground="#202124")
        style.configure("TFrame", background="#f2f3f5")
        style.configure("Workspace.TFrame", background="#f2f3f5")
        style.configure("TLabel", background="#f2f3f5")
        style.configure("Rail.TFrame", background="#17191d")
        style.configure("Rail.TLabel", background="#17191d", foreground="#9ca0a8", font=("Segoe UI", 9))
        style.configure("Brand.TLabel", background="#17191d", foreground="#ffffff", font=("Segoe UI Semibold", 16))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 25), foreground="#17191d")
        style.configure("Result.TLabel", font=("Segoe UI Semibold", 16), background="#eceef0",
                        foreground="#17191d", padding=(14, 12))
        style.configure("Section.TLabel", font=("Segoe UI Semibold", 11))
        style.configure("Muted.TLabel", foreground="#686c73", font=("Segoe UI", 9))
        style.configure("Card.TFrame", background="#ffffff")
        style.configure("Card.TLabel", background="#ffffff")
        style.configure("TButton", padding=(14, 9), background="#ffffff", bordercolor="#d9dadd",
                        relief="flat", focusthickness=0)
        style.map("TButton", background=[("active", "#e8e9eb"), ("pressed", "#dedfe2")])
        style.configure("Nav.TButton", anchor="w", padding=(13, 14), relief="flat", background="#17191d",
                        foreground="#d9dbe0", borderwidth=0, focusthickness=0)
        style.map("Nav.TButton", background=[("selected", "#30333a"), ("active", "#25282e")],
                  foreground=[("selected", "#ffffff"), ("active", "#ffffff")])
        style.configure("RailAction.TButton", padding=(12, 10), background="#24272d", foreground="#ffffff",
                        borderwidth=0, relief="flat")
        style.map("RailAction.TButton", background=[("active", "#30333a")])
        style.configure("TEntry", padding=8, fieldbackground="#ffffff", bordercolor="#d9dadd")
        style.configure("TCombobox", padding=7, fieldbackground="#ffffff", bordercolor="#d9dadd")
        style.map("TCombobox", fieldbackground=[("readonly", "#ffffff")],
                  selectbackground=[("readonly", "#ffffff")], selectforeground=[("readonly", "#242424")])
        style.configure("TNotebook", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(20, 11), background="#e7e8ea", borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", "#ffffff"), ("active", "#dedfe2")])
        style.layout("Workspace.TNotebook.Tab", [])
        style.layout("Workspace.TNotebook", [("Notebook.client", {"sticky": "nswe"})])
        style.configure("Workspace.TNotebook", borderwidth=0, tabmargins=0,
                        background="#f2f3f5", relief="flat")
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff",
                        rowheight=34, borderwidth=0, font=("Segoe UI", 10))
        style.configure("Treeview.Heading", font=("Segoe UI Semibold", 9), padding=10,
                        background="#e7e8ea", relief="flat")
        style.map("Treeview", background=[("selected", "#dadada")], foreground=[("selected", "#111111")])

    def _build_titlebar(self, parent):
        bar = tk.Frame(parent, height=44, background="#17191d")
        bar.pack(fill="x")
        bar.pack_propagate(False)
        title = tk.Label(bar, text="  SSEGURIDAD   /   CONTROL Y VERIFICACIÓN",
                         background="#17191d", foreground="#d7d9dd",
                         font=("Segoe UI Semibold", 9), anchor="w")
        title.pack(side="left", fill="both", expand=True)
        controls = tk.Frame(bar, background="#17191d")
        controls.pack(side="right", fill="y")
        for symbol, command, hover in (("—", self.minimize_window, "#292c32"),
                                        ("□", self.toggle_maximize, "#292c32"),
                                        ("×", self.close, "#b3261e")):
            button = tk.Button(controls, text=symbol, command=command, width=5,
                               background="#17191d", foreground="#f0f0f0",
                               activebackground=hover, activeforeground="#ffffff",
                               relief="flat", borderwidth=0, font=("Segoe UI", 12))
            button.pack(side="left", fill="y")
        for widget in (bar, title):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._drag_window)
            widget.bind("<Double-Button-1>", lambda _e: self.toggle_maximize())

    def _start_drag(self, event):
        if not self.maximized:
            self._drag_origin = (event.x_root - self.root.winfo_x(), event.y_root - self.root.winfo_y())

    def _drag_window(self, event):
        if self._drag_origin and not self.maximized:
            self.root.geometry(f"+{event.x_root-self._drag_origin[0]}+{event.y_root-self._drag_origin[1]}")

    def _show_in_taskbar(self):
        try:
            handle = ctypes.windll.user32.GetParent(self.root.winfo_id())
            extended = ctypes.windll.user32.GetWindowLongW(handle, -20)
            ctypes.windll.user32.SetWindowLongW(handle, -20, (extended & ~0x80) | 0x40000)
            self.root.withdraw()
            self.root.after(10, self.root.deiconify)
        except (AttributeError, OSError):
            pass

    def minimize_window(self):
        self.root.overrideredirect(False)
        self.root.iconify()
        self.root.bind("<Map>", self._restore_border, add="+")

    def _restore_border(self, _event=None):
        self.root.after(10, lambda: self.root.overrideredirect(True))

    def toggle_maximize(self):
        if self.fullscreen:
            return
        if self.maximized:
            self.root.geometry(self._normal_geometry)
        else:
            self._normal_geometry = self.root.geometry()
            self.root.geometry(f"{self.root.winfo_screenwidth()}x{self.root.winfo_screenheight()-40}+0+0")
        self.maximized = not self.maximized

    def _button(self, parent, text, command, icon=None, **kwargs):
        options = {"text": text, "command": command}
        if icon:
            options.update(image=self.icons.get(icon, 17), compound="left")
        return ttk.Button(parent, **options, **kwargs)

    def _heading(self, parent, title, subtitle):
        ttk.Label(parent, text=title, style="Title.TLabel").pack(anchor="w")
        ttk.Label(parent, text=subtitle, style="Muted.TLabel").pack(anchor="w", pady=(4, 22))

    def _video_panel(self, parent, title):
        box = ttk.Frame(parent, style="Card.TFrame", padding=6)
        box.pack(fill="both", expand=True)
        label = tk.Label(box, text=title, background="#111214", foreground="#dddddd",
                         font=("Segoe UI", 11), borderwidth=0)
        label.place(x=0, y=0, relwidth=1, relheight=1)
        return label

    def _recognition_layout(self, parent, title, subtitle):
        self._heading(parent, title, subtitle)
        toolbar = ttk.Frame(parent)
        toolbar.pack(fill="x", pady=(0, 14))
        overlay_button = ttk.Button(toolbar, text="Encuadre visible", command=self.toggle_overlay)
        overlay_button.pack(side="left")
        self.overlay_buttons.append(overlay_button)
        self._button(toolbar, "Guardar captura", self.snapshot, "camera").pack(side="right")
        body = ttk.Frame(parent)
        body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=286)
        body.rowconfigure(0, weight=1)
        video = ttk.Frame(body)
        video.grid(row=0, column=0, sticky="nsew", padx=(0, 18))
        side = ttk.Frame(body, style="Card.TFrame", padding=24, width=286)
        side.grid(row=0, column=1, sticky="nsew")
        side.grid_propagate(False)
        side.pack_propagate(False)
        return self._video_panel(video, "Esperando cámara"), side

    def toggle_overlay(self):
        self.overlay.set(not self.overlay.get())
        for button in self.overlay_buttons:
            button.configure(text="Encuadre visible" if self.overlay.get() else "Encuadre oculto")

    def _build_facial(self):
        self.facial_video, side = self._recognition_layout(
            self.facial_tab, "Reconocimiento facial", "Identidad y prueba de vida")
        ttk.Label(side, text="RESULTADO", style="Card.TLabel").pack(anchor="w", pady=(0, 14))
        self.facial_result = ttk.Label(side, text="Preparando", style="Result.TLabel", wraplength=220)
        self.facial_result.pack(anchor="w", fill="x")
        ttk.Separator(side).pack(fill="x", pady=20)
        self.facial_live = ttk.Label(side, text="Prueba de vida pendiente", style="Card.TLabel", wraplength=220)
        self.facial_live.pack(anchor="w")
        self.facial_details = ttk.Label(side, style="Card.TLabel", wraplength=220)
        self.facial_details.pack(anchor="w", pady=16)
        ttk.Label(side, text="Mire a la cámara. Mantenga el rostro visible y evite la luz detrás de usted.",
                  style="Card.TLabel", wraplength=220).pack(side="bottom", anchor="w", pady=10)

    def _build_epp(self):
        self.epp_video, side = self._recognition_layout(
            self.epp_tab, "Protección personal", "Verificación individual del equipo requerido")
        ttk.Label(side, text="RESULTADO", style="Card.TLabel").pack(anchor="w", pady=(0, 14))
        self.epp_result = ttk.Label(side, text="Esperando persona", style="Result.TLabel", wraplength=220)
        self.epp_result.pack(anchor="w", fill="x")
        self.item_labels = {}
        self.item_containers = {}
        items = ttk.Frame(side, style="Card.TFrame")
        items.pack(fill="x", pady=(10, 0))
        for name in ITEM_ICONS:
            container = ttk.Frame(items, style="Card.TFrame")
            self.item_containers[name] = container
            ttk.Separator(container).pack(fill="x", pady=(8, 8))
            row = ttk.Frame(container, style="Card.TFrame")
            row.pack(fill="x", pady=(0, 4))
            ttk.Label(row, image=self.icons.get(ITEM_ICONS[name], 23), style="Card.TLabel").pack(side="left", padx=(0, 10))
            text = ttk.Frame(row, style="Card.TFrame")
            text.pack(side="left", fill="x")
            ttk.Label(text, text=name.capitalize(), style="Card.TLabel").pack(anchor="w")
            label = ttk.Label(text, text="Por confirmar", style="Card.TLabel")
            label.pack(anchor="w", pady=(2, 0))
            self.item_labels[name] = label
        self.epp_details = ttk.Label(side, text="Muestre el equipo a la cámara",
                                     style="Card.TLabel", wraplength=220)
        self.epp_details.pack(anchor="w", pady=18)
        self._refresh_required_rows()

    def _refresh_required_rows(self):
        required = tuple(self.epp.required)
        if required == getattr(self, "_shown_required", None):
            return
        for name, container in self.item_containers.items():
            container.pack_forget()
        for name in required:
            if name in self.item_containers:
                self.item_containers[name].pack(fill="x")
        self._shown_required = required

    def _build_admin(self):
        self._heading(self.admin_tab, "Administración", "Personas, evidencias y configuración del puesto")
        self.admin_tabs = ttk.Notebook(self.admin_tab)
        self.admin_tabs.pack(fill="both", expand=True)
        people, records, settings, diagnostics = (ttk.Frame(self.admin_tabs, padding=(12, 18)) for _ in range(4))
        for tab, label, icon in ((people, "Personas", "users"), (records, "Registros", "records"),
                                 (settings, "Configuración", "settings"), (diagnostics, "Diagnóstico", "info")):
            self.admin_tabs.add(tab, text=" " + label, image=self.icons.get(icon, 18), compound="left")
        self.admin_tabs.bind("<<NotebookTabChanged>>", self._admin_changed)
        self._build_people(people)
        self._build_records(records)
        self._build_settings(settings)
        self._build_diagnostics(diagnostics)
        self.refresh_people()

    def _build_people(self, parent):
        left = ttk.Frame(parent, width=230)
        left.pack(side="left", fill="y", padx=(0, 18))
        right = ttk.Frame(parent)
        right.pack(side="left", fill="both", expand=True)
        ttk.Label(left, text="Personas enroladas", style="Section.TLabel").pack(anchor="w")
        self.people_search = tk.StringVar()
        ttk.Entry(left, textvariable=self.people_search, width=27).pack(fill="x", pady=10)
        self.people_search.trace_add("write", lambda *_: self.refresh_people())
        self.people_list = tk.Listbox(left, width=27, height=12, selectmode="extended",
                                     exportselection=False, background="#ffffff", foreground="#242424",
                                     selectbackground="#d6d6d6", selectforeground="#111111",
                                     relief="flat", borderwidth=0, font=("Segoe UI", 11))
        self.people_list.pack(fill="both", expand=True)
        self.people_count = ttk.Label(left, style="Muted.TLabel")
        self.people_count.pack(anchor="w", pady=8)
        self._button(left, "Actualizar", self.refresh_people, "refresh").pack(fill="x", pady=4)
        self._button(left, "Eliminar selección", self.remove_selected, "delete").pack(fill="x")
        self.enroll_video = self._video_panel(right, "Cámara de enrolamiento")
        ttk.Label(right, text="Nombre o identificador").pack(anchor="w", pady=(12, 4))
        self.enroll_name = ttk.Entry(right)
        self.enroll_name.pack(fill="x")
        bar = ttk.Frame(right)
        bar.pack(fill="x", pady=8)
        self._button(bar, "Enrolar / reemplazar", self.start_enrollment, "face").pack(side="left")
        ttk.Button(bar, text="Cancelar", command=self.cancel_enrollment).pack(side="left", padx=8)
        self.enroll_status = ttk.Label(right, text="Se requieren 8 muestras y prueba de vida.", wraplength=560)
        self.enroll_status.pack(anchor="w")

    def _build_records(self, parent):
        f = ttk.Frame(parent)
        f.pack(fill="x")
        self.record_person = ttk.Entry(f, width=18)
        self.record_type = ttk.Combobox(f, state="readonly", width=10, values=("Todos", "facial", "epp"))
        self.record_status = ttk.Combobox(f, state="readonly", width=19,
            values=("Todos", "RECONOCIDO", "DESCONOCIDO", "SPOOF_RECHAZADO", "COMPLETO", "INCOMPLETO"))
        self.record_type.set("Todos")
        self.record_status.set("Todos")
        self.record_from, self.record_to = ttk.Entry(f, width=12), ttk.Entry(f, width=12)
        for i, (title, widget) in enumerate((("Persona", self.record_person), ("Módulo", self.record_type),
                ("Resultado", self.record_status), ("Desde (AAAA-MM-DD)", self.record_from),
                ("Hasta (AAAA-MM-DD)", self.record_to))):
            ttk.Label(f, text=title, style="Muted.TLabel").grid(row=0, column=i, sticky="w", pady=(0, 4))
            widget.grid(row=1, column=i, sticky="ew", padx=(0, 10))
            widget.bind("<Return>", lambda e: self.search_events())
        actions = ttk.Frame(parent)
        actions.pack(fill="x", pady=12)
        self._button(actions, "Buscar", self.search_events, "search").pack(side="left")
        ttk.Button(actions, text="Hoy", command=lambda: self.date_preset(0)).pack(side="left", padx=6)
        ttk.Button(actions, text="Últimos 7 días", command=lambda: self.date_preset(6)).pack(side="left")
        ttk.Button(actions, text="Limpiar filtros", command=self.clear_filters).pack(side="left", padx=6)
        cols = ("fecha", "tipo", "persona", "estado", "casco", "chaleco", "guantes", "vida", "imagen")
        pane = ttk.Frame(parent)
        pane.pack(fill="both", expand=True)
        self.events_table = ttk.Treeview(pane, columns=cols, height=8, show="tree headings", selectmode="extended")
        self.events_table.heading("#0", text="Sel.")
        self.events_table.column("#0", width=42, stretch=False)
        for key, width in zip(cols, (158, 68, 130, 150, 95, 95, 95, 85, 65)):
            self.events_table.heading(key, text=key.capitalize(), command=lambda c=key: self.sort_events(c))
            self.events_table.column(key, width=width, minwidth=65, anchor="w")
        self.events_table.grid(row=0, column=0, sticky="nsew")
        pane.rowconfigure(0, weight=1)
        pane.columnconfigure(0, weight=1)
        sy = ttk.Scrollbar(pane, orient="vertical", command=self.events_table.yview)
        sx = ttk.Scrollbar(pane, orient="horizontal", command=self.events_table.xview)
        sy.grid(row=0, column=1, sticky="ns")
        sx.grid(row=1, column=0, sticky="ew")
        self.events_table.configure(yscrollcommand=sy.set, xscrollcommand=sx.set)
        self.events_table.bind("<Button-1>", self.toggle_row)
        self.events_table.bind("<<TreeviewSelect>>", self.selection_changed)
        self.events_table.bind("<Double-1>", lambda e: self.view_evidence())
        self.events_table.bind("<Control-a>", lambda e: self.select_all_events())
        bar = ttk.Frame(parent)
        bar.pack(fill="x", pady=(12, 0))
        self._button(bar, "Evidencia", self.view_evidence, "camera").pack(side="left")
        ttk.Button(bar, text="Seleccionar página", command=self.select_all_events).pack(side="left", padx=5)
        self._button(bar, "Eliminar selección", self.delete_selected_events, "delete").pack(side="left")
        self._button(bar, "Exportar CSV", self.export_events, "export").pack(side="left", padx=5)
        more = ttk.Menubutton(bar, text="Más acciones")
        menu = tk.Menu(more, tearoff=False)
        menu.add_command(label="Quitar selección", command=lambda: self.events_table.selection_remove(self.events_table.selection()))
        menu.add_separator()
        menu.add_command(label="Vaciar historial completo...", command=self.delete_all_events)
        more.configure(menu=menu)
        more.pack(side="right")
        pager = ttk.Frame(parent)
        pager.pack(fill="x", pady=(10, 0))
        ttk.Button(pager, text="Anterior", command=lambda: self.change_page(-1)).pack(side="left")
        ttk.Button(pager, text="Siguiente", command=lambda: self.change_page(1)).pack(side="left", padx=5)
        self.records_count = ttk.Label(pager, style="Muted.TLabel")
        self.records_count.pack(side="right")
        # Reserve action bars before giving remaining space to the table.
        bar.pack_configure(side="bottom")
        pager.pack_configure(side="bottom")
        pane.pack_forget()
        pane.pack(fill="both", expand=True)
        self.refresh_events()

    def _build_settings(self, parent):
        ttk.Label(parent, text="Equipo requerido", style="Section.TLabel").pack(anchor="w")
        ttk.Label(parent, text="Se aplica a la siguiente evaluación.", style="Muted.TLabel").pack(anchor="w", pady=5)
        box = ttk.Frame(parent)
        box.pack(fill="x", pady=(6, 16))
        self.requirement_vars = {}
        required = set(get_required_epp())
        for name in ITEM_ICONS:
            var = tk.BooleanVar(value=name in required)
            self.requirement_vars[name] = var
            ttk.Checkbutton(box, text=name.capitalize(), variable=var).pack(side="left", padx=(0, 16))
        ttk.Button(parent, text="Guardar requisitos", command=self.save_requirements).pack(anchor="w")
        ttk.Separator(parent).pack(fill="x", pady=20)
        ttk.Label(parent, text="Cámara y presentación", style="Section.TLabel").pack(anchor="w")
        form = ttk.Frame(parent)
        form.pack(anchor="w", pady=12)
        self.camera_setting = tk.StringVar(value=str(self.config["camera"]))
        self.resolution_setting = tk.StringVar(value=f'{self.config["width"]}x{self.config["height"]}')
        self.fps_setting = tk.StringVar(value=str(self.config["fps"]))
        fields = (
            ("Cámara (índice o URL)", ttk.Entry(form, textvariable=self.camera_setting, width=38)),
            ("Resolución", ttk.Combobox(form, textvariable=self.resolution_setting, state="readonly",
                                      values=("640x480", "1280x720", "1920x1080"))),
            ("FPS solicitados", ttk.Combobox(form, textvariable=self.fps_setting, state="readonly",
                                             values=("15", "25", "30", "60"))),
        )
        for i, (label, widget) in enumerate(fields):
            ttk.Label(form, text=label).grid(row=i, column=0, sticky="w", padx=(0, 25), pady=6)
            widget.grid(row=i, column=1, sticky="ew")
        ttk.Checkbutton(parent, text="Avisos de voz (Sabina)", variable=self.voice_enabled,
                        command=self.set_voice).pack(anchor="w", pady=6)
        ttk.Button(parent, text="Guardar configuración", command=self.save_settings).pack(anchor="w", pady=8)
        self.settings_message = ttk.Label(parent, text="Los cambios de cámara se aplican al reiniciar.",
                                           style="Muted.TLabel")
        self.settings_message.pack(anchor="w", pady=5)
        ttk.Label(parent, text="La prueba de vida y la confirmación temporal permanecen activas según la configuración del servicio.",
                  style="Muted.TLabel", wraplength=760).pack(anchor="w", pady=12)

    def _build_diagnostics(self, parent):
        ttk.Label(parent, text="Estado del puesto", style="Section.TLabel").pack(anchor="w")
        self.diagnostic_text = tk.Text(parent, height=15, background="#ffffff", foreground="#303030",
                                      relief="flat", font=("Consolas", 10), padx=16, pady=16, state="disabled")
        self.diagnostic_text.pack(fill="both", expand=True, pady=12)
        self._button(parent, "Exportar diagnóstico", self.export_diagnostics, "export").pack(anchor="w")
        ttk.Label(parent, text="El informe no incluye nombres, imágenes ni plantillas faciales.",
                  style="Muted.TLabel").pack(anchor="w", pady=8)

    def refresh_people(self):
        names = list_people()
        search = self.people_search.get().strip().casefold() if hasattr(self, "people_search") else ""
        self.people_list.delete(0, tk.END)
        for name in names:
            if search in name.casefold():
                self.people_list.insert(tk.END, name)
        self.people_count.configure(text=f"{self.people_list.size()} de {len(names)} personas")

    def remove_selected(self):
        names = [self.people_list.get(i) for i in self.people_list.curselection()]
        if not names:
            messagebox.showinfo("Personas", "Seleccione una o varias personas.")
            return
        if not messagebox.askyesno("Eliminar personas", f"¿Eliminar las plantillas de {len(names)} persona(s)? Los registros históricos se conservan."):
            return
        if self.facial.enrolling:
            messagebox.showinfo("Enrolamiento", "Cancele el enrolamiento antes de eliminar personas.")
            return
        try:
            for name in names:
                delete_person(name)
            self.facial.command("reload")
            self.refresh_people()
        except Exception as error:
            messagebox.showerror("Personas", str(error))

    def start_enrollment(self):
        self.facial.activate()
        super().start_enrollment()

    def _tab_changed(self, _event=None):
        if not hasattr(self, "admin_tabs"):
            return
        selected = self.tabs.index(self.tabs.select())
        for i, button in enumerate(self.nav_buttons):
            button.state(["selected" if i == selected else "!selected"])
        facial_active = selected == 0 or (selected == 2 and self.admin_tabs.index(self.admin_tabs.select()) == 0)
        self.facial.activate() if facial_active else self.facial.deactivate()
        self.epp.activate() if selected == 1 else self.epp.deactivate()
        self.last_camera_sequence = -1

    def _admin_changed(self, _event=None):
        self._tab_changed()
        if self.admin_tabs.index(self.admin_tabs.select()) == 1:
            self.refresh_events()

    def search_events(self):
        self._page = 0
        self.refresh_events()

    def refresh_events(self):
        if not hasattr(self, "events_table"):
            return
        try:
            for entry in (self.record_from, self.record_to):
                if entry.get().strip():
                    date.fromisoformat(entry.get().strip())
            if self.record_from.get() and self.record_to.get() and self.record_from.get() > self.record_to.get():
                raise ValueError("La fecha inicial debe ser anterior a la final.")
            rows = query_events(self.record_person.get(), self.record_type.get(), self.record_status.get(),
                                self.record_from.get(), self.record_to.get(), 101, offset=self._page * 100)
        except (ValueError, OSError) as error:
            messagebox.showerror("Filtros", f"Revise las fechas (AAAA-MM-DD).\n{error}")
            return
        self._has_more = len(rows) > 100
        self.filtered_events = rows[:100]
        self.events_table.delete(*self.events_table.get_children())
        for r in self.filtered_events:
            stamp = datetime.fromisoformat(r["timestamp"]).astimezone().strftime("%d/%m/%Y %H:%M:%S")
            self.events_table.insert("", "end", iid=str(r["id"]), text="☐", values=(
                stamp, r["event_type"], r["person_name"] or "—", r["status"],
                STATE_LABELS.get(r["helmet_status"], r["helmet_status"] or "—"),
                STATE_LABELS.get(r["vest_status"], r["vest_status"] or "—"),
                STATE_LABELS.get(r["gloves_status"], r["gloves_status"] or "—"),
                r["liveness_status"] or "—", "Sí" if r["image_path"] else "No"))
        self.selection_changed()

    def change_page(self, step):
        if step < 0 and self._page == 0 or step > 0 and not self._has_more:
            return
        self._page += step
        self.refresh_events()

    def clear_filters(self):
        self._page = 0
        super().clear_filters()

    def date_preset(self, days):
        for widget, value in ((self.record_from, date.today() - timedelta(days=days)),
                              (self.record_to, date.today())):
            widget.delete(0, tk.END)
            widget.insert(0, value.isoformat())
        self.search_events()

    def toggle_row(self, event):
        if self.events_table.identify_column(event.x) == "#0":
            row = self.events_table.identify_row(event.y)
            if row:
                if row in self.events_table.selection():
                    self.events_table.selection_remove(row)
                else:
                    self.events_table.selection_add(row)
            return "break"

    def selection_changed(self, _event=None):
        selected = set(self.events_table.selection())
        for row in self.events_table.get_children():
            self.events_table.item(row, text="☑" if row in selected else "☐")
        self.records_count.configure(text=f"Página {self._page + 1}  |  {len(self.filtered_events)} registros  |  {len(selected)} seleccionados")

    def sort_events(self, column):
        reverse = getattr(self, "_sort", None) == (column, False)
        by_id = {str(row["id"]): row for row in self.filtered_events}
        rows = sorted(((by_id[row]["timestamp"] if column == "fecha" else self.events_table.set(row, column), row)
                       for row in self.events_table.get_children()), reverse=reverse)
        for i, (_, row) in enumerate(rows):
            self.events_table.move(row, "", i)
        self._sort = (column, reverse)

    def view_evidence(self):
        record = self.selected_event()
        if not record:
            return
        window = tk.Toplevel(self.root)
        window.title(f"Registro #{record['id']} — {record['status']}")
        window.geometry("940x700")
        body = ttk.Frame(window, padding=20)
        body.pack(fill="both", expand=True)
        ttk.Label(body, text=record["person_name"] or "Persona sin identidad asociada",
                  style="Section.TLabel").pack(anchor="w")
        stamp = datetime.fromisoformat(record["timestamp"]).astimezone().strftime("%d/%m/%Y %H:%M:%S")
        ttk.Label(body, text=f'{stamp}   |   {record["event_type"]}   |   {record["status"]}').pack(anchor="w", pady=8)
        path = (ROOT / record["image_path"]).resolve() if record["image_path"] else None
        if path and path.is_relative_to((ROOT / "data" / "evidence").resolve()) and path.is_file():
            try:
                with Image.open(path) as source:
                    picture = source.copy()
                picture.thumbnail((880, 450))
                photo = ImageTk.PhotoImage(picture, master=window)
                label = ttk.Label(body, image=photo)
                label.image = photo
                label.pack(pady=10)
            except OSError:
                ttk.Label(body, text="No se pudo abrir la imagen.").pack()
        else:
            ttk.Label(body, text="Este registro no dispone de imagen accesible.").pack(pady=20)
        details = tk.Text(body, height=5, font=("Consolas", 9), relief="flat", background="#ffffff")
        details.pack(fill="both", expand=True, pady=8)
        try:
            text = json.dumps(json.loads(record["details_json"]), ensure_ascii=False, indent=2)
        except (TypeError, ValueError):
            text = "Sin detalles adicionales"
        details.insert("1.0", text)
        details.configure(state="disabled")
        ttk.Button(body, text="Cerrar", command=window.destroy).pack(anchor="e")

    def export_events(self):
        selected = set(self.events_table.selection())
        rows = [r for r in self.filtered_events if not selected or str(r["id"]) in selected]
        if not rows:
            messagebox.showinfo("Exportar", "No hay registros en esta página.")
            return
        path = filedialog.asksaveasfilename(title="Exportar selección o página actual",
                    defaultextension=".csv", filetypes=(("CSV", "*.csv"),), initialfile="registros.csv")
        if path:
            try:
                columns = tuple(rows[0].keys())
                def safe(value):
                    return "'" + value if isinstance(value, str) and value.startswith(("=", "+", "-", "@")) else value
                with open(path, "w", newline="", encoding="utf-8-sig") as output:
                    writer = csv.writer(output, delimiter=";")
                    writer.writerow(columns)
                    writer.writerows([safe(row[c]) for c in columns] for row in rows)
                messagebox.showinfo("Exportación", f"{len(rows)} registros exportados.")
            except OSError as error:
                messagebox.showerror("Exportación", str(error))

    def save_settings(self):
        try:
            width, height = map(int, self.resolution_setting.get().split("x"))
            changes = {"camera": self.camera_setting.get().strip(), "width": width, "height": height,
                       "fps": int(self.fps_setting.get()), "voice_enabled": self.voice_enabled.get()}
            updated = dict(self.config, **changes)
            save_config(updated)
            self.settings_message.configure(text="Guardado. Reinicie el programa para aplicar la cámara.")
            self.set_voice()
        except (ValueError, OSError) as error:
            messagebox.showerror("Configuración", str(error))

    def set_voice(self):
        if self.voice:
            self.voice.enabled = self.voice_enabled.get()
            # Discard queued messages when muted.
            if not self.voice.enabled:
                while True:
                    try:
                        self.voice.messages.get_nowait()
                    except queue.Empty:
                        break

    def toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        self.root.attributes("-fullscreen", self.fullscreen)

    def leave_fullscreen(self):
        self.fullscreen = False
        self.root.attributes("-fullscreen", False)

    def snapshot(self):
        frame = self.camera.latest.get()
        if frame is None or time.perf_counter() - frame.captured > 1:
            messagebox.showinfo("Captura", "No hay una imagen reciente de la cámara.")
            return
        path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=(("JPEG", "*.jpg"),))
        if path:
            try:
                Image.fromarray(cv2.cvtColor(frame.image, cv2.COLOR_BGR2RGB)).save(path, quality=92)
            except OSError as error:
                messagebox.showerror("Captura", str(error))

    def _diagnostics(self):
        now = time.perf_counter()
        fr, er, frame = self.facial.latest.get(), self.epp.latest.get(), self.camera.latest.get()
        return {"camera_status": self.camera.status, "facial_status": self.facial.status,
                "epp_status": self.epp.status, "voice_status": self.voice.status if self.voice else "No disponible",
                "actual_frame_size": list(frame.image.shape[1::-1]) if frame else None,
                "camera_frame_age_seconds": round(now - frame.captured, 2) if frame else None,
                "facial_ms": round(fr["inference_ms"], 1) if fr else None,
                "epp_ms": round(er["inference_ms"], 1) if er else None,
                "epp_frame_age_seconds": round(now - er["captured"], 2) if er else None,
                "required_epp": list(self.epp.required),
                "liveness_enabled": self.config.get("anti_spoofing_enabled", True)}

    def export_diagnostics(self):
        path = filedialog.asksaveasfilename(defaultextension=".json", filetypes=(("JSON", "*.json"),),
                                           initialfile="diagnostico.json")
        if path:
            try:
                with open(path, "w", encoding="utf-8") as output:
                    json.dump(self._diagnostics(), output, ensure_ascii=False, indent=2)
            except OSError as error:
                messagebox.showerror("Diagnóstico", str(error))

    def _display(self, label, frame, result):
        composed = frame.image.copy()
        if self.overlay.get() and result and time.perf_counter() - result["captured"] < .65:
            mask, image = result.get("annotation_mask"), result.get("preview")
            if mask is not None and image is not None and composed.shape == image.shape:
                composed[mask] = image[mask]
        image = Image.fromarray(cv2.cvtColor(composed, cv2.COLOR_BGR2RGB))
        image.thumbnail((max(160, label.winfo_width()), max(120, label.winfo_height())), Image.Resampling.BILINEAR)
        photo = ImageTk.PhotoImage(image, master=self.root)
        label.configure(image=photo, text="")
        label.image = photo

    def _record_epp(self, result):
        now = time.monotonic()
        people = result["people"]
        if len(people) != 1:
            self._pending_record = None
            self.voice_epp_state["pending"] = None
            self.clear_voice_queue()
            return
        person = people[0]
        if person["id"] != self._presence_id:
            self._presence_id = person["id"]
            self.event_state["epp"].update(key=None, seen=now)
            self.voice_epp_state.update(announced=None, pending=None, since=0., seen=now)
            self._pending_record = None
            self.clear_voice_queue()
        self.event_state["epp"]["seen"] = now
        self.voice_epp_state["seen"] = now
        if person["overall"] not in ("COMPLETO", "INCOMPLETO"):
            self._pending_record = None
            self.voice_epp_state["pending"] = None
            return
        key = (person["id"], person["overall"], tuple(sorted(person["decisions"].items())))
        if self._pending_record != key:
            self._pending_record, self._pending_since = key, now
        elif now - self._pending_since >= .8:
            self._record_once("epp", key, 0, event_type="epp", status=person["overall"],
                             frame=result["preview"], decisions=person["decisions"],
                             camera=self.config["camera"],
                             details={"presence_id": person["id"], "required": list(self.epp.required),
                                      "decisions": person["decisions"]})
        self._update_epp_voice(person["overall"], person["decisions"], now)

    def clear_voice_queue(self):
        if self.voice is not None:
            while True:
                try:
                    self.voice.messages.get_nowait()
                except queue.Empty:
                    break

    def _record_facial(self, result):
        # Require sustained observations for unknown/spoof logs as well as recognition.
        key = (result.get("identity"), result.get("liveness"), result.get("valid"), result.get("candidate"))
        now = time.monotonic()
        if key != self._facial_pending:
            self._facial_pending, self._facial_since = key, now
        if now - self._facial_since >= .7:
            super()._record_facial(result)

    def tick(self):
        if self.stop.is_set():
            return
        now = time.perf_counter()
        selected = self.tabs.index(self.tabs.select())
        frame, fr, er = self.camera.latest.get(), self.facial.latest.get(), self.epp.latest.get()
        camera_fresh = frame is not None and now - frame.captured < 1.
        for state in self.event_state.values():
            if time.monotonic() - state["seen"] > 3.:
                state["key"] = None
        self.system_status.configure(text=f"Cámara: {self.camera.status}    Facial: {self.facial.status}    EPP: {self.epp.status}")
        if camera_fresh and frame.sequence != self.last_camera_sequence:
            self.last_camera_sequence = frame.sequence
            self._camera_times.append(now)
            target, result = (self.facial_video, fr) if selected == 0 else (
                (self.epp_video, er) if selected == 1 else (self.enroll_video, fr))
            if selected != 2 or self.admin_tabs.index(self.admin_tabs.select()) == 0:
                self._display(target, frame, result)
        if len(self._camera_times) > 1 and camera_fresh:
            fps = (len(self._camera_times) - 1) / max(.01, self._camera_times[-1] - self._camera_times[0])
            self.performance.configure(text=f"Vista {fps:.0f} FPS")
        else:
            self.performance.configure(text="Cámara disponible" if camera_fresh else "Sin imagen reciente")
        if selected in (0, 2):
            fresh = camera_fresh and fr and now - fr["captured"] < 1.
            if fresh:
                if selected == 0 and self.last_sequences["facial"] != fr["sequence"]:
                    self.last_sequences["facial"] = fr["sequence"]
                    self._record_facial(fr)
                live = fr.get("liveness", "verificando")
                if selected == 0:
                    title = ("Presentación rechazada" if live == "falso" else
                             fr["identity"] if fr.get("identity") and live == "real" else
                             "Persona no registrada" if fr["valid"] and live == "real" and not fr.get("candidate") else
                             "Verificando identidad" if fr["valid"] else "Esperando rostro")
                    self.facial_result.configure(text=title)
                    self.facial_live.configure(text={"real": "Prueba de vida confirmada", "falso": "Posible foto o pantalla",
                        "sin rostro": "Sin rostro válido"}.get(live, "Prueba de vida pendiente"))
                    self.facial_details.configure(text=f'{fr["quality"]}\n\nAnálisis: {fr["inference_ms"]:.0f} ms')
                else:
                    self.enroll_status.configure(text=self.facial.status + " — " + fr["quality"])
                    if self.was_enrolling and not self.facial.enrolling:
                        self.refresh_people()
                    self.was_enrolling = self.facial.enrolling
            elif selected == 0:
                self.facial_result.configure(text="Sin análisis reciente")
                self.facial_live.configure(text="Verificación pendiente")
                self.facial_details.configure(text="Compruebe la cámara y el estado del servicio.")
        if selected == 1:
            self._refresh_required_rows()
            fresh = camera_fresh and er and now - er["captured"] < 1.
            if fresh:
                if self.last_sequences["epp"] != er["sequence"]:
                    self.last_sequences["epp"] = er["sequence"]
                    self._record_epp(er)
                person = er["people"][0] if len(er["people"]) == 1 else None
                titles = {"COMPLETO": "Equipo completo", "INCOMPLETO": "Equipo incompleto", "VERIFICANDO": "Verificando equipo"}
                self.epp_result.configure(text=titles.get(person["overall"], "Verificando") if person else "Esperando persona")
                for name, label in self.item_labels.items():
                    state = person["decisions"].get(name, "verificando") if person else "verificando"
                    label.configure(text=STATE_LABELS.get(state, state) if name in self.epp.required else "No requerido")
                self.epp_details.configure(text=f'{er.get("guidance", "")}\n\nAnálisis: {er["inference_ms"]:.0f} ms')
            else:
                self.epp_result.configure(text="Sin análisis reciente")
                self.epp_details.configure(text="Verificación pendiente. Compruebe la cámara.")
                self._pending_record = None
                self.voice_epp_state["pending"] = None
                for label in self.item_labels.values():
                    label.configure(text="Por confirmar")
        if selected == 2 and self.admin_tabs.index(self.admin_tabs.select()) == 3:
            if now - getattr(self, "_diagnostic_time", 0) >= 1:
                self._diagnostic_time = now
                self.diagnostic_text.configure(state="normal")
                self.diagnostic_text.delete("1.0", "end")
                self.diagnostic_text.insert("1.0", json.dumps(self._diagnostics(), ensure_ascii=False, indent=2))
                self.diagnostic_text.configure(state="disabled")
        self.root.after(30, self.tick)
