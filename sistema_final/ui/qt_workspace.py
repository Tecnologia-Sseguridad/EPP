"""Modern PyQt6 desktop interface for the existing local vision services."""
import csv
import html
import json
import queue
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QTimer, QPoint, QSize
from PyQt6.QtGui import QAction, QColor, QFont, QIcon, QImage, QPainter, QPainterPath, QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QApplication, QCheckBox, QComboBox, QDialog, QFileDialog,
    QFormLayout, QFrame, QGridLayout, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSpinBox,
    QStackedWidget, QTabWidget, QTableWidget, QTableWidgetItem, QTextEdit,
    QMenu, QToolButton, QVBoxLayout, QWidget,
)

from .icons import render_icon
from sistema_final.core.configuration import ROOT, save_config
from sistema_final.core.database import (
    delete_events, delete_person, get_required_epp, list_enrollment_images, list_people, query_events,
    record_event, set_required_epp,
)

STATE_LABELS = {"si": "Confirmado", "no": "Falta", "verificando": "Analizando",
                "no visible": "Sin evidencia", "fuera de encuadre": "Revisar encuadre"}
EPP_LABELS = {"casco": "Casco", "chaleco": "Chaleco", "guantes": "Guantes",
              "antiparras": "Antiparras", "botas": "Botas"}


def icon(name, color="#5f6368", size=21):
    return QIcon(QPixmap.fromImage(ImageQt(render_icon(name, size, color))))


def rounded_thumbnail(path, size=46):
    source=QPixmap(str(path)).scaled(size,size,Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation)
    target=QPixmap(size,size);target.fill(Qt.GlobalColor.transparent);painter=QPainter(target);painter.setRenderHint(QPainter.RenderHint.Antialiasing)
    shape=QPainterPath();shape.addRoundedRect(0,0,size,size,7,7);painter.setClipPath(shape);painter.drawPixmap(0,0,source);painter.end();return target


class TitleBar(QFrame):
    def __init__(self, window):
        super().__init__()
        self.window, self.origin = window, None
        self.setObjectName("titlebar")
        self.setFixedHeight(46)
        row = QHBoxLayout(self); row.setContentsMargins(18, 0, 6, 0); row.setSpacing(2)
        name = QLabel("CONTROL Y VERIFICACIÓN"); name.setObjectName("windowTitle")
        row.addWidget(name); row.addStretch()
        for text, slot, kind in (("—", window.showMinimized, "windowButton"),
                                 ("□", window.toggle_maximize, "windowButton"),
                                 ("×", window.close, "closeButton")):
            button = QPushButton(text); button.setObjectName(kind); button.setFixedSize(48, 36)
            button.clicked.connect(slot); row.addWidget(button)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.origin = event.globalPosition().toPoint() - self.window.frameGeometry().topLeft()

    def mouseMoveEvent(self, event):
        if self.origin is not None and event.buttons() & Qt.MouseButton.LeftButton and not self.window.isMaximized():
            self.window.move(event.globalPosition().toPoint() - self.origin)

    def mouseReleaseEvent(self, event): self.origin = None
    def mouseDoubleClickEvent(self, event): self.window.toggle_maximize()


class VideoPanel(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("videoCard")
        layout = QVBoxLayout(self); layout.setContentsMargins(8, 8, 8, 8)
        self.label = QLabel("Esperando cámara"); self.label.setObjectName("video")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter); self.label.setMinimumSize(520, 340)
        layout.addWidget(self.label)

    def show_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], rgb.strides[0], QImage.Format.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(image).scaled(self.label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                                 Qt.TransformationMode.SmoothTransformation)
        self.label.setPixmap(pixmap)


class ResultPanel(QFrame):
    def __init__(self, title="Esperando análisis"):
        super().__init__(); self.setObjectName("card"); self.setFixedWidth(310)
        self.layout = QVBoxLayout(self); self.layout.setContentsMargins(24, 24, 24, 24); self.layout.setSpacing(12)
        eyebrow = QLabel("RESULTADO"); eyebrow.setObjectName("eyebrow"); self.layout.addWidget(eyebrow)
        self.title = QLabel(title); self.title.setObjectName("resultTitle"); self.title.setWordWrap(True)
        self.layout.addWidget(self.title)
        line = QFrame(); line.setFrameShape(QFrame.Shape.HLine); line.setObjectName("line"); self.layout.addWidget(line)
        self.body = QVBoxLayout(); self.body.setSpacing(10); self.layout.addLayout(self.body); self.layout.addStretch()
        self.note = QLabel(); self.note.setWordWrap(True); self.note.setObjectName("muted"); self.layout.addWidget(self.note)


class QtMainWindow(QMainWindow):
    def __init__(self, camera, facial, epp, stop, config, voice=None, smoke_test=False):
        super().__init__()
        self.camera, self.facial, self.epp, self.stop = camera, facial, epp, stop
        self.config, self.voice = config, voice
        self.overlay = True; self.last_camera_sequence = -1; self.last_facial_sequence = -1
        self.last_epp_sequence = -1; self.filtered_events = []; self.page = 0; self.page_size=8;self.total_events=0;self.has_more = False
        self.event_keys = {"facial": None, "epp": None}; self.event_seen = {"facial": 0., "epp": 0.}
        self.pending_epp = None; self.pending_since = 0.; self.presence_id = None
        self.voice_announced = None; self.voice_pending = None; self.voice_since = 0.
        self.enrolling_before = False; self.drag = None
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.Window)
        self.setMinimumSize(1080, 700); self.resize(1380, 880)
        self.setWindowTitle("SSeguridad | Control y verificación")
        self._build(); self._style(); self.select_page(0)
        self.timer = QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(30)
        if smoke_test: QTimer.singleShot(2500, self.close)

    def _build(self):
        root = QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        outer = QVBoxLayout(root); outer.setContentsMargins(1, 1, 1, 1); outer.setSpacing(0)
        outer.addWidget(TitleBar(self))
        body = QHBoxLayout(); body.setSpacing(0); outer.addLayout(body, 1)
        rail = QFrame(); rail.setObjectName("sidebar"); rail.setFixedWidth(224)
        side = QVBoxLayout(rail); side.setContentsMargins(18, 28, 18, 20); side.setSpacing(8)
        brand = QLabel(); brand.setObjectName("brandLogo"); brand.setAlignment(Qt.AlignmentFlag.AlignLeft|Qt.AlignmentFlag.AlignVCenter)
        brand.setPixmap(QPixmap(str(ROOT/"logo.png")).scaled(178,78,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));side.addWidget(brand)
        side.addSpacing(22); self.nav = []
        for index, (text, name) in enumerate((("Reconocimiento", "face"), ("Protección EPP", "shield"),
                                               ("Administración", "users"))):
            button = QPushButton(text); button.setObjectName("nav"); button.setIcon(icon(name, "#c6c9ce"));
            button.setCheckable(True); button.clicked.connect(lambda _, i=index: self.select_page(i))
            side.addWidget(button); self.nav.append(button)
        self.admin_nav=[]
        for tab,(text,name) in enumerate((("Personas","users"),("Registros","records"),("Configuración","settings"),("Diagnóstico","info"))):
            button=QPushButton(text);button.setObjectName("subnav");button.setIcon(icon(name,"#b9c4d1"));button.setCheckable(True);button.clicked.connect(lambda _,i=tab:self.select_admin_tab(i));side.addWidget(button);self.admin_nav.append(button)
        side.addStretch()
        full = QPushButton("Pantalla completa"); full.setObjectName("sideAction"); full.clicked.connect(self.toggle_fullscreen)
        side.addWidget(QLabel("F11  Pantalla completa\nEsc  Volver"), alignment=Qt.AlignmentFlag.AlignLeft)
        side.addWidget(full); body.addWidget(rail)
        content = QFrame(); content.setObjectName("content")
        column = QVBoxLayout(content); column.setContentsMargins(28, 24, 28, 14); column.setSpacing(10)
        self.stack = QStackedWidget(); column.addWidget(self.stack, 1)
        body.addWidget(content, 1)
        self.stack.addWidget(self._facial_page()); self.stack.addWidget(self._epp_page()); self.stack.addWidget(self._admin_page())

    def _heading(self, title, subtitle):
        box = QWidget(); col = QVBoxLayout(box); col.setContentsMargins(0, 0, 0, 14); col.setSpacing(4)
        head = QLabel(title); head.setObjectName("heading"); col.addWidget(head)
        sub = QLabel(subtitle); sub.setObjectName("muted"); col.addWidget(sub); return box

    def _recognition_page(self, title, subtitle):
        page = QWidget(); col = QVBoxLayout(page); col.setContentsMargins(0, 0, 0, 0); col.setSpacing(12)
        col.addWidget(self._heading(title, subtitle))
        tools = QHBoxLayout(); overlay = QPushButton("Encuadre visible"); overlay.setObjectName("secondary")
        overlay.clicked.connect(lambda: self.toggle_overlay(overlay)); tools.addWidget(overlay); tools.addStretch()
        capture = QPushButton("Guardar captura"); capture.setIcon(icon("camera")); capture.clicked.connect(self.snapshot)
        tools.addWidget(capture); col.addLayout(tools)
        row = QHBoxLayout(); row.setSpacing(18); video = VideoPanel(); result = ResultPanel()
        row.addWidget(video, 1); row.addWidget(result); col.addLayout(row, 1); return page, video, result

    def _facial_page(self):
        page, self.facial_video, self.facial_result = self._recognition_page(
            "Reconocimiento facial", "Identidad y prueba de vida en procesamiento local")
        self.facial_live = QLabel("Prueba de vida pendiente"); self.facial_live.setObjectName("detailStrong")
        self.facial_detail = QLabel(); self.facial_detail.setWordWrap(True); self.facial_detail.setObjectName("muted")
        self.facial_result.body.addWidget(self.facial_live); self.facial_result.body.addWidget(self.facial_detail)
        self.facial_result.note.setText("Mire a la cámara y mantenga el rostro visible.")
        return page

    def _epp_page(self):
        page, self.epp_video, self.epp_result = self._recognition_page(
            "Protección personal", "Verificación del equipo requerido para una persona")
        self.epp_rows = {}
        for name in EPP_LABELS:
            row = QFrame(); row.setObjectName("resultRow"); line = QHBoxLayout(row); line.setContentsMargins(10, 10, 10, 10)
            symbol = QLabel(); symbol.setPixmap(icon({"casco":"helmet","chaleco":"vest","guantes":"glove"}.get(name,"shield")).pixmap(24,24))
            label = QLabel(EPP_LABELS[name]); label.setObjectName("detailStrong"); state = QLabel("Analizando"); state.setObjectName("muted")
            line.addWidget(symbol); line.addWidget(label); line.addStretch(); line.addWidget(state)
            self.epp_result.body.addWidget(row); self.epp_rows[name] = (row, state)
        self.epp_result.note.setText("Mantenga visibles la cabeza, el torso y ambas manos.")
        return page

    def _admin_page(self):
        page = QWidget(); col = QVBoxLayout(page); col.setContentsMargins(0,0,0,0)
        heading=QWidget();headrow=QHBoxLayout(heading);headrow.setContentsMargins(0,0,0,14);self.admin_heading_icon=QLabel();self.admin_heading_icon.setObjectName("headingIcon");self.admin_heading_icon.setAlignment(Qt.AlignmentFlag.AlignCenter);self.admin_heading_icon.setFixedSize(58,58);self.admin_heading_icon.setPixmap(icon("users","#2878ef",28).pixmap(28,28));headrow.addWidget(self.admin_heading_icon);titles=QVBoxLayout();titles.setSpacing(4);self.admin_title=QLabel("Personas");self.admin_title.setObjectName("heading");self.admin_subtitle=QLabel("Enrolamiento y gestión de identidades");self.admin_subtitle.setObjectName("muted");titles.addWidget(self.admin_title);titles.addWidget(self.admin_subtitle);headrow.addLayout(titles);headrow.addStretch();clock=QFrame();clock.setObjectName("dateCard");clockrow=QHBoxLayout(clock);clockrow.setContentsMargins(16,9,16,9);self.admin_date=QLabel();self.admin_date.setObjectName("muted");self.admin_time=QLabel();self.admin_time.setObjectName("adminTime");clockrow.addWidget(self.admin_date);clockrow.addSpacing(14);clockrow.addWidget(self.admin_time);headrow.addWidget(clock);col.addWidget(heading)
        self.admin_tabs = QTabWidget(); self.admin_tabs.setDocumentMode(True); col.addWidget(self.admin_tabs, 1)
        self.admin_tabs.addTab(self._people_tab(), icon("users"), "Personas")
        self.admin_tabs.addTab(self._records_tab(), icon("records"), "Registros")
        self.admin_tabs.addTab(self._settings_tab(), icon("settings"), "Configuración")
        self.admin_tabs.addTab(self._diagnostics_tab(), icon("info"), "Diagnóstico")
        self.admin_tabs.tabBar().hide();self.admin_tabs.currentChanged.connect(self.admin_changed); return page

    def _people_tab(self):
        page=QWidget();row=QHBoxLayout(page);row.setContentsMargins(4,12,4,4);row.setSpacing(16)
        left=QFrame();left.setObjectName("card");left.setFixedWidth(300);lc=QVBoxLayout(left);lc.setContentsMargins(16,16,16,16);lc.setSpacing(11)
        title=QLabel("Personas");title.setObjectName("peopleTitle");lc.addWidget(title)
        self.people_search=QLineEdit();self.people_search.setPlaceholderText("Buscar por nombre...");self.people_search.textChanged.connect(self.refresh_people);lc.addWidget(self.people_search)
        new_person=QPushButton("＋  Nueva persona");new_person.setObjectName("primaryBlue");new_person.clicked.connect(self.new_person);lc.addWidget(new_person)
        self.people_list=QListWidget();self.people_list.setObjectName("peopleList");self.people_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection);self.people_list.currentItemChanged.connect(self.person_selected);lc.addWidget(self.people_list,1)
        self.people_count=QLabel();self.people_count.setObjectName("muted");lc.addWidget(self.people_count);row.addWidget(left)

        right=QWidget();rc=QVBoxLayout(right);rc.setContentsMargins(0,0,0,0);rc.setSpacing(14)
        profile=QFrame();profile.setObjectName("profileCard");ph=QHBoxLayout(profile);ph.setContentsMargins(20,16,20,16);avatar=QLabel("?");avatar.setObjectName("profileAvatar");avatar.setAlignment(Qt.AlignmentFlag.AlignCenter);avatar.setFixedSize(58,58);self.profile_avatar=avatar;ph.addWidget(avatar);names=QVBoxLayout();self.profile_name=QLabel("Nueva persona");self.profile_name.setObjectName("profileName");self.profile_meta=QLabel("Complete el nombre y capture las muestras faciales");self.profile_meta.setObjectName("muted");names.addWidget(self.profile_name);names.addWidget(self.profile_meta);ph.addLayout(names);ph.addStretch();self.profile_state=QLabel("●  Pendiente");self.profile_state.setObjectName("profilePending");ph.addWidget(self.profile_state);remove=QToolButton();remove.setObjectName("moreButton");remove.setText("⋮");remove.clicked.connect(self.profile_menu);ph.addWidget(remove);rc.addWidget(profile)

        section=QFrame();section.setObjectName("profileCard");sc=QVBoxLayout(section);sc.setContentsMargins(16,14,16,14);tabs=QHBoxLayout();active=QLabel("▣  Enrolamiento");active.setObjectName("enrollTab");tabs.addWidget(active);tabs.addSpacing(30);tabs.addWidget(QLabel("Información",objectName="muted"));tabs.addSpacing(30);tabs.addWidget(QLabel("Registros",objectName="muted"));tabs.addStretch();sc.addLayout(tabs)
        workspace=QHBoxLayout();workspace.setSpacing(14);camera_card=QFrame();camera_card.setObjectName("innerCard");cc=QVBoxLayout(camera_card);camera_title=QLabel("Cámara en vivo");camera_title.setObjectName("section");cc.addWidget(camera_title);self.enroll_video=VideoPanel();self.enroll_video.label.setMinimumSize(360,260);cc.addWidget(self.enroll_video,1);self.enroll_status=QLabel("Mire a la cámara y mantenga el rostro visible.");self.enroll_status.setObjectName("enrollHint");self.enroll_status.setWordWrap(True);cc.addWidget(self.enroll_status);workspace.addWidget(camera_card,3)
        samples=QFrame();samples.setObjectName("innerCard");gc=QVBoxLayout(samples);sample_title=QLabel("Muestras de enrolamiento");sample_title.setObjectName("section");gc.addWidget(sample_title);grid=QGridLayout();grid.setSpacing(8);self.enroll_samples=[]
        for index in range(8):slot=QLabel(str(index+1));slot.setObjectName("sampleSlot");slot.setAlignment(Qt.AlignmentFlag.AlignCenter);slot.setMinimumSize(104,76);grid.addWidget(slot,index//2,index%2);self.enroll_samples.append(slot)
        gc.addLayout(grid,1);capture=QPushButton("Capturar / reemplazar");capture.setIcon(icon("camera","#ffffff"));capture.clicked.connect(self.start_enrollment);gc.addWidget(capture);workspace.addWidget(samples,2);sc.addLayout(workspace,1);rc.addWidget(section,1)

        data=QFrame();data.setObjectName("profileCard");dc=QVBoxLayout(data);dc.setContentsMargins(18,15,18,15);data_title=QLabel("Datos de la persona");data_title.setObjectName("section");dc.addWidget(data_title);form=QHBoxLayout();field=QVBoxLayout();field.addWidget(QLabel("Nombre completo",objectName="muted"));self.enroll_name=QLineEdit();self.enroll_name.setPlaceholderText("Nombre o identificador");self.enroll_name.textChanged.connect(self.preview_person_name);field.addWidget(self.enroll_name);form.addLayout(field,2);form.addStretch(1);dc.addLayout(form);buttons=QHBoxLayout();enroll=QPushButton("Guardar y enrolar");enroll.clicked.connect(self.start_enrollment);cancel=QPushButton("Cancelar");cancel.setObjectName("secondary");cancel.clicked.connect(lambda:self.facial.command("cancel"));buttons.addWidget(enroll);buttons.addWidget(cancel);buttons.addStretch();dc.addLayout(buttons);rc.addWidget(data);row.addWidget(right,1)
        self.enroll_preview_count=0;self.refresh_people();return page

    def _records_tab(self):
        page=QWidget(); col=QVBoxLayout(page); col.setContentsMargins(4,18,4,4); col.setSpacing(12)
        stats=QHBoxLayout();stats.setSpacing(12);self.record_stats=[]
        for label,color,icon_name in (("Registros hoy","#2878ef","users"),("Reconocimientos","#19a765","check"),("Accesos rechazados","#dc4453",None),("EPP incompletos","#e7a323","shield")):
            card=QFrame();card.setObjectName("statCard");box=QHBoxLayout(card);box.setContentsMargins(16,13,16,13);symbol=QLabel();symbol.setObjectName("statIcon");symbol.setAlignment(Qt.AlignmentFlag.AlignCenter);symbol.setFixedSize(45,45)
            if icon_name:symbol.setPixmap(icon(icon_name,"#ffffff",23).pixmap(23,23))
            else:symbol.setText("×");symbol.setStyleSheet(f"background:{color};border-radius:22px;color:white;font:700 27px 'Segoe UI'")
            if icon_name:symbol.setStyleSheet(f"background:{color};border-radius:22px")
            texts=QVBoxLayout();value=QLabel("0");value.setObjectName("statValue");caption=QLabel(label);caption.setObjectName("muted");texts.addWidget(value);texts.addWidget(caption);box.addWidget(symbol);box.addLayout(texts);box.addStretch();stats.addWidget(card,1);self.record_stats.append(value)
        col.addLayout(stats)
        module_bar=QFrame();module_bar.setObjectName("moduleBar");modules=QHBoxLayout(module_bar);modules.setContentsMargins(8,8,8,8);modules.setSpacing(6);modules.addWidget(QLabel("Vista de registros",objectName="section"));modules.addStretch();self.module_buttons=[]
        for label,value in (("Todos","Todos"),("Reconocimiento facial","facial"),("Protección EPP","epp")):
            button=QPushButton(label);button.setObjectName("moduleChoice");button.setCheckable(True);button.setProperty("module",value);button.clicked.connect(lambda checked,b=button:self.select_record_module(b));modules.addWidget(button);self.module_buttons.append(button)
        self.module_buttons[0].setChecked(True);col.addWidget(module_bar)
        filters=QHBoxLayout(); self.record_person=QLineEdit(); self.record_person.setPlaceholderText("Buscar por persona")
        self.record_type=QComboBox(); self.record_type.addItems(("Todos","facial","epp"));self.record_type.hide(); self.record_status=QComboBox()
        self.record_from=QLineEdit(); self.record_from.setPlaceholderText("Desde AAAA-MM-DD"); self.record_to=QLineEdit(); self.record_to.setPlaceholderText("Hasta AAAA-MM-DD")
        for w in (self.record_person,self.record_from,self.record_to,self.record_status): filters.addWidget(w)
        search=QPushButton("Buscar"); search.setIcon(icon("search","#ffffff"));search.clicked.connect(self.search_events); filters.addWidget(search);clear=QPushButton("Limpiar");clear.setObjectName("secondary");clear.clicked.connect(self.clear_record_filters);filters.addWidget(clear);col.addLayout(filters)
        self.events=QTableWidget(0,11);self.events.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows);self.events.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection);self.events.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers);self.events.verticalHeader().setVisible(False);self.events.verticalHeader().setDefaultSectionSize(62);self.events.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch);self.events.horizontalHeader().sectionClicked.connect(self.header_clicked);self.events.doubleClicked.connect(self.view_evidence);col.addWidget(self.events,1)
        pagination=QHBoxLayout();self.records_count=QLabel();self.records_count.setObjectName("muted");pagination.addWidget(self.records_count);pagination.addStretch();previous=QPushButton("‹");previous.setObjectName("pageButton");previous.clicked.connect(lambda:self.change_page(-1));pagination.addWidget(previous);self.page_buttons=[]
        for number in range(1,6):button=QPushButton(str(number));button.setObjectName("pageButton");button.setCheckable(True);button.clicked.connect(lambda _,n=number:self.go_to_page(n-1));pagination.addWidget(button);self.page_buttons.append(button)
        following=QPushButton("›");following.setObjectName("pageButton");following.clicked.connect(lambda:self.change_page(1));pagination.addWidget(following);self.per_page=QComboBox();self.per_page.addItems(("8 por página","15 por página","30 por página"));self.per_page.currentIndexChanged.connect(self.change_page_size);pagination.addWidget(self.per_page);col.addLayout(pagination)
        bar=QHBoxLayout();export=QPushButton("Exportar CSV");export.setIcon(icon("export","#ffffff"));export.clicked.connect(self.export_events);report=QPushButton("Exportar reporte");report.setObjectName("secondary");report.clicked.connect(self.export_report);delete=QPushButton("Eliminar selección");delete.setObjectName("danger");delete.setIcon(icon("delete","#c33140"));delete.clicked.connect(self.delete_selected)
        for w in (export,report,delete):bar.addWidget(w)
        bar.addStretch();col.addLayout(bar);self.update_record_view();return page

    def _settings_tab(self):
        scroll=QScrollArea(); scroll.setWidgetResizable(True); page=QWidget(); scroll.setWidget(page); col=QVBoxLayout(page); col.setContentsMargins(18,22,18,22); col.setSpacing(14)
        heading=QLabel("Equipo requerido"); heading.setObjectName("section"); col.addWidget(heading); required=set(get_required_epp()); self.requirement_checks={}
        checks=QHBoxLayout()
        for name,label in EPP_LABELS.items(): check=QCheckBox(label); check.setChecked(name in required); checks.addWidget(check); self.requirement_checks[name]=check
        checks.addStretch(); col.addLayout(checks); save=QPushButton("Guardar requisitos"); save.clicked.connect(self.save_requirements); col.addWidget(save,0,Qt.AlignmentFlag.AlignLeft)
        line=QFrame(); line.setFrameShape(QFrame.Shape.HLine); col.addWidget(line); camera=QLabel("Cámara y presentación"); camera.setObjectName("section"); col.addWidget(camera)
        form=QFormLayout(); self.camera_edit=QLineEdit(str(self.config["camera"])); self.resolution=QComboBox(); self.resolution.addItems(("640x480","1280x720","1920x1080")); self.resolution.setCurrentText(f'{self.config["width"]}x{self.config["height"]}'); self.fps=QSpinBox(); self.fps.setRange(5,60); self.fps.setValue(self.config["fps"]); self.voice_check=QCheckBox("Avisos de voz"); self.voice_check.setChecked(self.config.get("voice_enabled",True)); form.addRow("Cámara",self.camera_edit); form.addRow("Resolución",self.resolution); form.addRow("FPS solicitados",self.fps); form.addRow("",self.voice_check); col.addLayout(form)
        save_config_button=QPushButton("Guardar configuración"); save_config_button.clicked.connect(self.save_settings); col.addWidget(save_config_button,0,Qt.AlignmentFlag.AlignLeft); self.settings_note=QLabel("Los cambios de cámara se aplican al reiniciar."); self.settings_note.setObjectName("muted"); col.addWidget(self.settings_note); col.addStretch(); return scroll

    def _diagnostics_tab(self):
        page=QWidget(); col=QVBoxLayout(page); col.setContentsMargins(8,18,8,8); self.diagnostics=QTextEdit(); self.diagnostics.setReadOnly(True); self.diagnostics.setObjectName("diagnostics"); col.addWidget(self.diagnostics); return page

    def _style(self):
        self.setStyleSheet("""
        * { font-family: 'Segoe UI Variable Text','Segoe UI'; font-size: 13px; color: #172033; }
        #root { background: #15171b; } #titlebar, #sidebar { background: #17191d; }
        #windowTitle { color: #dfe1e5; font: 600 11px 'Segoe UI'; letter-spacing: 1px; }
        #windowButton, #closeButton { background: transparent; color: #e8eaed; border: 0; font-size: 16px; }
        #windowButton:hover { background: #2c3036; } #closeButton:hover { background: #b3261e; }
        #content { background: #f4f7fb; font-size:14px; } #brandLogo { background: transparent; }
        #sideMuted, #sidebar > QLabel { color: #9ca1aa; } QPushButton#nav { color: #dfe1e5; text-align: left; padding: 14px 16px; border: 0; border-radius: 8px; background: transparent; }
        QPushButton#nav:hover { background: #202a36; } QPushButton#nav:checked { background: #1d5fca; color: white; }
        QPushButton#subnav { color:#b9c4d1;text-align:left;padding:11px 14px 11px 28px;border:0;border-radius:8px;background:transparent; }
        QPushButton#subnav:hover { background:#202a36;color:white; } QPushButton#subnav:checked { background:#253447;color:white;font-weight:600; }
        QPushButton#sideAction { color: white; background: #292d33; border: 0; border-radius: 8px; padding: 12px; }
        QPushButton#primaryBlue { background:#1769e8;color:white;padding:12px;border-radius:8px; } QPushButton#primaryBlue:hover { background:#0e58ca; }
        #heading { font: 700 30px 'Segoe UI Variable Display','Segoe UI'; color: #172033; } #headingIcon { background:#e7f0ff;border-radius:14px; } #section { font: 600 16px 'Segoe UI'; }
        #muted { color: #6c7077; } #eyebrow { color: #656970; font: 600 11px 'Segoe UI'; letter-spacing: 1px; }
        #resultTitle { font: 700 21px 'Segoe UI'; padding: 14px; background: #eef0f2; border-radius: 8px; }
        #detailStrong { font: 600 13px 'Segoe UI'; } #line { color: #e3e5e8; }
        #card, #videoCard, #moduleBar, #statCard { background: white; border: 1px solid #e1e7ef; border-radius: 12px; }
        #statValue { color:#152033;font:700 25px 'Segoe UI'; }
        #dateCard { background:white;border:1px solid #e1e7ef;border-radius:12px; } #adminTime { color:#152033;font:700 20px 'Segoe UI'; }
        #profileCard, #innerCard { background:white;border:1px solid #dfe6ee;border-radius:12px; } #innerCard { background:#fbfcfe; }
        #peopleTitle { color:#172033;font:700 21px 'Segoe UI'; } #profileName { color:#101827;font:700 22px 'Segoe UI'; }
        #profileAvatar { background:#e6eef9;color:#2868be;border-radius:29px;font:700 17px 'Segoe UI'; }
        #profileActive { color:#138552;background:#e5f6ed;border-radius:8px;padding:7px 11px;font-weight:600; } #profilePending { color:#9a6710;background:#fff4dc;border-radius:8px;padding:7px 11px;font-weight:600; }
        #enrollTab { color:#1764d4;font-weight:700;padding:9px 14px;border-bottom:3px solid #2878ef; }
        #enrollHint { color:#1764d4;background:#e9f2ff;border-radius:8px;padding:11px;font-weight:600; }
        #sampleSlot { background:#eef3f8;color:#8b98a8;border:1px dashed #bdc8d5;border-radius:9px;font-weight:700; }
        #peopleList { border:0;background:transparent; } #peopleList::item { padding:10px;border-radius:9px;border-bottom:1px solid #edf1f5; } #peopleList::item:selected { background:#e5efff;color:#145abb;border-left:3px solid #2878ef; }
        #resultRow { background: #f7f8f9; border-radius: 8px; } #video { background: #101114; color: #9ca1aa; border-radius: 8px; }
        QPushButton { background: #202328; color: white; border: 0; border-radius: 7px; padding: 10px 16px; font-weight: 600; }
        QPushButton:hover { background: #34383f; } QPushButton#secondary { background: white; color: #25272b; border: 1px solid #d7dade; }
        QPushButton#secondary:hover { background: #f0f1f3; }
        QPushButton#danger { background:#fff1f2;color:#c33140;border:1px solid #f2c8cd; } QPushButton#danger:hover { background:#ffe4e7; }
        QPushButton#moduleChoice { background: transparent; color: #526071; border: 1px solid transparent; padding: 9px 15px; }
        QPushButton#moduleChoice:hover { background: #eef4fd; color: #174e9d; }
        QPushButton#moduleChoice:checked { background: #e7f0ff; color: #155ac5; border: 1px solid #bed4f7; }
        QLineEdit, QComboBox, QSpinBox { background: white; border: 1px solid #d5d8dc; border-radius: 7px; padding: 9px; selection-background-color: #4a5058; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #60656d; }
        QListWidget, QTableWidget, QTextEdit { background: white; border: 1px solid #e0e6ee; border-radius: 10px; outline: 0; gridline-color: transparent; }
        QListWidget::item { padding: 10px; border-radius: 5px; } QListWidget::item:selected { background: #e5e7ea; color: #1b1d20; }
        QHeaderView::section { background: #f3f6fa; border: 0; border-bottom: 1px solid #dfe6ef; padding: 11px 7px; font-size:13px;font-weight: 600; }
        QTableWidget::item { padding: 9px; border-bottom: 1px solid #edf1f5; } QTableWidget::item:selected { background: #e4efff; color: #102a4c; }
        QLabel#stateSymbol { color:white;font-weight:700;border-radius:10px; } QLabel#stateSymbol[tone="ok"] { background:#19a765; } QLabel#stateSymbol[tone="bad"] { background:#dc4453; } QLabel#stateSymbol[tone="neutral"] { background:#a4afbc; }
        QLabel#stateText[tone="ok"] { color:#11854f;font-weight:600; } QLabel#stateText[tone="bad"] { color:#c33140;font-weight:600; } QLabel#stateText[tone="neutral"] { color:#7b8795; }
        QLabel#moduleBadge { padding:7px 9px;border-radius:8px;font-weight:600; } QLabel#moduleBadge[module="facial"] { background:#e8f1ff;color:#1764d4; } QLabel#moduleBadge[module="epp"] { background:#eeeaff;color:#6747d8; }
        QLabel#thumbnail { background:#eef2f6;border:1px solid #d6dee8;border-radius:8px; }
        QToolButton#moreButton { background:transparent;border:0;border-radius:6px;font:700 22px 'Segoe UI';padding:4px; } QToolButton#moreButton:hover { background:#e8eef6; }
        QPushButton#pageButton { min-width:34px;max-width:34px;padding:8px;background:white;color:#405066;border:1px solid #dce4ee; } QPushButton#pageButton:checked { background:#2878ef;color:white;border-color:#2878ef; }
        QCheckBox::indicator, QTableWidget::indicator { width:17px;height:17px;border:1px solid #aeb9c7;border-radius:4px;background:white; } QTableWidget::indicator:checked { background:#2878ef;border-color:#2878ef; }
        QTabWidget::pane { background: white; border: 1px solid #e0e2e5; border-radius: 9px; top: -1px; }
        QTabBar::tab { background: transparent; padding: 12px 20px; margin-right: 4px; border-bottom: 2px solid transparent; }
        QTabBar::tab:selected { color: #111316; border-bottom: 2px solid #33373d; font-weight: 600; }
        QCheckBox { spacing: 8px; } QCheckBox::indicator { width: 18px; height: 18px; border: 1px solid #aeb2b8; border-radius: 5px; background: white; }
        QCheckBox::indicator:checked { background: #34383f; border-color: #34383f; }
        QScrollArea { border: 0; background: transparent; } QScrollBar:vertical { width: 10px; background: transparent; } QScrollBar::handle:vertical { background: #c5c8cd; border-radius: 5px; min-height: 35px; }
        """)

    def select_page(self, index):
        self.stack.setCurrentIndex(index)
        for i,b in enumerate(self.nav): b.setChecked(i==index)
        for i,b in enumerate(self.admin_nav):b.setVisible(index==2);b.setChecked(index==2 and i==self.admin_tabs.currentIndex())
        facial_active=index==0 or (index==2 and self.admin_tabs.currentIndex()==0)
        self.facial.activate() if facial_active else self.facial.deactivate()
        self.epp.activate() if index==1 else self.epp.deactivate()

    def admin_changed(self,index):
        titles=(("Personas","Enrolamiento y gestión de identidades"),("Registros","Historial de accesos, verificaciones faciales y control de EPP"),("Configuración","Requisitos, cámara y comportamiento del puesto"),("Diagnóstico","Estado técnico de los servicios locales"))
        if hasattr(self,"admin_title"):self.admin_title.setText(titles[index][0]);self.admin_subtitle.setText(titles[index][1]);self.admin_heading_icon.setPixmap(icon(("users","records","settings","info")[index],"#2878ef",28).pixmap(28,28))
        for i,b in enumerate(self.admin_nav):b.setChecked(i==index)
        if self.stack.currentIndex()==2: self.select_page(2)
        if index==1: self.refresh_events()

    def select_admin_tab(self,index):
        self.stack.setCurrentIndex(2);self.admin_tabs.setCurrentIndex(index);self.admin_changed(index)

    def toggle_overlay(self, button): self.overlay=not self.overlay; button.setText("Encuadre visible" if self.overlay else "Encuadre oculto")
    def toggle_maximize(self): self.showNormal() if self.isMaximized() else self.showMaximized()
    def toggle_fullscreen(self): self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_F11: self.toggle_fullscreen()
        elif event.key()==Qt.Key.Key_Escape and self.isFullScreen(): self.showNormal()
        else: super().keyPressEvent(event)

    def compose(self, frame, result, age=.7):
        image=frame.image.copy()
        if self.overlay and result and time.perf_counter()-result["captured"]<age:
            mask,preview=result.get("annotation_mask"),result.get("preview")
            if mask is not None and preview is not None and preview.shape==image.shape: image[mask]=preview[mask]
        return image

    def refresh_people(self):
        if not hasattr(self,"people_list"): return
        selected={item.text() for item in self.people_list.selectedItems()}; search=self.people_search.text().casefold(); names=list_people()
        self.people_list.clear()
        for name in names:
            if search in name.casefold():
                item=QListWidgetItem(name);item.setIcon(icon("face","#2878ef",25));item.setSizeHint(QSize(0,54));self.people_list.addItem(item);item.setSelected(name in selected)
        self.people_count.setText(f"{self.people_list.count()} de {len(names)} personas")

    def new_person(self):
        self.people_list.clearSelection();self.people_list.setCurrentItem(None);self.enroll_name.clear();self.profile_name.setText("Nueva persona");self.profile_avatar.setText("+");self.profile_meta.setText("Complete el nombre y capture las muestras faciales");self.profile_state.setText("●  Pendiente");self.profile_state.setObjectName("profilePending");self.enroll_preview_count=0
        for index,slot in enumerate(self.enroll_samples):slot.clear();slot.setText(str(index+1))

    def person_selected(self,current,previous=None):
        if current is None:return
        name=current.text();self.enroll_name.blockSignals(True);self.enroll_name.setText(name);self.enroll_name.blockSignals(False);self.profile_name.setText(name);initials="".join(part[0] for part in name.split()[:2]).upper() or "?";self.profile_avatar.setText(initials);images=list_enrollment_images(name);self.profile_meta.setText(f"Identidad facial enrolada · {max(8,len(images))} plantillas biométricas");self.profile_state.setText("●  Activo");self.profile_state.setObjectName("profileActive");self.profile_state.style().unpolish(self.profile_state);self.profile_state.style().polish(self.profile_state)
        for index,slot in enumerate(self.enroll_samples):
            slot.clear();slot.setText(str(index+1))
            if index<len(images):slot.setText("");slot.setPixmap(QPixmap(str(images[index])).scaled(slot.size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation))

    def preview_person_name(self,text):
        name=text.strip();self.profile_name.setText(name or "Nueva persona");self.profile_avatar.setText("".join(part[0] for part in name.split()[:2]).upper() if name else "+")

    def profile_menu(self):
        menu=QMenu(self);remove=menu.addAction("Eliminar persona y enrolamiento");chosen=menu.exec(self.cursor().pos())
        if chosen==remove:self.remove_people()

    def update_enrollment_samples(self,facial_result):
        match=re.search(r"(\d+)/8",self.facial.status)
        if not match:return
        count=min(8,int(match.group(1)))
        if count<=self.enroll_preview_count:return
        frame=facial_result.get("preview")
        if frame is not None:
            mask=facial_result.get("annotation_mask")
            if mask is not None:
                points=cv2.findNonZero(mask.astype("uint8"))
                if points is not None:
                    x,y,w,h=cv2.boundingRect(points);pad=max(w,h)//3;x1=max(0,x-pad);y1=max(0,y-pad);x2=min(frame.shape[1],x+w+pad);y2=min(frame.shape[0],y+h+pad);frame=frame[y1:y2,x1:x2]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);image=QImage(rgb.data,rgb.shape[1],rgb.shape[0],rgb.strides[0],QImage.Format.Format_RGB888).copy()
            for index in range(self.enroll_preview_count,count):self.enroll_samples[index].setText("");self.enroll_samples[index].setPixmap(QPixmap.fromImage(image).scaled(self.enroll_samples[index].size(),Qt.AspectRatioMode.KeepAspectRatioByExpanding,Qt.TransformationMode.SmoothTransformation))
        self.enroll_preview_count=count

    def start_enrollment(self):
        name=self.enroll_name.text().strip()
        if not name or len(name)>80: QMessageBox.information(self,"Nombre","Escriba un nombre válido."); return
        if name in list_people() and QMessageBox.question(self,"Reemplazar",f"¿Reemplazar las muestras de {name}?")!=QMessageBox.StandardButton.Yes: return
        try:
            self.enroll_preview_count=0
            for index,slot in enumerate(self.enroll_samples):slot.clear();slot.setText(str(index+1))
            self.facial.command("enroll",name); self.facial.activate();self.profile_state.setText("●  Enrolando")
        except queue.Full: QMessageBox.information(self,"Enrolamiento","Existe otra operación pendiente.")

    def remove_people(self):
        names=[i.text() for i in self.people_list.selectedItems()]
        if not names: return
        if QMessageBox.question(self,"Eliminar personas",f"¿Eliminar {len(names)} persona(s)?")!=QMessageBox.StandardButton.Yes: return
        for name in names: delete_person(name)
        self.facial.command("reload"); self.refresh_people()

    def state_badge(self,text,positive=None):
        widget=QWidget();row=QHBoxLayout(widget);row.setContentsMargins(6,0,6,0);row.setSpacing(7);symbol=QLabel("✓" if positive else "×" if positive is False else "•");symbol.setObjectName("stateSymbol");symbol.setAlignment(Qt.AlignmentFlag.AlignCenter);symbol.setFixedSize(21,21);symbol.setProperty("tone","ok" if positive else "bad" if positive is False else "neutral");label=QLabel(text);label.setObjectName("stateText");label.setProperty("tone","ok" if positive else "bad" if positive is False else "neutral");row.addWidget(symbol);row.addWidget(label);row.addStretch();return widget

    def module_badge(self,event_type):
        widget=QWidget();row=QHBoxLayout(widget);row.setContentsMargins(5,0,5,0);badge=QLabel("Reconocimiento" if event_type=="facial" else "Protección EPP");badge.setObjectName("moduleBadge");badge.setProperty("module",event_type);badge.setAlignment(Qt.AlignmentFlag.AlignCenter);row.addWidget(badge);return widget

    def action_button(self,row):
        button=QToolButton();button.setObjectName("moreButton");button.setText("⋮");button.clicked.connect(lambda:self.open_record_menu(row,button));return button

    def open_record_menu(self,row,button):
        if row>=len(self.filtered_events):return
        for current in range(self.events.rowCount()):self.events.item(current,0).setCheckState(Qt.CheckState.Unchecked)
        self.events.clearSelection();self.events.selectRow(row);menu=QMenu(self);view=menu.addAction("Ver evidencia");remove=menu.addAction("Eliminar registro");chosen=menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        if chosen==view:self.view_evidence()
        elif chosen==remove:self.delete_selected()

    def refresh_events(self):
        if not hasattr(self,"events"): return
        self.refresh_record_stats()
        try:
            all_rows=query_events(self.record_person.text(),self.record_type.currentText(),self.record_status.currentText(),self.record_from.text(),self.record_to.text(),100000,0)
            rows=query_events(self.record_person.text(),self.record_type.currentText(),self.record_status.currentText(),self.record_from.text(),self.record_to.text(),self.page_size,self.page*self.page_size)
        except Exception as error: QMessageBox.warning(self,"Filtros",str(error)); return
        self.total_events=len(all_rows);self.has_more=(self.page+1)*self.page_size<self.total_events;self.filtered_events=rows;self.events.clearContents();self.events.setRowCount(len(rows))
        module=self.record_type.currentText()
        for row,record in enumerate(self.filtered_events):
            try: stamp=datetime.fromisoformat(record["timestamp"]).astimezone().strftime("%d/%m/%Y\n%H:%M:%S")
            except ValueError: stamp=record["timestamp"]
            if module=="facial":values=("",stamp,record["person_name"] or "Sin identificar",record["status"],record["liveness_status"] or "—",f'{record["facial_score"]:.3f}' if record["facial_score"] is not None else "—","","")
            elif module=="epp":values=("",stamp,record["person_name"] or "Sin identificar",record["status"],STATE_LABELS.get(record["helmet_status"],record["helmet_status"] or "—"),STATE_LABELS.get(record["vest_status"],record["vest_status"] or "—"),STATE_LABELS.get(record["gloves_status"],record["gloves_status"] or "—"),"","")
            else:values=("",stamp,"Reconocimiento" if record["event_type"]=="facial" else "Protección EPP",record["person_name"] or "Sin identificar",record["status"],STATE_LABELS.get(record["helmet_status"],record["helmet_status"] or "—"),STATE_LABELS.get(record["vest_status"],record["vest_status"] or "—"),STATE_LABELS.get(record["gloves_status"],record["gloves_status"] or "—"),record["liveness_status"] or "—","","")
            for col,value in enumerate(values):
                item=QTableWidgetItem(str(value));item.setTextAlignment(Qt.AlignmentFlag.AlignVCenter|Qt.AlignmentFlag.AlignLeft)
                self.events.setItem(row,col,item)
            check=self.events.item(row,0);check.setFlags(Qt.ItemFlag.ItemIsEnabled|Qt.ItemFlag.ItemIsSelectable|Qt.ItemFlag.ItemIsUserCheckable);check.setCheckState(Qt.CheckState.Unchecked)
            status_col=3 if module in ("facial","epp") else 4;positive=record["status"] in ("RECONOCIDO","COMPLETO");status_text={"RECONOCIDO":"Autorizado","DESCONOCIDO":"No registrado","SPOOF_RECHAZADO":"Rechazado","COMPLETO":"EPP correcto","INCOMPLETO":"EPP incompleto"}.get(record["status"],record["status"]);self.events.setCellWidget(row,status_col,self.state_badge(status_text,positive))
            self.events.item(row,status_col).setText("")
            epp_start=4 if module=="epp" else 5 if module=="Todos" else None
            if epp_start is not None:
                for offset,key in enumerate(("helmet_status","vest_status","gloves_status")):
                    state=record[key]
                    if state:self.events.item(row,epp_start+offset).setText("");self.events.setCellWidget(row,epp_start+offset,self.state_badge("",True if state=="si" else False if state=="no" else None))
            if module=="Todos":self.events.item(row,2).setText("");self.events.setCellWidget(row,2,self.module_badge(record["event_type"]))
            evidence_col=len(values)-2
            path=ROOT/record["image_path"] if record["image_path"] else None
            if path and path.exists():
                preview=QLabel();preview.setObjectName("thumbnail");preview.setFixedSize(52,52);preview.setAlignment(Qt.AlignmentFlag.AlignCenter);preview.setPixmap(rounded_thumbnail(path));holder=QWidget();layout=QHBoxLayout(holder);layout.setContentsMargins(0,0,0,0);layout.addWidget(preview,0,Qt.AlignmentFlag.AlignCenter);self.events.setCellWidget(row,evidence_col,holder)
            self.events.setCellWidget(row,len(values)-1,self.action_button(row))
        start=self.page*self.page_size+1 if self.total_events else 0;end=min((self.page+1)*self.page_size,self.total_events);self.records_count.setText(f"Mostrando {start} - {end} de {self.total_events} registros");self.update_page_buttons()

    def refresh_record_stats(self):
        if not hasattr(self,"record_stats"):return
        try:records=query_events(limit=10000)
        except Exception:return
        today=datetime.now().astimezone().date();today_count=recognized=rejected=incomplete=0
        for record in records:
            try:
                if datetime.fromisoformat(record["timestamp"]).astimezone().date()==today:today_count+=1
            except ValueError:pass
            if record["status"]=="RECONOCIDO":recognized+=1
            elif record["status"] in ("DESCONOCIDO","SPOOF_RECHAZADO"):rejected+=1
            elif record["status"]=="INCOMPLETO":incomplete+=1
        for label,value in zip(self.record_stats,(today_count,recognized,rejected,incomplete)):label.setText(str(value))

    def select_record_module(self,selected):
        for button in self.module_buttons:button.setChecked(button is selected)
        self.record_type.setCurrentText(selected.property("module"));self.page=0;self.update_record_view()

    def update_record_view(self):
        module=self.record_type.currentText()
        if module=="facial":headers=("☐","Fecha y hora","Persona","Resultado","Prueba de vida","Similitud","Imagen","Acciones");statuses=("Todos","RECONOCIDO","DESCONOCIDO","SPOOF_RECHAZADO")
        elif module=="epp":headers=("☐","Fecha y hora","Persona","Resultado","Casco","Chaleco","Guantes","Imagen","Acciones");statuses=("Todos","COMPLETO","INCOMPLETO")
        else:headers=("☐","Fecha y hora","Módulo","Persona","Resultado","Casco","Chaleco","Guantes","Vida","Imagen","Acciones");statuses=("Todos","RECONOCIDO","DESCONOCIDO","SPOOF_RECHAZADO","COMPLETO","INCOMPLETO")
        previous=self.record_status.currentText();self.record_status.blockSignals(True);self.record_status.clear();self.record_status.addItems(statuses);self.record_status.setCurrentText(previous if previous in statuses else "Todos");self.record_status.blockSignals(False)
        self.events.setColumnCount(len(headers));self.events.setHorizontalHeaderLabels(headers);header=self.events.horizontalHeader();header.setStretchLastSection(False)
        for column in range(len(headers)):header.setSectionResizeMode(column,QHeaderView.ResizeMode.Fixed)
        widths={"☐":44,"Fecha y hora":130,"Módulo":170,"Persona":170,"Resultado":190,"Casco":98,"Chaleco":98,"Guantes":98,"Vida":95,"Prueba de vida":125,"Similitud":95,"Imagen":86,"Acciones":88}
        for column,name in enumerate(headers):self.events.setColumnWidth(column,widths.get(name,110))
        person_col=headers.index("Persona");header.setSectionResizeMode(person_col,QHeaderView.ResizeMode.Stretch);self.refresh_events()

    def clear_record_filters(self):
        self.record_person.clear();self.record_from.clear();self.record_to.clear();self.record_status.setCurrentIndex(0);self.page=0;self.refresh_events()

    def search_events(self): self.page=0; self.refresh_events()
    def change_page(self,delta):
        if delta<0 and self.page==0 or delta>0 and not self.has_more:return
        self.page+=delta; self.refresh_events()
    def go_to_page(self,page):
        total=max(1,(self.total_events+self.page_size-1)//self.page_size)
        if 0<=page<total:self.page=page;self.refresh_events()
    def change_page_size(self):
        self.page_size=(8,15,30)[self.per_page.currentIndex()];self.page=0;self.refresh_events()
    def update_page_buttons(self):
        total=max(1,(self.total_events+self.page_size-1)//self.page_size);start=max(0,min(self.page-2,total-5))
        for index,button in enumerate(self.page_buttons):
            page=start+index;button.setVisible(page<total);button.setText(str(page+1));button.setProperty("page",page);button.setChecked(page==self.page)
            try:button.clicked.disconnect()
            except TypeError:pass
            button.clicked.connect(lambda _,b=button:self.go_to_page(int(b.property("page"))))
    def header_clicked(self,column):
        if column!=0:return
        select=not all(self.events.item(row,0).checkState()==Qt.CheckState.Checked for row in range(self.events.rowCount()))
        for row in range(self.events.rowCount()):self.events.item(row,0).setCheckState(Qt.CheckState.Checked if select else Qt.CheckState.Unchecked)
    def selected_records(self):
        checked=[self.filtered_events[row] for row in range(self.events.rowCount()) if self.events.item(row,0).checkState()==Qt.CheckState.Checked]
        return checked or [self.filtered_events[i.row()] for i in self.events.selectionModel().selectedRows()]
    def delete_selected(self):
        records=self.selected_records()
        if records and QMessageBox.question(self,"Eliminar",f"¿Eliminar {len(records)} registro(s) y sus imágenes?")==QMessageBox.StandardButton.Yes: delete_events([r["id"] for r in records]); self.refresh_events()
    def view_evidence(self):
        records=self.selected_records()
        if not records:return
        record=records[0]; dialog=QDialog(self); dialog.setWindowTitle(f"Evidencia #{record['id']}"); dialog.resize(900,650); col=QVBoxLayout(dialog); col.addWidget(QLabel(f"{record['status']} · {record['person_name'] or 'Sin identidad'}"))
        image=QLabel("Imagen no disponible",alignment=Qt.AlignmentFlag.AlignCenter); path=ROOT/record["image_path"] if record["image_path"] else None
        if path and path.exists(): image.setPixmap(QPixmap(str(path)).scaled(840,500,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))
        col.addWidget(image,1); close=QPushButton("Cerrar"); close.clicked.connect(dialog.accept); col.addWidget(close,0,Qt.AlignmentFlag.AlignRight); dialog.exec()
    def export_events(self):
        rows=self.selected_records() or self.filtered_events
        if not rows:return
        path,_=QFileDialog.getSaveFileName(self,"Exportar registros","registros.csv","CSV (*.csv)")
        if path:
            with open(path,"w",newline="",encoding="utf-8-sig") as output:
                writer=csv.writer(output,delimiter=";"); columns=tuple(rows[0].keys()); writer.writerow(columns); writer.writerows(tuple(r[c] for c in columns) for r in rows)

    def export_report(self):
        rows=self.selected_records() or self.filtered_events
        if not rows:return
        path,_=QFileDialog.getSaveFileName(self,"Exportar reporte","reporte_registros.html","HTML (*.html)")
        if not path:return
        body=[]
        for record in rows:
            body.append("<tr>"+"".join(f"<td>{html.escape(str(value if value is not None else '—'))}</td>" for value in (record["timestamp"],record["event_type"],record["person_name"],record["status"],record["helmet_status"],record["vest_status"],record["gloves_status"],record["liveness_status"]))+"</tr>")
        document="<!doctype html><meta charset='utf-8'><title>Reporte de registros</title><style>body{font-family:Segoe UI;margin:36px;color:#172033}table{border-collapse:collapse;width:100%}th,td{padding:10px;border-bottom:1px solid #dfe5ec;text-align:left}th{background:#f1f5f9}</style><h1>Reporte de registros</h1><p>Generado "+datetime.now().strftime("%d/%m/%Y %H:%M")+"</p><table><thead><tr><th>Fecha</th><th>Módulo</th><th>Persona</th><th>Resultado</th><th>Casco</th><th>Chaleco</th><th>Guantes</th><th>Vida</th></tr></thead><tbody>"+"".join(body)+"</tbody></table>"
        Path(path).write_text(document,encoding="utf-8")

    def save_requirements(self):
        try: saved=set_required_epp([n for n,c in self.requirement_checks.items() if c.isChecked()]); self.epp.set_required(saved); QMessageBox.information(self,"EPP","Requisitos actualizados.")
        except ValueError as error: QMessageBox.warning(self,"EPP",str(error))
    def save_settings(self):
        try:
            width,height=map(int,self.resolution.currentText().split("x")); self.config.update(camera=self.camera_edit.text().strip(),width=width,height=height,fps=self.fps.value(),voice_enabled=self.voice_check.isChecked()); save_config(self.config)
            if self.voice:self.voice.enabled=self.voice_check.isChecked()
            self.settings_note.setText("Configuración guardada. Reinicie para aplicar la cámara.")
        except Exception as error: QMessageBox.warning(self,"Configuración",str(error))
    def snapshot(self):
        frame=self.camera.latest.get()
        if frame is None:return
        path,_=QFileDialog.getSaveFileName(self,"Guardar captura","captura.jpg","JPEG (*.jpg)")
        if path: cv2.imwrite(path,frame.image)

    def record_once(self,kind,key,**event):
        now=time.monotonic()
        if self.event_keys[kind] is not None:return
        record_event(**event); self.event_keys[kind]=key; self.event_seen[kind]=now
    def record_facial(self,result):
        live=result.get("liveness")
        if live=="falso":status,name="SPOOF_RECHAZADO",None
        elif result.get("identity"):status,name="RECONOCIDO",result["identity"]
        elif result.get("valid") and live=="real" and not result.get("candidate"):status,name="DESCONOCIDO",None
        else:return
        self.record_once("facial",(status,name),event_type="facial",status=status,frame=result["preview"],person_name=name,liveness_status=live,facial_score=result.get("score"),camera=self.config["camera"])
    def record_epp(self,result):
        if len(result["people"])!=1:return
        person=result["people"][0]; now=time.monotonic()
        if person["id"]!=self.presence_id:self.presence_id=person["id"];self.event_keys["epp"]=None;self.voice_announced=None
        if person["overall"] not in ("COMPLETO","INCOMPLETO"):self.pending_epp=None;return
        key=(person["id"],person["overall"],tuple(person["decisions"].items()))
        if key!=self.pending_epp:self.pending_epp=key;self.pending_since=now;return
        if now-self.pending_since>=.8:self.record_once("epp",key,event_type="epp",status=person["overall"],frame=result["preview"],decisions=person["decisions"],camera=self.config["camera"])
        if self.voice and key!=self.voice_announced and now-self.pending_since>=1:self.voice.announce_epp(person["overall"],person["decisions"],self.epp.required);self.voice_announced=key

    def tick(self):
        now=time.perf_counter(); frame=self.camera.latest.get(); fr=self.facial.latest.get(); er=self.epp.latest.get(); page=self.stack.currentIndex()
        wall=datetime.now();self.admin_date.setText(wall.strftime("%d/%m/%Y"));self.admin_time.setText(wall.strftime("%H:%M"))
        for kind in self.event_keys:
            if time.monotonic()-self.event_seen[kind]>3:self.event_keys[kind]=None
        if frame is not None and frame.sequence!=self.last_camera_sequence:
            self.last_camera_sequence=frame.sequence
            if page==0:self.facial_video.show_frame(self.compose(frame,fr))
            elif page==1:self.epp_video.show_frame(self.compose(frame,er))
            elif page==2 and self.admin_tabs.currentIndex()==0:self.enroll_video.show_frame(self.compose(frame,fr))
        if page in (0,2) and fr and now-fr["captured"]<1:
            if page==0 and fr["sequence"]!=self.last_facial_sequence:self.last_facial_sequence=fr["sequence"];self.record_facial(fr)
            if page==0:
                live=fr.get("liveness"); title="Presentación rechazada" if live=="falso" else fr["identity"] if fr.get("identity") else "Verificando identidad"
                self.facial_result.title.setText(title); self.facial_live.setText({"real":"Prueba de vida confirmada","falso":"Posible foto o pantalla"}.get(live,"Prueba de vida pendiente")); self.facial_detail.setText(f'{fr["quality"]}\nAnálisis: {fr["inference_ms"]:.0f} ms')
            else:
                self.enroll_status.setText(self.facial.status+" · "+fr["quality"])
                if self.facial.enrolling:self.update_enrollment_samples(fr)
                if self.enrolling_before and not self.facial.enrolling:
                    completed_name=self.enroll_name.text().strip();self.refresh_people();matches=self.people_list.findItems(completed_name,Qt.MatchFlag.MatchExactly)
                    if matches:self.people_list.setCurrentItem(matches[0]);self.person_selected(matches[0])
                    self.profile_state.setText("●  Activo");self.profile_state.setObjectName("profileActive");self.profile_state.style().unpolish(self.profile_state);self.profile_state.style().polish(self.profile_state)
                self.enrolling_before=self.facial.enrolling
        if page==1:
            person=er["people"][0] if er and now-er["captured"]<1 and len(er["people"])==1 else None
            if er and now-er["captured"]<1 and er["sequence"]!=self.last_epp_sequence:self.last_epp_sequence=er["sequence"];self.record_epp(er)
            self.epp_result.title.setText({"COMPLETO":"Equipo completo","INCOMPLETO":"Equipo incompleto","VERIFICANDO":"Verificando equipo"}.get(person["overall"],"Esperando persona") if person else "Esperando persona")
            for name,(row,state) in self.epp_rows.items():row.setVisible(name in self.epp.required); value=person["decisions"].get(name,"verificando") if person else "verificando";state.setText(STATE_LABELS.get(value,value))
            if er:self.epp_result.note.setText(er.get("guidance","Mantenga visibles los elementos requeridos."))
        if page==2 and self.admin_tabs.currentIndex()==3:self.diagnostics.setPlainText(json.dumps({"camera":self.camera.status,"facial":self.facial.status,"epp":self.epp.status,"voice":self.voice.status if self.voice else "No disponible"},ensure_ascii=False,indent=2))

    def closeEvent(self,event): self.stop.set(); event.accept()


def run_qt(camera, facial, epp, stop, config, voice=None, smoke_test=False):
    app=QApplication.instance() or QApplication([]); app.setApplicationName("SSeguridad")
    window=QtMainWindow(camera,facial,epp,stop,config,voice,smoke_test); window.show(); return app.exec()
