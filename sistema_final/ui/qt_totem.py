"""Full-screen facial access view for an unattended checkpoint."""
import time
from datetime import datetime

import cv2
from PyQt6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QGraphicsOpacityEffect, QHBoxLayout, QLabel, QMainWindow,
    QProgressBar, QSizePolicy, QStackedWidget, QVBoxLayout, QWidget,
)

from sistema_final.core.configuration import ROOT
from sistema_final.core.database import record_event


class FeedbackOverlay(QFrame):
    """Compact result overlay that leaves the live camera visible."""
    def __init__(self,parent):
        super().__init__(parent); self.setObjectName("feedback"); self.setFixedSize(470,190); self.hide()
        col=QVBoxLayout(self); col.setContentsMargins(24,18,24,18); col.setSpacing(8)
        self.symbol=QLabel(); self.symbol.setObjectName("feedbackSymbol"); self.symbol.setFixedSize(68,68)
        self.symbol.setAlignment(Qt.AlignmentFlag.AlignCenter); col.addWidget(self.symbol,0,Qt.AlignmentFlag.AlignHCenter)
        self.title=QLabel(); self.title.setObjectName("feedbackTitle"); self.title.setAlignment(Qt.AlignmentFlag.AlignCenter); col.addWidget(self.title)
        self.detail=QLabel(); self.detail.setObjectName("feedbackDetail"); self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter); col.addWidget(self.detail)
        self.effect=QGraphicsOpacityEffect(self); self.setGraphicsEffect(self.effect)
        self.animation=QPropertyAnimation(self.effect,b"opacity",self); self.animation.setDuration(850)
        self.animation.setStartValue(.72); self.animation.setEndValue(1.0); self.animation.setEasingCurve(QEasingCurve.Type.InOutSine)
        self.animation.setLoopCount(-1)

    def display(self,success,title,detail):
        self.setProperty("result","success" if success else "failure")
        self.symbol.setText("✓" if success else "×"); self.title.setText(title); self.detail.setText(detail)
        self.style().unpolish(self); self.style().polish(self); self.show(); self.raise_(); self.animation.start()

    def dismiss(self):
        self.animation.stop(); self.hide()


class CameraView(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("cameraCard")
        layout=QVBoxLayout(self); layout.setContentsMargins(7,7,7,7)
        self.image=QLabel("Esperando cámara"); self.image.setObjectName("camera")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter); self.image.setMinimumSize(640,360)
        self.image.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored)
        layout.addWidget(self.image)
        self.feedback=FeedbackOverlay(self)

    def resizeEvent(self,event):
        super().resizeEvent(event)
        self.feedback.move((self.width()-self.feedback.width())//2,
                           (self.height()-self.feedback.height())//2)

    def show_frame(self, frame, guide="idle"):
        """Presenta el video sin deformarlo y sin filtros costosos por frame."""
        source=frame.copy(); height,width=source.shape[:2]
        if guide!="hidden":
            length=min(width,height)//14; margin=min(width,height)//18
            color=(242,242,242) if guide=="idle" else (255,142,66)
            thick=max(2,min(width,height)//210)
            for x,y,sx,sy in ((margin,margin,1,1),(width-margin,margin,-1,1),
                               (margin,height-margin,1,-1),(width-margin,height-margin,-1,-1)):
                cv2.line(source,(x,y),(x+sx*length,y),color,thick,cv2.LINE_AA)
                cv2.line(source,(x,y),(x,y+sy*length),color,thick,cv2.LINE_AA)
        rgb=cv2.cvtColor(source,cv2.COLOR_BGR2RGB)
        image=QImage(rgb.data,width,height,rgb.strides[0],QImage.Format.Format_RGB888).copy()
        self.image.setPixmap(QPixmap.fromImage(image).scaled(
            self.image.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.FastTransformation))


class DecisionCard(QFrame):
    def __init__(self):
        super().__init__(); self.setObjectName("decisionCard")
        col=QVBoxLayout(self); col.setContentsMargins(40,40,40,40); col.addStretch()
        self.symbol=QLabel(); self.symbol.setObjectName("decisionSymbol"); self.symbol.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.title=QLabel(); self.title.setObjectName("decisionTitle"); self.title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detail=QLabel(); self.detail.setObjectName("decisionDetail"); self.detail.setAlignment(Qt.AlignmentFlag.AlignCenter); self.detail.setWordWrap(True)
        col.addWidget(self.symbol); col.addSpacing(20); col.addWidget(self.title); col.addSpacing(12); col.addWidget(self.detail); col.addStretch()

    def set_result(self, success, title, detail):
        self.setProperty("result", "success" if success else "failure")
        self.style().unpolish(self); self.style().polish(self)
        self.symbol.setText("✓" if success else "×"); self.title.setText(title); self.detail.setText(detail)


class TotemWindow(QMainWindow):
    def __init__(self,camera,facial,stop,config,smoke_test=False):
        super().__init__(); self.camera,self.facial,self.stop,self.config=camera,facial,stop,config
        self.last_sequence=-1; self.presence=False; self.presence_token=0; self.last_face=0.; self.unknown_since=None
        self.recorded=None; self.hold_until=0.; self.current_state="idle"
        self.setWindowTitle("SSeguridad | Punto de acceso"); self.setMinimumSize(900,650)
        self._build(); self._style(); self.showFullScreen(); self.facial.activate()
        self.timer=QTimer(self); self.timer.timeout.connect(self.tick); self.timer.start(33)
        if smoke_test: QTimer.singleShot(2500,self.close)

    def _build(self):
        root=QWidget(); root.setObjectName("root"); self.setCentralWidget(root)
        col=QVBoxLayout(root); col.setContentsMargins(34,24,34,24); col.setSpacing(18)
        header=QHBoxLayout(); self.logo=QLabel(); logo=QPixmap(str(ROOT/"logo.png")); self.logo.setPixmap(logo.scaled(285,82,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation)); header.addWidget(self.logo); header.addStretch()
        clockbox=QVBoxLayout(); self.date=QLabel(); self.date.setObjectName("date"); self.date.setAlignment(Qt.AlignmentFlag.AlignRight); self.clock=QLabel(); self.clock.setObjectName("clock"); self.clock.setAlignment(Qt.AlignmentFlag.AlignRight); clockbox.addWidget(self.date); clockbox.addWidget(self.clock); header.addLayout(clockbox); col.addLayout(header)
        self.content=QStackedWidget(); self.camera_page=QWidget(); camera_col=QVBoxLayout(self.camera_page); camera_col.setContentsMargins(12,0,12,0)
        self.camera_view=CameraView(); camera_col.addWidget(self.camera_view,1)
        self.headline=QLabel("Mire a la cámara"); self.headline.setObjectName("headline"); self.headline.setAlignment(Qt.AlignmentFlag.AlignCenter); camera_col.addWidget(self.headline)
        self.instruction=QLabel("Ubique su rostro dentro del recuadro"); self.instruction.setObjectName("instruction"); self.instruction.setAlignment(Qt.AlignmentFlag.AlignCenter); camera_col.addWidget(self.instruction)
        self.progress=QProgressBar(); self.progress.setRange(0,1); self.progress.setValue(0)
        self.progress.setTextVisible(False); self.progress.setFixedHeight(7); self.progress.setMaximumWidth(460)
        camera_col.addWidget(self.progress,0,Qt.AlignmentFlag.AlignHCenter)
        self.content.addWidget(self.camera_page); col.addWidget(self.content,1)
        footer=QHBoxLayout(); footer.addWidget(QLabel("Control de acceso",objectName="footer")); footer.addStretch(); col.addLayout(footer)

    def _style(self):
        self.setStyleSheet("""
        #root { background: #0c131b; } QLabel { color: #f5f7fa; font-family: 'Segoe UI'; }
        #date { color: #9ca8b7; font-size: 15px; } #clock { font: 700 29px 'Segoe UI'; }
        #cameraCard { background: #121d28; border: 1px solid #3c4b5c; border-radius: 16px; }
        #camera { background: #111923; border-radius: 11px; color: #8793a0; }
        #headline { font: 700 34px 'Segoe UI'; margin-top: 14px; } #instruction { color: #b5bec9; font-size: 19px; margin-bottom: 8px; }
        QProgressBar { height: 7px; border: 0; border-radius: 3px; background: #293746; }
        QProgressBar::chunk { background: #4b8cff; border-radius: 3px; }
        #feedback { border-radius: 18px; }
        #feedback[result="success"] { background: rgba(8, 48, 36, 238); border: 2px solid #42d991; }
        #feedback[result="failure"] { background: rgba(58, 18, 23, 238); border: 2px solid #f0525d; }
        #feedbackSymbol { font: 700 43px 'Segoe UI'; border: 4px solid #e8edf2; border-radius: 34px; }
        #feedbackTitle { font: 700 24px 'Segoe UI'; } #feedbackDetail { color: #c8d0da; font-size: 15px; }
        #footer { color: #8793a0; font-size: 14px; }
        """)

    @staticmethod
    def actionable_message(result):
        if not result:return "Ubique su rostro dentro del área"
        quality=(result.get("quality") or "").lower()
        if "demasiado pequeño" in quality:return "Acérquese un poco"
        if "una sola persona" in quality:return "Una persona a la vez"
        if "centra" in quality:return "Rostro fuera del área"
        if "iluminación" in quality:return "Ilumine su rostro de frente"
        if "borrosa" in quality:return "Mire de frente y manténgase quieto"
        if result.get("liveness")=="sin rostro" or "sin rostro" in quality:return "Ubique su rostro dentro del área"
        return "Mire de frente" if not result.get("valid") else "Mantenga el rostro visible"

    def show_idle(self,quality=None):
        self.current_state="idle"; self.content.setCurrentIndex(0); self.progress.setRange(0,1); self.progress.setValue(0); self.camera_view.feedback.dismiss()
        self.headline.setText("Mire a la cámara"); self.instruction.setText(quality or "Ubique su rostro dentro del recuadro")

    def show_verifying(self,text="Mantenga el rostro visible"):
        self.current_state="verifying"; self.content.setCurrentIndex(0); self.progress.setRange(0,0); self.camera_view.feedback.dismiss()
        self.headline.setText("Verificando identidad…"); self.instruction.setText(text)

    def show_decision(self,success,title,detail,hold=3.0):
        self.current_state="success" if success else "failure"; self.content.setCurrentIndex(0)
        self.progress.setRange(0,1); self.progress.setValue(0); self.camera_view.feedback.display(success,title,detail)
        self.hold_until=time.monotonic()+hold

    def record(self,key,result,status,name=None):
        if self.recorded==key:return
        try:
            record_event("facial",status,frame=result.get("preview"),person_name=name,
                         liveness_status=result.get("liveness"),facial_score=result.get("score"),
                         camera=self.config["camera"],details={"mode":"totem","quality":result.get("quality"),"margin":result.get("gap")})
            self.recorded=key
        except Exception:
            pass

    def tick(self):
        now=time.perf_counter(); wall=datetime.now()
        days=("Lun.","Mar.","Mié.","Jue.","Vie.","Sáb.","Dom.")
        months=("Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic")
        self.date.setText(f"{days[wall.weekday()]}, {wall.day:02d} {months[wall.month-1]} {wall.year}")
        self.clock.setText(wall.strftime("%H:%M"))
        frame=self.camera.latest.get(); result=self.facial.latest.get(); fresh=result and now-result["captured"]<.8
        face_seen=bool(fresh and (result.get("valid") or result.get("candidate") or result.get("liveness") in ("real","falso","verificando","inconcluso")))
        if frame is not None and frame.sequence!=self.last_sequence:
            self.last_sequence=frame.sequence; image=frame.image.copy()
            if fresh:
                mask,preview=result.get("annotation_mask"),result.get("preview")
                if mask is not None and preview is not None and preview.shape==image.shape:image[mask]=preview[mask]
            guide="hidden" if time.monotonic()<self.hold_until else "detected" if face_seen else "idle"
            self.camera_view.show_frame(image,guide)
        if face_seen:
            if not self.presence:self.presence_token+=1
            self.last_face=now; self.presence=True
        elif self.presence and now-self.last_face>1.0:
            self.presence=False; self.recorded=None; self.unknown_since=None; self.hold_until=0; self.show_idle()
        if time.monotonic()<self.hold_until:return
        if not fresh or not face_seen:self.show_idle(self.actionable_message(result if fresh else None));return
        live=result.get("liveness")
        if live=="falso":
            self.show_decision(False,"Verificación rechazada","No se aceptan fotografías ni pantallas.\nInténtelo nuevamente.",3.2); self.record(("spoof",self.presence_token),result,"SPOOF_RECHAZADO");return
        if result.get("identity") and live=="real":
            name=result["identity"]; self.show_decision(True,"Acceso autorizado",f"Bienvenido, {name}\nRegistro realizado · {wall.strftime('%H:%M')}",3.0); self.record(("recognized",name,self.presence_token),result,"RECONOCIDO",name);return
        if result.get("candidate") or live in ("verificando","inconcluso"):
            self.unknown_since=None; self.show_verifying();return
        if result.get("valid") and live=="real":
            if self.unknown_since is None:self.unknown_since=now
            if now-self.unknown_since<1.2:self.show_verifying("Comparando identidad…")
            else:self.show_decision(False,"Persona no registrada","No fue posible autorizar el acceso.\nSolicite asistencia.",3.0);self.record(("unknown",self.presence_token),result,"DESCONOCIDO")
            return
        self.show_idle(self.actionable_message(result))

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape:self.close()
        elif event.key()==Qt.Key.Key_F11:self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def closeEvent(self,event):self.stop.set();event.accept()


def run_totem(camera,facial,stop,config,smoke_test=False):
    app=QApplication.instance() or QApplication([]);app.setApplicationName("SSeguridad Tótem")
    window=TotemWindow(camera,facial,stop,config,smoke_test);window.show();return app.exec()
