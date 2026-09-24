"""Full-screen PPE checkpoint view backed by the existing EPP service."""
import time
from datetime import datetime

import cv2
from PIL.ImageQt import ImageQt
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QIcon, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication, QFrame, QHBoxLayout, QLabel, QMainWindow, QProgressBar,
    QSizePolicy, QVBoxLayout, QWidget,
)

from .icons import render_icon
from sistema_final.core.configuration import ROOT
from sistema_final.core.database import record_event

ITEMS={"casco":("Casco","helmet"),"chaleco":("Chaleco","vest"),"guantes":("Guantes","glove"),
       "antiparras":("Antiparras","shield"),"botas":("Botas","shield")}
CLASS_ITEM={"helmet":("casco",True),"no_helmet":("casco",False),"vest":("chaleco",True),
            "none":("chaleco",False),"gloves":("guantes",True),"no_gloves":("guantes",False),
            "goggles":("antiparras",True),"no_goggle":("antiparras",False),
            "boots":("botas",True),"no_boots":("botas",False)}


def qicon(name,color="#d6dce5",size=25):return QIcon(QPixmap.fromImage(ImageQt(render_icon(name,size,color))))


class LiveView(QFrame):
    def __init__(self):
        super().__init__();self.setObjectName("cameraCard");layout=QVBoxLayout(self);layout.setContentsMargins(7,7,7,7)
        self.image=QLabel("Esperando cámara");self.image.setObjectName("camera");self.image.setAlignment(Qt.AlignmentFlag.AlignCenter);self.image.setMinimumSize(540,420);self.image.setSizePolicy(QSizePolicy.Policy.Ignored,QSizePolicy.Policy.Ignored);layout.addWidget(self.image)

    def show_frame(self,frame,equipment):
        image=frame.copy();height,width=image.shape[:2];length=min(width,height)//14;margin=min(width,height)//18
        for x,y,sx,sy in ((margin,margin,1,1),(width-margin,margin,-1,1),(margin,height-margin,1,-1),(width-margin,height-margin,-1,-1)):
            cv2.line(image,(x,y),(x+sx*length,y),(238,190,65),3,cv2.LINE_AA);cv2.line(image,(x,y),(x,y+sy*length),(238,190,65),3,cv2.LINE_AA)
        for item in equipment:
            mapped=CLASS_ITEM.get(item["name"])
            if not mapped:continue
            name,positive=mapped;color=(85,211,81) if positive else (70,68,244);x1,y1,x2,y2=map(int,item["box"])
            cv2.rectangle(image,(x1,y1),(x2,y2),color,2,cv2.LINE_AA);text=ITEMS[name][0]+("  ✓" if positive else "  ×")
            size=cv2.getTextSize(text,cv2.FONT_HERSHEY_SIMPLEX,.48,1)[0];tx=max(3,min(width-size[0]-15,x2+7));ty=max(22,min(height-5,y1+22))
            cv2.rectangle(image,(tx-7,ty-18),(tx+size[0]+7,ty+6),color,-1,cv2.LINE_AA);cv2.putText(image,text,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,.48,(15,18,20),1,cv2.LINE_AA)
        rgb=cv2.cvtColor(image,cv2.COLOR_BGR2RGB);qimage=QImage(rgb.data,width,height,rgb.strides[0],QImage.Format.Format_RGB888).copy();self.image.setPixmap(QPixmap.fromImage(qimage).scaled(self.image.size(),Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation))


class EppTotemWindow(QMainWindow):
    def __init__(self,camera,epp,facial,stop,config,voice=None,smoke_test=False):
        super().__init__();self.camera,self.epp,self.facial,self.stop,self.config,self.voice=camera,epp,facial,stop,config,voice
        self.last_sequence=-1;self.presence_id=None;self.presence_started=0.;self.last_person_at=0.;self.pending=None;self.pending_since=0.;self.recorded=None
        self.identity=None;self.facial_score=None;self.liveness_status=None;self.face_size=0.
        self.setWindowTitle("SSeguridad | Verificación EPP");self.setMinimumSize(1000,680);self._build();self._style();self.showFullScreen();self.epp.activate();self.facial.activate()
        self.timer=QTimer(self);self.timer.timeout.connect(self.tick);self.timer.start(33)
        if smoke_test:QTimer.singleShot(2500,self.close)

    def _build(self):
        root=QWidget();root.setObjectName("root");self.setCentralWidget(root);outer=QVBoxLayout(root);outer.setContentsMargins(32,22,32,24);outer.setSpacing(18)
        header=QHBoxLayout();logo=QLabel();logo.setPixmap(QPixmap(str(ROOT/"logo.png")).scaled(275,78,Qt.AspectRatioMode.KeepAspectRatio,Qt.TransformationMode.SmoothTransformation));header.addWidget(logo);header.addStretch();clockbox=QVBoxLayout();self.date=QLabel();self.date.setObjectName("date");self.date.setAlignment(Qt.AlignmentFlag.AlignRight);self.clock=QLabel();self.clock.setObjectName("clock");self.clock.setAlignment(Qt.AlignmentFlag.AlignRight);clockbox.addWidget(self.date);clockbox.addWidget(self.clock);header.addLayout(clockbox);outer.addLayout(header)
        body=QHBoxLayout();body.setSpacing(28);self.status=QFrame();self.status.setObjectName("statusPanel");self.status.setFixedWidth(380);left=QVBoxLayout(self.status);left.setContentsMargins(30,28,30,28);left.setSpacing(13)
        eyebrow=QLabel("CONTROL Y VERIFICACIÓN");eyebrow.setObjectName("eyebrow");left.addWidget(eyebrow);self.title=QLabel("Verificación\nde EPP");self.title.setObjectName("title");self.title.setWordWrap(True);left.addWidget(self.title);self.subtitle=QLabel("Ubíquese frente a la cámara y mantenga su cuerpo visible");self.subtitle.setObjectName("subtitle");self.subtitle.setWordWrap(True);left.addWidget(self.subtitle)
        self.progress=QProgressBar();self.progress.setTextVisible(False);self.progress.setRange(0,1);self.progress.setValue(0);self.progress.setFixedHeight(7);left.addWidget(self.progress);self.rows={}
        for name,(label,icon_name) in ITEMS.items():
            row=QFrame();row.setObjectName("itemRow");layout=QHBoxLayout(row);layout.setContentsMargins(12,10,12,10);symbol=QLabel();symbol.setPixmap(qicon(icon_name).pixmap(25,25));text=QLabel(label);text.setObjectName("itemName");state=QLabel("Pendiente");state.setObjectName("itemState");layout.addWidget(symbol);layout.addWidget(text);layout.addStretch();layout.addWidget(state);left.addWidget(row);self.rows[name]=(row,state)
        left.addStretch();self.identity_label=QLabel("Identidad: esperando rostro");self.identity_label.setObjectName("identity");self.identity_label.setWordWrap(True);left.addWidget(self.identity_label);self.result_note=QLabel("Esperando una persona");self.result_note.setObjectName("resultNote");self.result_note.setWordWrap(True);left.addWidget(self.result_note);body.addWidget(self.status);self.live=LiveView();body.addWidget(self.live,1);outer.addLayout(body,1)
        footer=QHBoxLayout();footer.addWidget(QLabel("Control de acceso",objectName="footer"));footer.addStretch();self.operational=QLabel("●  Sistema operativo");self.operational.setObjectName("operational");footer.addWidget(self.operational);outer.addLayout(footer)

    def _style(self):
        self.setStyleSheet("""
        #root{background:#0c131b} QLabel{color:#f6f7f9;font-family:'Segoe UI'} #date{color:#98a4b2;font-size:15px} #clock{font:700 29px 'Segoe UI'}
        #statusPanel{background:#111b25;border:1px solid #263543;border-radius:16px} #eyebrow{color:#7e8b99;font:600 11px 'Segoe UI';letter-spacing:1px} #title{font:700 38px 'Segoe UI'} #subtitle{color:#b5bec9;font-size:18px} #cameraCard{background:#121d28;border:1px solid #3c4b5c;border-radius:16px} #camera{background:#101821;border-radius:11px;color:#8793a0}
        #itemRow{background:#182532;border-radius:9px} #itemName{font:600 14px 'Segoe UI'} #itemState{color:#9faab6;font-size:13px} #identity{color:#dce4ec;font:600 16px 'Segoe UI';padding:11px;background:#182532;border-radius:9px} #resultNote{font:600 18px 'Segoe UI';padding:14px;background:#192632;border-radius:9px}
        QProgressBar{border:0;border-radius:3px;background:#293746} QProgressBar::chunk{background:#4b8cff;border-radius:3px} #footer{color:#8793a0;font-size:14px} #operational{color:#63d89b;font-size:14px}
        """)

    def update_status(self,person,guidance):
        decisions=person["decisions"] if person else {};overall=person["overall"] if person else "VERIFICANDO"
        if overall=="COMPLETO":self.title.setText("EPP correcto");self.subtitle.setText("Acceso autorizado");self.result_note.setText("Registro realizado");self.status.setStyleSheet("#statusPanel{background:#0c3026;border:2px solid #3ed187;border-radius:16px}")
        elif overall=="INCOMPLETO":self.title.setText("EPP incompleto");self.subtitle.setText("Complete los elementos indicados");self.result_note.setText(guidance);self.status.setStyleSheet("#statusPanel{background:#34171b;border:2px solid #ef4d58;border-radius:16px}")
        else:self.title.setText("Analizando\nequipo de protección" if person else "Verificación\nde EPP");self.subtitle.setText("Permanezca quieto unos segundos" if person else guidance);self.result_note.setText(guidance);self.status.setStyleSheet("")
        self.progress.setRange(0,0 if person and overall=="VERIFICANDO" else 1);self.progress.setValue(0)
        for name,(row,state) in self.rows.items():
            required=name in self.epp.required;row.setVisible(required);value=decisions.get(name,"verificando");state.setText({"si":"Confirmado ✓","no":"Falta ×","verificando":"Analizando","no visible":"No visible"}.get(value,value));state.setStyleSheet("color:#52d991" if value=="si" else "color:#ff626b" if value=="no" else "color:#9faab6")

    def handle_event(self,result,person):
        if not person or person["overall"] not in ("COMPLETO","INCOMPLETO"):self.pending=None;return
        now=time.perf_counter()
        if self.identity is None and now-self.presence_started<3.5:return
        key=(person["id"],self.identity,person["overall"],tuple(sorted(person["decisions"].items())))
        if key!=self.pending:self.pending=key;self.pending_since=now;return
        if now-self.pending_since<.9 or self.recorded==key:return
        try:
            record_event("epp",person["overall"],frame=result["preview"],person_name=self.identity,decisions=person["decisions"],liveness_status=self.liveness_status,facial_score=self.facial_score,camera=self.config["camera"],details={"mode":"totem_epp_facial","presence_id":person["id"],"required":list(self.epp.required),"face_size":self.face_size,"identity_confirmed":bool(self.identity)});self.recorded=key
            if self.voice:self.voice.announce_epp(person["overall"],person["decisions"],self.epp.required)
        except Exception:self.operational.setText("Sistema operativo · registro pendiente")

    def update_identity(self,now):
        facial=self.facial.latest.get();fresh=facial and now-facial["captured"]<1
        if fresh:
            self.face_size=float(facial.get("face_size") or 0);self.liveness_status=facial.get("liveness");self.facial_score=facial.get("score")
            if facial.get("identity") and self.liveness_status=="real" and self.face_size>=70:self.identity=facial["identity"]
        if self.identity:self.identity_label.setText(f"Identidad: {self.identity}");self.identity_label.setStyleSheet("color:#52d991")
        elif fresh and self.face_size and self.face_size<70:self.identity_label.setText("Identidad: rostro muy lejano\nAcérquese levemente o mire a la cámara");self.identity_label.setStyleSheet("color:#f2bd57")
        elif fresh and self.liveness_status in ("falso","inconcluso"):self.identity_label.setText("Identidad: prueba de vida pendiente");self.identity_label.setStyleSheet("color:#f2bd57")
        else:self.identity_label.setText("Identidad: no confirmada");self.identity_label.setStyleSheet("")

    def tick(self):
        now=time.perf_counter();wall=datetime.now();days=("Lun.","Mar.","Mié.","Jue.","Vie.","Sáb.","Dom.");months=("Ene","Feb","Mar","Abr","May","Jun","Jul","Ago","Sep","Oct","Nov","Dic");self.date.setText(f"{days[wall.weekday()]}, {wall.day:02d} {months[wall.month-1]} {wall.year}");self.clock.setText(wall.strftime("%H:%M"))
        frame=self.camera.latest.get();result=self.epp.latest.get();fresh=result and time.perf_counter()-result["captured"]<1;person=result["people"][0] if fresh and len(result["people"])==1 else None
        if person:
            self.last_person_at=now
            if person["id"]!=self.presence_id:self.presence_id=person["id"];self.presence_started=now;self.recorded=None;self.pending=None;self.identity=None;self.facial_score=None;self.liveness_status=None;self.face_size=0.
            self.update_identity(now)
        elif self.presence_id is not None and now-self.last_person_at>1.2:
            self.presence_id=None;self.recorded=None;self.pending=None;self.identity=None;self.identity_label.setText("Identidad: esperando rostro");self.identity_label.setStyleSheet("")
        if frame is not None and frame.sequence!=self.last_sequence:self.last_sequence=frame.sequence;self.live.show_frame(frame.image,result.get("equipment",[]) if fresh else [])
        guidance=result.get("guidance","Ubíquese frente a la cámara") if fresh else "Ubíquese frente a la cámara";self.update_status(person,guidance)
        if fresh:self.handle_event(result,person)

    def keyPressEvent(self,event):
        if event.key()==Qt.Key.Key_Escape:self.close()
        elif event.key()==Qt.Key.Key_F11:self.showNormal() if self.isFullScreen() else self.showFullScreen()
    def closeEvent(self,event):self.stop.set();event.accept()


def run_epp_totem(camera,epp,facial,stop,config,voice=None,smoke_test=False):
    app=QApplication.instance() or QApplication([]);app.setApplicationName("SSeguridad Tótem EPP");window=EppTotemWindow(camera,epp,facial,stop,config,voice,smoke_test);window.show();return app.exec()
