import csv, queue, time, tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, ttk
import cv2
from PIL import Image, ImageTk
from sistema_final.core.configuration import ROOT
from sistema_final.core.database import (delete_events, delete_person, get_required_epp, list_people,
    query_events, record_event, set_required_epp)

class MainWindow:
    def __init__(self, root, camera, facial, epp, stop, config, voice=None):
        self.root, self.camera, self.facial, self.epp, self.stop, self.config, self.voice = root, camera, facial, epp, stop, config, voice
        self.last_sequences={"facial":-1,"epp":-1,"admin":-1}; self.last_camera_sequence=-1; self.was_enrolling=False; self.filtered_events=[]
        self.event_state={k:{"key":None,"time":0.,"seen":0.} for k in ("facial","epp")}
        self.voice_epp_state={"announced":None,"pending":None,"since":0.,"seen":0.}
        root.title("Sistema de reconocimiento facial y EPP"); root.geometry("1180x800"); root.minsize(960,700)
        style=ttk.Style(); style.theme_use("clam"); style.configure("Title.TLabel",font=("Segoe UI",20,"bold")); style.configure("Result.TLabel",font=("Segoe UI",15,"bold"))
        head=ttk.Frame(root,padding=(18,14)); head.pack(fill="x")
        ttk.Label(head,text="Control de reconocimiento y EPP",style="Title.TLabel").pack(side="left")
        self.system_status=ttk.Label(head,text="Iniciando cámara..."); self.system_status.pack(side="right")
        self.tabs=ttk.Notebook(root); self.tabs.pack(fill="both",expand=True,padx=16,pady=(0,16))
        self.facial_tab=ttk.Frame(self.tabs,padding=14); self.epp_tab=ttk.Frame(self.tabs,padding=14); self.admin_tab=ttk.Frame(self.tabs,padding=14)
        for tab,text in ((self.facial_tab,"  Detector facial  "),(self.epp_tab,"  Detector EPP  "),(self.admin_tab,"  Administración  ")): self.tabs.add(tab,text=text)
        self._build_facial(); self._build_epp(); self._build_admin(); self.tabs.bind("<<NotebookTabChanged>>",self._tab_changed)
        root.protocol("WM_DELETE_WINDOW",self.close); self._tab_changed(); root.after(20,self.tick)

    def _video_panel(self,parent,title):
        box=ttk.LabelFrame(parent,text=title,padding=10); box.pack(fill="both",expand=True)
        label=ttk.Label(box,text="Esperando imagen...",anchor="center"); label.pack(fill="both",expand=True); return label
    def _build_facial(self):
        ttk.Label(self.facial_tab,text="Reconocimiento facial",style="Title.TLabel").pack(anchor="w")
        ttk.Label(self.facial_tab,text="Una persona por vez · YuNet + SFace · procesamiento local").pack(anchor="w",pady=(0,10))
        self.facial_video=self._video_panel(self.facial_tab,"Última imagen analizada")
        self.facial_result=ttk.Label(self.facial_tab,text="Preparando...",style="Result.TLabel"); self.facial_result.pack(anchor="w",pady=(12,3))
        self.facial_details=ttk.Label(self.facial_tab); self.facial_details.pack(anchor="w")
    def _build_epp(self):
        ttk.Label(self.epp_tab,text="Verificación de EPP",style="Title.TLabel").pack(anchor="w")
        ttk.Label(self.epp_tab,text="Requisitos configurables · seguimiento y evidencia temporal").pack(anchor="w",pady=(0,10))
        self.epp_video=self._video_panel(self.epp_tab,"Análisis EPP")
        self.epp_result=ttk.Label(self.epp_tab,text="El modelo se cargará al abrir esta pestaña",style="Result.TLabel"); self.epp_result.pack(anchor="w",pady=(12,3))
        self.epp_details=ttk.Label(self.epp_tab); self.epp_details.pack(anchor="w")
    def _build_admin(self):
        ttk.Label(self.admin_tab,text="Administración",style="Title.TLabel").pack(anchor="w")
        self.admin_tabs=ttk.Notebook(self.admin_tab); self.admin_tabs.pack(fill="both",expand=True,pady=(10,0))
        people,records,settings=(ttk.Frame(self.admin_tabs,padding=12) for _ in range(3))
        for tab,text in ((people,"Personas y enrolamiento"),(records,"Registros"),(settings,"Configuración")): self.admin_tabs.add(tab,text=text)
        self.admin_tabs.bind("<<NotebookTabChanged>>",lambda _e: self.refresh_events() if self.admin_tabs.index(self.admin_tabs.select())==1 else None)
        left=ttk.Frame(people); left.pack(side="left",fill="y",padx=(0,14)); right=ttk.Frame(people); right.pack(side="left",fill="both",expand=True)
        ttk.Label(left,text="Personas registradas",font=("Segoe UI",12,"bold")).pack(anchor="w")
        self.people_list=tk.Listbox(left,width=30,height=18,exportselection=False); self.people_list.pack(fill="y",expand=True,pady=8)
        ttk.Button(left,text="Actualizar",command=self.refresh_people).pack(fill="x"); ttk.Button(left,text="Eliminar seleccionada",command=self.remove_selected).pack(fill="x",pady=6)
        self.enroll_video=self._video_panel(right,"Cámara de enrolamiento"); form=ttk.Frame(right); form.pack(fill="x",pady=10)
        ttk.Label(form,text="Nombre o identificador:").pack(side="left"); self.enroll_name=ttk.Entry(form,width=30); self.enroll_name.pack(side="left",padx=8)
        ttk.Button(form,text="Enrolar / reemplazar",command=self.start_enrollment).pack(side="left"); ttk.Button(form,text="Cancelar",command=self.cancel_enrollment).pack(side="left",padx=6)
        self.enroll_status=ttk.Label(right,text="Se capturan 8 muestras separadas en el tiempo."); self.enroll_status.pack(anchor="w")
        self._build_records(records); self._build_settings(settings); self.refresh_people()

    def _build_records(self,parent):
        f=ttk.LabelFrame(parent,text="Filtros",padding=8); f.pack(fill="x")
        self.record_person=ttk.Entry(f,width=18); self.record_type=ttk.Combobox(f,state="readonly",width=10,values=("Todos","facial","epp")); self.record_type.set("Todos")
        self.record_status=ttk.Combobox(f,state="readonly",width=18,values=("Todos","RECONOCIDO","DESCONOCIDO","SPOOF_RECHAZADO","COMPLETO","INCOMPLETO")); self.record_status.set("Todos")
        self.record_from=ttk.Entry(f,width=12); self.record_to=ttk.Entry(f,width=12)
        for i,(text,w) in enumerate((("Persona",self.record_person),("Tipo",self.record_type),("Estado",self.record_status),("Desde AAAA-MM-DD",self.record_from),("Hasta",self.record_to))): ttk.Label(f,text=text).grid(row=0,column=i,sticky="w"); w.grid(row=1,column=i,padx=(0,8))
        ttk.Button(f,text="Buscar",command=self.refresh_events).grid(row=1,column=5); ttk.Button(f,text="Limpiar",command=self.clear_filters).grid(row=1,column=6,padx=5)
        cols=("fecha","tipo","persona","estado","casco","chaleco","guantes","vida","imagen"); pane=ttk.Frame(parent); pane.pack(fill="both",expand=True,pady=10)
        self.events_table=ttk.Treeview(pane,columns=cols,show="headings",selectmode="extended")
        for c,w in zip(cols,(145,60,125,125,70,70,70,85,50)): self.events_table.heading(c,text=c.title()); self.events_table.column(c,width=w,anchor="center")
        scroll=ttk.Scrollbar(pane,command=self.events_table.yview); self.events_table.configure(yscrollcommand=scroll.set); self.events_table.pack(side="left",fill="both",expand=True); scroll.pack(side="right",fill="y")
        self.events_table.bind("<Double-1>",lambda _e:self.view_evidence()); bar=ttk.Frame(parent); bar.pack(fill="x")
        ttk.Button(bar,text="Ver imagen",command=self.view_evidence).pack(side="left")
        ttk.Button(bar,text="Seleccionar todos",command=self.select_all_events).pack(side="left",padx=(6,0))
        ttk.Button(bar,text="Borrar seleccionados",command=self.delete_selected_events).pack(side="left",padx=6)
        ttk.Button(bar,text="Borrar todos",command=self.delete_all_events).pack(side="left")
        ttk.Button(bar,text="Exportar CSV",command=self.export_events).pack(side="left",padx=6)
        self.records_count=ttk.Label(bar); self.records_count.pack(side="right"); self.refresh_events()
    def _build_settings(self,parent):
        rows=(("Cámara",self.config["camera"]),("Resolución",f'{self.config["width"]} × {self.config["height"]} a {self.config["fps"]} FPS'),("Umbral facial",self.config["facial_threshold"]),("Margen facial",self.config["facial_margin"]),("Prueba de vida","MiniFASNet V2 + V1SE" if self.config.get("anti_spoofing_enabled",True) else "Desactivada"),("Confianza EPP",self.config["epp_confidence"]))
        for i,(a,b) in enumerate(rows): ttk.Label(parent,text=a,font=("Segoe UI",10,"bold")).grid(row=i,column=0,sticky="w",padx=(0,25),pady=6); ttk.Label(parent,text=b).grid(row=i,column=1,sticky="w")
        box=ttk.LabelFrame(parent,text="EPP obligatorio",padding=10); box.grid(row=len(rows),column=0,columnspan=2,sticky="ew",pady=18); required=set(get_required_epp()); self.requirement_vars={}
        for i,(key,label) in enumerate((("casco","Casco"),("chaleco","Chaleco"),("guantes","Guantes"))): self.requirement_vars[key]=tk.BooleanVar(value=key in required); ttk.Checkbutton(box,text=label,variable=self.requirement_vars[key]).grid(row=0,column=i,padx=(0,18))
        ttk.Button(box,text="Guardar requisitos",command=self.save_requirements).grid(row=0,column=3)
        ttk.Label(parent,text="Cada evento guarda una imagen JPEG como evidencia.").grid(row=len(rows)+1,column=0,columnspan=2,sticky="w")

    def _tab_changed(self,_e=None):
        n=self.tabs.index(self.tabs.select()); self.facial.activate() if n in (0,2) else self.facial.deactivate(); self.epp.activate() if n==1 else self.epp.deactivate()
    @staticmethod
    def show(label,frame):
        image=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); image.thumbnail((900,520)); photo=ImageTk.PhotoImage(image); label.configure(image=photo,text=""); label.image=photo
    @staticmethod
    def show_live(label,frame,result,max_age):
        """Muestra la cámara fresca y superpone solamente el último análisis."""
        composed=frame.image.copy()
        if result and time.perf_counter()-result["captured"]<max_age:
            mask=result.get("annotation_mask"); overlay=result.get("preview")
            if mask is not None and overlay is not None and overlay.shape==composed.shape and mask.shape==composed.shape[:2]:
                composed[mask]=overlay[mask]
        MainWindow.show(label,composed)
    def refresh_people(self):
        self.people_list.delete(0,tk.END)
        for name in list_people(): self.people_list.insert(tk.END,name)
    def start_enrollment(self):
        name=self.enroll_name.get().strip()
        if not name or len(name)>80: messagebox.showinfo("Nombre","Escribe un nombre válido de 1 a 80 caracteres."); return
        if name in list_people() and not messagebox.askyesno("Reemplazar",f"¿Reemplazar las muestras de {name}?"): return
        try: self.facial.command("enroll",name)
        except queue.Full: messagebox.showinfo("Espera","Hay otra operación pendiente.")
    def cancel_enrollment(self):
        try: self.facial.command("cancel")
        except queue.Full: pass
    def remove_selected(self):
        selected=self.people_list.curselection()
        if not selected: messagebox.showinfo("Personas","Selecciona una persona."); return
        name=self.people_list.get(selected[0])
        if messagebox.askyesno("Eliminar",f"¿Eliminar las plantillas de {name}?"): delete_person(name); self.facial.command("reload"); self.refresh_people()
    def save_requirements(self):
        try: saved=set_required_epp([k for k,v in self.requirement_vars.items() if v.get()]); self.epp.set_required(saved); messagebox.showinfo("EPP","Requisitos guardados y aplicados.")
        except ValueError as e: messagebox.showinfo("EPP",str(e))
    def refresh_events(self):
        if not hasattr(self,"events_table"): return
        self.filtered_events=query_events(self.record_person.get(),self.record_type.get(),self.record_status.get(),self.record_from.get(),self.record_to.get(),1000); self.events_table.delete(*self.events_table.get_children())
        for r in self.filtered_events:
            try: stamp=datetime.fromisoformat(r["timestamp"]).astimezone().strftime("%Y-%m-%d %H:%M:%S")
            except ValueError: stamp=r["timestamp"]
            self.events_table.insert("","end",iid=str(r["id"]),values=(stamp,r["event_type"],r["person_name"] or "—",r["status"],r["helmet_status"] or "—",r["vest_status"] or "—",r["gloves_status"] or "—",r["liveness_status"] or "—","Sí" if r["image_path"] else "No"))
        self.records_count.configure(text=f"{len(self.filtered_events)} registros")
    def clear_filters(self):
        for w in (self.record_person,self.record_from,self.record_to): w.delete(0,tk.END)
        self.record_type.set("Todos"); self.record_status.set("Todos"); self.refresh_events()
    def selected_event(self):
        selected=self.events_table.selection()
        if not selected: messagebox.showinfo("Registros","Selecciona un registro."); return
        return next((r for r in self.filtered_events if r["id"]==int(selected[0])),None)
    def select_all_events(self):
        children=self.events_table.get_children()
        if children: self.events_table.selection_set(children)
    def delete_selected_events(self):
        selected=self.events_table.selection()
        if not selected: messagebox.showinfo("Registros","Selecciona uno o varios registros."); return
        if not messagebox.askyesno("Eliminar registros",f"¿Eliminar permanentemente {len(selected)} registro(s) y sus imágenes?"): return
        deleted=delete_events([int(value) for value in selected]); self.refresh_events()
        messagebox.showinfo("Registros",f"Se eliminaron {deleted} registro(s).")
    def delete_all_events(self):
        if not self.events_table.get_children(): messagebox.showinfo("Registros","No hay registros para eliminar."); return
        if not messagebox.askyesno("Eliminar todo","¿Eliminar permanentemente TODOS los registros y sus imágenes?"): return
        deleted=delete_events(); self.refresh_events(); messagebox.showinfo("Registros",f"Se eliminaron {deleted} registro(s).")
    def view_evidence(self):
        r=self.selected_event()
        if not r: return
        path=ROOT/r["image_path"] if r["image_path"] else None
        if not path or not path.exists(): messagebox.showerror("Evidencia","La imagen no está disponible."); return
        image=Image.open(path); image.thumbnail((1000,700)); window=tk.Toplevel(self.root); window.title(f"Evidencia #{r['id']}"); photo=ImageTk.PhotoImage(image); label=ttk.Label(window,image=photo); label.image=photo; label.pack(padx=10,pady=10)
    def export_events(self):
        if not self.filtered_events: messagebox.showinfo("Exportar","No hay registros para exportar."); return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=(("CSV","*.csv"),),initialfile="registros.csv")
        if path:
            cols=tuple(self.filtered_events[0].keys())
            with open(path,"w",newline="",encoding="utf-8-sig") as f: w=csv.writer(f,delimiter=";"); w.writerow(cols); w.writerows(tuple(r[c] for c in cols) for r in self.filtered_events)
    def _record_once(self,kind,key,cooldown,**event):
        now=time.monotonic(); state=self.event_state[kind]; state["seen"]=now
        # El mismo resultado se registra una sola vez durante una presencia
        # continua. El estado se libera cuando la persona abandona la cámara.
        if state["key"] is not None: return False
        try:
            record_event(**event); state.update(key=key,time=now)
            return True
        except Exception as error:
            self.system_status.configure(text=f"Error guardando registro: {error}")
            return False
    def _record_facial(self,r):
        live=r.get("liveness","desactivado")
        if r.get("valid") or r.get("candidate") or live in ("real","verificando","inconcluso","falso"):
            self.event_state["facial"]["seen"]=time.monotonic()
        if live=="falso": status,name="SPOOF_RECHAZADO",None
        elif r.get("identity"): status,name="RECONOCIDO",r["identity"]
        elif r.get("valid") and live in ("real","desactivado") and not r.get("candidate"): status,name="DESCONOCIDO",None
        else: return
        self._record_once("facial",(status,name),15,event_type="facial",status=status,frame=r["preview"],person_name=name,liveness_status=live,facial_score=r.get("score"),camera=self.config["camera"],details={"quality":r.get("quality"),"margin":r.get("gap")})
    def _record_epp(self,r):
        if len(r["people"])!=1: return
        current_time=time.monotonic()
        self.event_state["epp"]["seen"]=current_time
        self.voice_epp_state["seen"]=current_time
        p=r["people"][0]; decisions=p["decisions"]
        required=[decisions[name] for name in self.epp.required if name in decisions]
        # Solo se guarda una conclusión: completo si todo está confirmado, o
        # incompleto cuando al menos un elemento obligatorio está confirmado ausente.
        if required and all(value=="si" for value in required): status="COMPLETO"
        elif any(value=="no" for value in required): status="INCOMPLETO"
        else:
            self.voice_epp_state["pending"]=None
            return
        key=status
        self._record_once("epp",key,12,event_type="epp",status=status,frame=r["preview"],decisions=decisions,camera=self.config["camera"],details={"track_id":p.get("id"),"required":list(self.epp.required)})
        self._update_epp_voice(status,decisions,current_time)

    def _update_epp_voice(self,status,decisions,current_time):
        if self.voice is None: return
        missing=tuple(name for name in self.epp.required if decisions.get(name)=="no")
        voice_key=(status,missing)
        state=self.voice_epp_state
        if voice_key==state["announced"]:
            state["pending"]=None
            return
        if voice_key!=state["pending"]:
            state["pending"]=voice_key
            state["since"]=current_time
            return
        # La lógica EPP ya filtra temporalmente; esta segunda espera evita que
        # Sabina anuncie una transición causada por un único salto visual.
        if current_time-state["since"]>=1.0:
            self.voice.announce_epp(status,decisions,self.epp.required)
            state["announced"]=voice_key
            state["pending"]=None

    def tick(self):
        if self.stop.is_set(): return
        now=time.perf_counter(); selected=self.tabs.index(self.tabs.select()); self.system_status.configure(text=f"Cámara: {self.camera.status} · Facial: {self.facial.status} · EPP: {self.epp.status}")
        for state in self.event_state.values():
            if state["key"] and time.monotonic()-state["seen"]>5.0: state["key"]=None
        if (self.voice_epp_state["announced"] is not None or self.voice_epp_state["pending"] is not None) and time.monotonic()-self.voice_epp_state["seen"]>5.0:
            self.voice_epp_state.update(announced=None,pending=None,since=0.)
        fr=self.facial.latest.get(); er=self.epp.latest.get(); camera_frame=self.camera.latest.get()
        if camera_frame is not None and camera_frame.sequence!=self.last_camera_sequence:
            self.last_camera_sequence=camera_frame.sequence
            if selected==0: self.show_live(self.facial_video,camera_frame,fr,.55)
            elif selected==1: self.show_live(self.epp_video,camera_frame,er,.75)
            else: self.show_live(self.enroll_video,camera_frame,fr,.55)
        if selected in (0,2) and fr and now-fr["captured"]<.7:
            key="facial" if selected==0 else "admin"; target=self.facial_video if selected==0 else self.enroll_video
            if fr["sequence"]!=self.last_sequences[key]: self.last_sequences[key]=fr["sequence"]; self._record_facial(fr) if selected==0 else None
            live=fr.get("liveness","desactivado")
            if selected==0:
                title=("Prueba de vida rechazada: posible foto o pantalla" if live=="falso" else "Verificando que sea una persona real..." if live in ("inconcluso","verificando") else f"Persona reconocida: {fr['identity']}" if fr["identity"] else "Verificando identidad..." if fr["candidate"] else "Persona no registrada" if fr["valid"] else fr["quality"])
                self.facial_result.configure(text=title); self.facial_details.configure(text=f"Prueba de vida: {live} ({fr.get('liveness_score',0):.3f}) · Similitud: {fr['score']:.3f} · Margen: {fr['gap']:.3f} · Análisis: {fr['inference_ms']:.0f} ms")
            else:
                self.enroll_status.configure(text=self.facial.status+" · "+live+" · "+fr["quality"])
                if self.was_enrolling and not self.facial.enrolling: self.refresh_people()
                self.was_enrolling=self.facial.enrolling
        if selected==1 and er and now-er["captured"]<1:
            if er["sequence"]!=self.last_sequences["epp"]: self.last_sequences["epp"]=er["sequence"]; self._record_epp(er)
            people=er["people"]
            if not people: self.epp_result.configure(text="Buscando una persona..."); self.epp_details.configure(text="")
            elif len(people)>1: self.epp_result.configure(text="Debe aparecer una sola persona"); self.epp_details.configure(text=f"Personas detectadas: {len(people)}")
            else:
                p=people[0]; self.epp_result.configure(text=f"Estado EPP: {p['overall']}"); self.epp_details.configure(text=" · ".join(f"{k.title()}: {v}" for k,v in p["decisions"].items())+f" · Análisis: {er['inference_ms']:.0f} ms")
        self.root.after(25,self.tick)
    def close(self): self.stop.set(); self.root.after(100,self.root.destroy)
