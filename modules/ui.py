import os
import webbrowser
import customtkinter as ctk
from typing import Callable, Tuple
import cv2
from modules.gpu_processing import gpu_cvt_color, gpu_resize, gpu_flip
from PIL import Image, ImageOps
import time
import json
import queue
import threading
import numpy as np
import requests
import tempfile
import modules.globals
import modules.metadata
from modules.face_analyser import (
    get_one_face,
    get_many_faces,
    detect_one_face_fast,
    detect_many_faces_fast,
    get_unique_faces_from_target_image,
    get_unique_faces_from_target_video,
    add_blank_map,
    has_valid_map,
    simplify_maps,
)
from modules.capturer import get_video_frame, get_video_frame_total
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    is_image,
    is_video,
    resolve_relative_path,
    has_image_extension,
)
from modules.video_capture import VideoCapturer
from modules.gettext import LanguageManager
from modules.ui_tooltip import ToolTip
from modules import globals
import platform

if platform.system() == "Windows":
    from pygrabber.dshow_graph import FilterGraph

# --- Tk 9.0 compatibility patch ---
# In Tk 9.0, Menu.index("end") returns "" instead of raising TclError
# when the menu is empty. CustomTkinter's CTkOptionMenu doesn't handle
# this, causing crashes. This patch adds the missing guard.
try:
    from customtkinter.windows.widgets.core_widget_classes import DropdownMenu as _DropdownMenu

    _original_add_menu_commands = _DropdownMenu._add_menu_commands

    def _patched_add_menu_commands(self, *args, **kwargs):
        try:
            end_index = self._menu.index("end")
            if end_index == "" or end_index is None:
                return
        except Exception:
            pass
        _original_add_menu_commands(self, *args, **kwargs)

    _DropdownMenu._add_menu_commands = _patched_add_menu_commands
except (ImportError, AttributeError):
    pass  # CustomTkinter version doesn't have this class path
# --- End Tk 9.0 patch ---

ROOT = None
POPUP = None
POPUP_LIVE = None
ROOT_HEIGHT = 720
ROOT_WIDTH = 700

PREVIEW = None
PREVIEW_MAX_HEIGHT = 700
PREVIEW_MAX_WIDTH = 1200
PREVIEW_DEFAULT_WIDTH = 640
PREVIEW_DEFAULT_HEIGHT = 360

POPUP_WIDTH = 750
POPUP_HEIGHT = 810
POPUP_SCROLL_WIDTH = (740,)
POPUP_SCROLL_HEIGHT = 700

POPUP_LIVE_WIDTH = 900
POPUP_LIVE_HEIGHT = 820
POPUP_LIVE_SCROLL_WIDTH = (890,)
POPUP_LIVE_SCROLL_HEIGHT = 700

MAPPER_PREVIEW_MAX_HEIGHT = 100
MAPPER_PREVIEW_MAX_WIDTH = 100

DEFAULT_BUTTON_WIDTH = 200
DEFAULT_BUTTON_HEIGHT = 40

RECENT_DIRECTORY_SOURCE = None
RECENT_DIRECTORY_TARGET = None
RECENT_DIRECTORY_OUTPUT = None

_ = None
preview_label = None
preview_slider = None
source_label = None
target_label = None
status_label = None
popup_status_label = None
popup_status_label_live = None
source_label_dict = {}
source_label_dict_live = {}
target_label_dict_live = {}

img_ft, vid_ft = modules.globals.file_types


def init(start: Callable[[], None], destroy: Callable[[], None], lang: str) -> ctk.CTk:
    global ROOT, PREVIEW, _

    lang_manager = LanguageManager(lang)
    _ = lang_manager._
    ROOT = create_root(start, destroy)
    PREVIEW = create_preview(ROOT)

    return ROOT


def save_switch_states():
    switch_states = {
        "keep_fps": modules.globals.keep_fps,
        "keep_audio": modules.globals.keep_audio,
        "keep_frames": modules.globals.keep_frames,
        "many_faces": modules.globals.many_faces,
        "map_faces": modules.globals.map_faces,
        "poisson_blend": modules.globals.poisson_blend,
        "color_correction": modules.globals.color_correction,
        "nsfw_filter": modules.globals.nsfw_filter,
        "live_mirror": modules.globals.live_mirror,
        "live_resizable": modules.globals.live_resizable,
        "fp_ui": modules.globals.fp_ui,
        "show_fps": modules.globals.show_fps,
        "mouth_mask": modules.globals.mouth_mask,
        "show_mouth_mask_box": modules.globals.show_mouth_mask_box,
        "mouth_mask_size": modules.globals.mouth_mask_size,
    }
    with open("switch_states.json", "w") as f:
        json.dump(switch_states, f)


def load_switch_states():
    try:
        with open("switch_states.json", "r") as f:
            switch_states = json.load(f)
        modules.globals.keep_fps = switch_states.get("keep_fps", True)
        modules.globals.keep_audio = switch_states.get("keep_audio", True)
        modules.globals.keep_frames = switch_states.get("keep_frames", False)
        modules.globals.many_faces = switch_states.get("many_faces", False)
        modules.globals.map_faces = switch_states.get("map_faces", False)
        modules.globals.poisson_blend = switch_states.get("poisson_blend", False)
        modules.globals.color_correction = switch_states.get("color_correction", False)
        modules.globals.nsfw_filter = switch_states.get("nsfw_filter", False)
        modules.globals.live_mirror = switch_states.get("live_mirror", False)
        modules.globals.live_resizable = switch_states.get("live_resizable", False)
        modules.globals.fp_ui = switch_states.get("fp_ui", {"face_enhancer": False})
        modules.globals.show_fps = switch_states.get("show_fps", False)
        modules.globals.mouth_mask_size = switch_states.get("mouth_mask_size", 0.0)
        # mouth_mask is driven by the slider: on if size > 0, off if 0
        modules.globals.mouth_mask = modules.globals.mouth_mask_size > 0
        modules.globals.show_mouth_mask_box = False  # always start hidden
    except FileNotFoundError:
        # If the file doesn't exist, use default values
        pass


def create_root(start: Callable[[], None], destroy: Callable[[], None]) -> ctk.CTk:  # noqa: C901
    global source_label, target_label, status_label, show_fps_switch

    load_switch_states()

    ctk.deactivate_automatic_dpi_awareness()
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme(resolve_relative_path("ui.json"))

    root = ctk.CTk()
    root.minsize(ROOT_WIDTH, ROOT_HEIGHT)
    root.resizable(True, True)
    root.title("Persona — AI Face Swap")
    root.protocol("WM_DELETE_WINDOW", lambda: destroy())

    # Centre the window on screen at launch
    root.update_idletasks()
    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    cx = (sw - ROOT_WIDTH)  // 2
    cy = max(0, (sh - ROOT_HEIGHT) // 2 - 30)
    root.geometry(f"{ROOT_WIDTH}x{ROOT_HEIGHT}+{cx}+{cy}")

    # ── Design tokens ─────────────────────────────────────────
    ACCENT      = ("#007AFF", "#0A84FF")
    GREEN       = ("#28A745", "#30D158")
    RED         = ("#DC3545", "#FF453A")
    RED_HOVER   = ("#B02A37", "#E03028")
    SLOT_BG     = ("#EBEBF0", "#2C2C2E")
    SLOT_BORDER = ("#C7C7CC", "#48484A")
    SEC_TXT     = ("#6C6C70", "#8E8E93")
    GHOST_FG    = ("gray80",  "gray30")
    GHOST_HV    = ("gray70",  "gray40")
    GHOST_TXT   = ("gray15",  "gray90")

    # ── Status bar (packed FIRST so it anchors to bottom) ─────
    status_bar = ctk.CTkFrame(root, fg_color=("gray86", "gray16"),
                               corner_radius=0, height=26)
    status_bar.pack(side="bottom", fill="x")
    status_bar.pack_propagate(False)

    global status_label
    status_label = ctk.CTkLabel(
        status_bar, text="Ready",
        font=ctk.CTkFont(size=11), text_color=SEC_TXT, anchor="w",
    )
    status_label.pack(side="left", padx=12)

    _dl = ctk.CTkLabel(
        status_bar, text="deeplivecam.net",
        font=ctk.CTkFont(size=11), text_color=ACCENT, cursor="hand2",
    )
    _dl.pack(side="right", padx=12)
    _dl.bind("<Button>", lambda e: webbrowser.open("https://deeplivecam.net"))

    # ── Main content — canvas fills left, scrollbar right when needed ──
    # pack(side="left") + pack(side="right") is the only tkinter idiom where
    # the scrollbar truly vacates its space when hidden — no reserved columns,
    # no asymmetric gaps.
    import tkinter as tk

    MAX_CONTENT_W = 700
    PAD = 12

    def _canvas_bg():
        return "#1C1C1E" if ctk.get_appearance_mode() == "Dark" else "#F2F2F7"

    # Wrapper — plain tk.Frame so children get exact dimensions (no CTk offset)
    _wrap = tk.Frame(root, bg=_canvas_bg())
    _wrap.pack(fill="both", expand=True)

    # Scrollbar packed BEFORE canvas so side="right" reserves space first
    _vscroll = ctk.CTkScrollbar(_wrap, command=lambda *a: _canvas.yview(*a))
    _vscroll_shown = False

    # Canvas fills whatever is left after the (optional) scrollbar
    _canvas = tk.Canvas(_wrap, highlightthickness=0, bd=0, bg=_canvas_bg())
    _canvas.pack(side="left", fill="both", expand=True)
    _canvas.configure(yscrollcommand=lambda f, l: (
        _vscroll.set(f, l), root.after(30, _update_scrollbar)))

    # Inner content frame — plain tk.Frame, zero internal padding
    main = tk.Frame(_canvas, bg=_canvas_bg())
    _win_id = _canvas.create_window(0, 0, window=main, anchor="nw")

    def _fit_inner_width(event):
        _canvas.itemconfig(_win_id, width=event.width)
        root.after(5, _rebalance)

    def _update_scrollbar(event=None):
        nonlocal _vscroll_shown
        _canvas.configure(scrollregion=_canvas.bbox("all"))
        try:
            first, last = _canvas.yview()
            needs = not (first <= 0.001 and last >= 0.999)
            if needs and not _vscroll_shown:
                _vscroll.pack(side="right", fill="y")
                _vscroll_shown = True
            elif not needs and _vscroll_shown:
                _vscroll.pack_forget()
                _vscroll_shown = False
        except Exception:
            pass

    def _on_mousewheel(event):
        try:
            first, last = _canvas.yview()
            if not (first <= 0.001 and last >= 0.999):
                _canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        except Exception:
            pass

    _canvas.bind("<Configure>", _fit_inner_width)
    root.bind_all("<MouseWheel>", _on_mousewheel)

    def _sync_canvas_bg(v=None):
        bg = _canvas_bg()
        _canvas.configure(bg=bg)
        _wrap.configure(bg=bg)
        main.configure(bg=bg)

    # ── Centered content container with max-width ─────────────
    ctr = ctk.CTkFrame(main, fg_color="transparent", corner_radius=0)
    ctr.pack(fill="x", padx=PAD)

    def _rebalance(event=None):
        try:
            avail = _canvas.winfo_width()
            if avail > MAX_CONTENT_W + PAD * 2:
                side = (avail - MAX_CONTENT_W) // 2
                ctr.pack_configure(padx=side)
            else:
                ctr.pack_configure(padx=PAD)
        except Exception:
            pass

    root.bind("<Configure>", lambda e: root.after(10, _rebalance))

    # ── Helpers ───────────────────────────────────────────────
    def card(title: str | None = None) -> ctk.CTkFrame:
        frm = ctk.CTkFrame(ctr, corner_radius=10)
        frm.pack(fill="x", pady=(0, 7))
        if title:
            ctk.CTkLabel(frm, text=title,
                         font=ctk.CTkFont(size=10, weight="bold"),
                         text_color=SEC_TXT, anchor="w",
                         ).pack(fill="x", padx=PAD, pady=(9, 3))
        return frm

    def make_opt_switch(parent, row, col, label, var_name, default):
        bv = ctk.BooleanVar(value=getattr(modules.globals, var_name, default))
        sw = ctk.CTkSwitch(
            parent, text=label, variable=bv, cursor="hand2",
            command=lambda: (setattr(modules.globals, var_name, bv.get()),
                             save_switch_states()),
        )
        sw.grid(row=row, column=col, sticky="w", padx=PAD, pady=3)
        return bv, sw

    # ═══════════════════════════════════════════════════════════
    # HEADER — single compact row
    # ═══════════════════════════════════════════════════════════
    hdr = ctk.CTkFrame(ctr, fg_color="transparent", height=42)
    hdr.pack(fill="x", pady=(10, 6))
    hdr.pack_propagate(False)

    ctk.CTkLabel(hdr, text="Persona",
                 font=ctk.CTkFont(size=20, weight="bold"),
                 anchor="w").pack(side="left")

    _mode_var = ctk.StringVar(value=ctk.get_appearance_mode().capitalize())

    def _set_mode(v):
        ctk.set_appearance_mode(v.lower())
        root.after(50, _sync_canvas_bg)

    ctk.CTkSegmentedButton(
        hdr, values=["Light", "Dark"],
        variable=_mode_var,
        command=_set_mode,
        width=116, height=26,
        font=ctk.CTkFont(size=11),
    ).pack(side="right", anchor="center")

    # ═══════════════════════════════════════════════════════════
    # MEDIA CARD — full-width split slots
    # ═══════════════════════════════════════════════════════════
    mc = ctk.CTkFrame(ctr, corner_radius=10)
    mc.pack(fill="x", pady=(0, 7))

    mi = ctk.CTkFrame(mc, fg_color="transparent")
    mi.pack(fill="x", padx=PAD, pady=(9, 11))
    mi.columnconfigure(0, weight=1)
    mi.columnconfigure(1, weight=0, minsize=42)
    mi.columnconfigure(2, weight=1)

    def _make_slot(parent, lbl_text, ph_text):
        """Build a labelled image-slot column; returns (container, slot_frame, label_widget)."""
        col = ctk.CTkFrame(parent, fg_color="transparent")
        ctk.CTkLabel(col, text=lbl_text,
                     font=ctk.CTkFont(size=10, weight="bold"),
                     text_color=SEC_TXT, anchor="w").pack(fill="x", pady=(0, 3))
        sl = ctk.CTkFrame(col, corner_radius=8, fg_color=SLOT_BG,
                          border_width=1, border_color=SLOT_BORDER, height=158)
        sl.pack(fill="x")
        sl.pack_propagate(False)
        lw = ctk.CTkLabel(sl, text=ph_text, font=ctk.CTkFont(size=11),
                          text_color=SEC_TXT, fg_color="transparent")
        lw.place(relx=0, rely=0, relwidth=1, relheight=1)
        return col, sl, lw

    src_col, _src_slot, source_label = _make_slot(mi, "Source", "No image selected")
    src_col.grid(row=0, column=0, sticky="nsew", padx=(0, 3))

    src_btns = ctk.CTkFrame(src_col, fg_color="transparent")
    src_btns.pack(fill="x", pady=(4, 0))
    src_btns.columnconfigure(0, weight=1)

    select_face_button = ctk.CTkButton(
        src_btns, text="Select Face", cursor="hand2", height=30,
        command=lambda: select_source_path())
    select_face_button.grid(row=0, column=0, sticky="ew", padx=(0, 4))
    ToolTip(select_face_button, _("Choose the source face image"))

    random_face_button = ctk.CTkButton(
        src_btns, text="Random", cursor="hand2", height=30, width=66,
        fg_color=GHOST_FG, hover_color=GHOST_HV, text_color=GHOST_TXT,
        command=lambda: fetch_random_face())
    random_face_button.grid(row=0, column=1)
    ToolTip(random_face_button, _("Random face from thispersondoesnotexist.com"))

    # Swap button (centre column)
    xc = ctk.CTkFrame(mi, fg_color="transparent", width=42)
    xc.grid(row=0, column=1, sticky="ns")
    xc.pack_propagate(False)
    xc.grid_propagate(False)
    swap_faces_button = ctk.CTkButton(
        xc, text="⇆", cursor="hand2", width=30, height=30,
        fg_color=GHOST_FG, hover_color=GHOST_HV, text_color=GHOST_TXT,
        font=ctk.CTkFont(size=14),
        command=lambda: swap_faces_paths())
    swap_faces_button.place(relx=0.5, rely=0.37, anchor="center")
    ToolTip(swap_faces_button, _("Swap source and target"))

    tgt_col, _tgt_slot, target_label = _make_slot(mi, "Target", "No image or video selected")
    tgt_col.grid(row=0, column=2, sticky="nsew", padx=(3, 0))

    select_target_button = ctk.CTkButton(
        tgt_col, text="Select Target", cursor="hand2", height=30,
        command=lambda: select_target_path())
    select_target_button.pack(fill="x", pady=(4, 0))
    ToolTip(select_target_button, _("Choose target image or video"))

    # ═══════════════════════════════════════════════════════════
    # OPTIONS — tight 2-col grid
    # ═══════════════════════════════════════════════════════════
    oc = card("OPTIONS")
    og = ctk.CTkFrame(oc, fg_color="transparent")
    og.pack(fill="x", padx=PAD, pady=(0, 8))
    og.columnconfigure(0, weight=1)
    og.columnconfigure(1, weight=1)

    keep_fps_value,         _s0 = make_opt_switch(og, 0, 0, _("Keep FPS"),      "keep_fps",         True)
    keep_audio_value,       _s1 = make_opt_switch(og, 0, 1, _("Keep Audio"),    "keep_audio",       True)
    keep_frames_value,      _s2 = make_opt_switch(og, 1, 0, _("Keep Frames"),   "keep_frames",      False)
    many_faces_value,       _s3 = make_opt_switch(og, 1, 1, _("Many Faces"),    "many_faces",       False)
    show_fps_value, show_fps_switch = make_opt_switch(og, 2, 0, _("Show FPS"),  "show_fps",         False)
    color_correction_value, _s5 = make_opt_switch(og, 2, 1, _("Fix Blue Cam"), "color_correction", False)

    map_faces_var = ctk.BooleanVar(value=modules.globals.map_faces)
    _mfsw = ctk.CTkSwitch(og, text=_("Map Faces"), variable=map_faces_var, cursor="hand2",
                           command=lambda: (setattr(modules.globals, "map_faces", map_faces_var.get()),
                                            save_switch_states(),
                                            close_mapper_window() if not map_faces_var.get() else None))
    _mfsw.grid(row=3, column=0, sticky="w", padx=PAD, pady=3)

    poisson_blend_value = ctk.BooleanVar(value=modules.globals.poisson_blend)
    _pbsw = ctk.CTkSwitch(og, text=_("Poisson Blend"), variable=poisson_blend_value, cursor="hand2",
                           command=lambda: (setattr(modules.globals, "poisson_blend",
                                                    poisson_blend_value.get()),
                                            save_switch_states()))
    _pbsw.grid(row=3, column=1, sticky="w", padx=PAD, pady=3)

    # ═══════════════════════════════════════════════════════════
    # ENHANCEMENT — model full row, then 2×2 slider grid
    # ═══════════════════════════════════════════════════════════
    ec = card("ENHANCEMENT")
    ei = ctk.CTkFrame(ec, fg_color="transparent")
    ei.pack(fill="x", padx=PAD, pady=(0, 10))

    # Model row
    mr = ctk.CTkFrame(ei, fg_color="transparent")
    mr.pack(fill="x", pady=(0, 6))
    ctk.CTkLabel(mr, text="Model", anchor="w",
                 width=82, font=ctk.CTkFont(size=13)).pack(side="left")

    enhancer_options = ["None", "GFPGAN", "GPEN-512", "GPEN-256"]
    enhancer_key_map = {"None": None, "GFPGAN": "face_enhancer",
                        "GPEN-512": "face_enhancer_gpen512", "GPEN-256": "face_enhancer_gpen256"}
    initial_enhancer = "None"
    if modules.globals.fp_ui.get("face_enhancer", False):            initial_enhancer = "GFPGAN"
    elif modules.globals.fp_ui.get("face_enhancer_gpen512", False):  initial_enhancer = "GPEN-512"
    elif modules.globals.fp_ui.get("face_enhancer_gpen256", False):  initial_enhancer = "GPEN-256"
    enhancer_variable = ctk.StringVar(value=initial_enhancer)

    def on_enhancer_change(choice: str):
        for k in ["face_enhancer", "face_enhancer_gpen256", "face_enhancer_gpen512"]:
            update_tumbler(k, False)
        sk = enhancer_key_map.get(choice)
        if sk:
            update_tumbler(sk, True)
        save_switch_states()

    ctk.CTkOptionMenu(mr, variable=enhancer_variable, values=enhancer_options,
                      command=on_enhancer_change, height=30,
                      ).pack(side="left", fill="x", expand=True)

    # Opacity — full-width row (most-used, gets the most space)
    transparency_var = ctk.DoubleVar(value=modules.globals.opacity)

    def on_transparency_change(value: float):
        val = float(value)
        modules.globals.opacity = val
        pct = int(val * 100)
        modules.globals.face_swapper_enabled = pct > 0
        update_status(f"Opacity: {pct}%" if pct > 0 else "Face swap disabled (0%)")

    opr = ctk.CTkFrame(ei, fg_color="transparent")
    opr.pack(fill="x", pady=(0, 4))
    ctk.CTkLabel(opr, text="Opacity", anchor="w",
                 width=82, font=ctk.CTkFont(size=13)).pack(side="left")
    ctk.CTkSlider(opr, from_=0.0, to=1.0, variable=transparency_var,
                  command=on_transparency_change, height=16,
                  ).pack(side="left", fill="x", expand=True)

    # Sharpness + Mouth Mask — side-by-side (2-col)
    sg = ctk.CTkFrame(ei, fg_color="transparent")
    sg.pack(fill="x")
    sg.columnconfigure(0, weight=1)
    sg.columnconfigure(1, weight=1)

    sharpness_var = ctk.DoubleVar(value=modules.globals.sharpness)

    def on_sharpness_change(value: float):
        modules.globals.sharpness = float(value)

    shr = ctk.CTkFrame(sg, fg_color="transparent")
    shr.grid(row=0, column=0, sticky="ew", padx=(0, 8))
    ctk.CTkLabel(shr, text="Sharpness", anchor="w",
                 width=72, font=ctk.CTkFont(size=13)).pack(side="left")
    ctk.CTkSlider(shr, from_=0.0, to=5.0, variable=sharpness_var,
                  command=on_sharpness_change, height=16,
                  ).pack(side="left", fill="x", expand=True)

    mouth_mask_var = ctk.BooleanVar(value=modules.globals.mouth_mask)
    show_mouth_mask_box_var = ctk.BooleanVar(value=modules.globals.show_mouth_mask_box)
    mouth_mask_size_var = ctk.DoubleVar(value=modules.globals.mouth_mask_size)

    def on_mouth_mask_size_change(value: float):
        val = float(value)
        modules.globals.mouth_mask_size = val
        if val > 0:
            modules.globals.mouth_mask = True
            mouth_mask_var.set(True)
        else:
            modules.globals.mouth_mask = False
            mouth_mask_var.set(False)
            modules.globals.show_mouth_mask_box = False

    def on_mouth_mask_slider_release(event):
        modules.globals.show_mouth_mask_box = False

    def on_mouth_mask_slider_press(event):
        if modules.globals.mouth_mask_size > 0:
            modules.globals.show_mouth_mask_box = True

    mmr = ctk.CTkFrame(sg, fg_color="transparent")
    mmr.grid(row=0, column=1, sticky="ew")
    ctk.CTkLabel(mmr, text="Mouth Mask", anchor="w",
                 width=82, font=ctk.CTkFont(size=13)).pack(side="left")
    mm_sl = ctk.CTkSlider(mmr, from_=0.0, to=100.0, variable=mouth_mask_size_var,
                          command=on_mouth_mask_size_change, height=16)
    mm_sl.pack(side="left", fill="x", expand=True)
    mm_sl.bind("<ButtonPress-1>",   on_mouth_mask_slider_press)
    mm_sl.bind("<ButtonRelease-1>", on_mouth_mask_slider_release)

    # ═══════════════════════════════════════════════════════════
    # CAMERA + ACTIONS — single merged bar (no card overhead)
    # ═══════════════════════════════════════════════════════════
    ba = ctk.CTkFrame(ctr, corner_radius=10)
    ba.pack(fill="x", pady=(0, 7))

    bar = ctk.CTkFrame(ba, fg_color="transparent")
    bar.pack(fill="x", padx=PAD, pady=9)
    bar.columnconfigure(0, weight=1)   # camera dropdown expands

    available_cameras = get_available_cameras()
    camera_indices, camera_names = available_cameras
    no_cam = not camera_names or camera_names[0] == "No cameras found"

    camera_variable = ctk.StringVar(value="No camera" if no_cam
                                    else camera_names[0])
    ctk.CTkOptionMenu(
        bar,
        variable=camera_variable,
        values=["No camera"] if no_cam else camera_names,
        state="disabled" if no_cam else "normal",
        height=32,
    ).grid(row=0, column=0, sticky="ew", padx=(0, 6))

    live_button = ctk.CTkButton(
        bar, text="Go Live", cursor="hand2",
        height=32, width=78,
        fg_color=GREEN, hover_color=("#1E7B36", "#27AE50"),
        text_color=("#FFFFFF", "#FFFFFF"),
        font=ctk.CTkFont(size=12, weight="bold"),
        command=lambda: webcam_preview(
            root,
            camera_indices[camera_names.index(camera_variable.get())]
            if not no_cam else None),
        state="disabled" if no_cam else "normal",
    )
    live_button.grid(row=0, column=1, padx=(0, 10))
    ToolTip(live_button, _("Start real-time webcam face swap"))

    # thin divider
    ctk.CTkFrame(bar, fg_color=("gray75", "gray32"),
                 width=1, height=28).grid(row=0, column=2, padx=(0, 10))

    start_button = ctk.CTkButton(
        bar, text="Start", cursor="hand2",
        height=32, width=84,
        font=ctk.CTkFont(size=13, weight="bold"),
        command=lambda: analyze_target(start, root))
    start_button.grid(row=0, column=3, padx=(0, 5))
    ToolTip(start_button, _("Process target with selected face"))

    preview_button = ctk.CTkButton(
        bar, text="Preview", cursor="hand2",
        height=32, width=84,
        fg_color=GHOST_FG, hover_color=GHOST_HV, text_color=GHOST_TXT,
        font=ctk.CTkFont(size=12),
        command=lambda: toggle_preview())
    preview_button.grid(row=0, column=4, padx=(0, 5))
    ToolTip(preview_button, _("Toggle output preview"))

    stop_button = ctk.CTkButton(
        bar, text="Stop", cursor="hand2",
        height=32, width=72,
        fg_color=RED, hover_color=RED_HOVER,
        font=ctk.CTkFont(size=13, weight="bold"),
        command=lambda: destroy())
    stop_button.grid(row=0, column=5)
    ToolTip(stop_button, _("Stop and quit"))

    return root


def close_mapper_window():
    global POPUP, POPUP_LIVE
    if POPUP and POPUP.winfo_exists():
        POPUP.destroy()
        POPUP = None
    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        POPUP_LIVE.destroy()
        POPUP_LIVE = None


def analyze_target(start: Callable[[], None], root: ctk.CTk):
    if POPUP != None and POPUP.winfo_exists():
        update_status("Please complete pop-up or close it.")
        return

    if modules.globals.map_faces:
        modules.globals.source_target_map = []

        if is_image(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_image()
        elif is_video(modules.globals.target_path):
            update_status("Getting unique faces")
            get_unique_faces_from_target_video()

        if len(modules.globals.source_target_map) > 0:
            create_source_target_popup(start, root, modules.globals.source_target_map)
        else:
            update_status("No faces found in target")
    else:
        select_output_path(start)


def create_source_target_popup(
        start: Callable[[], None], root: ctk.CTk, map: list
) -> None:
    global POPUP, popup_status_label

    POPUP = ctk.CTkToplevel(root)
    POPUP.title(_("Source x Target Mapper"))
    POPUP.geometry(f"{POPUP_WIDTH}x{POPUP_HEIGHT}")
    POPUP.focus()

    def on_submit_click(start):
        if has_valid_map():
            POPUP.destroy()
            select_output_path(start)
        else:
            update_pop_status("Atleast 1 source with target is required!")

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP, width=POPUP_SCROLL_WIDTH, height=POPUP_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_button_click(map, button_num):
        map = update_popup_source(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_button_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=50, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        image = Image.fromarray(gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB))
        image = image.resize(
            (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        tk_image = ctk.CTkImage(image, size=image.size)

        target_image = ctk.CTkLabel(
            scrollable_frame,
            text=f"T-{id}",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        target_image.grid(row=id, column=3, padx=10, pady=10)
        target_image.configure(image=tk_image)

    popup_status_label = ctk.CTkLabel(POPUP, text=None, justify="center")
    popup_status_label.grid(row=1, column=0, pady=15)

    close_button = ctk.CTkButton(
        POPUP, text=_("Submit"), command=lambda: on_submit_click(start)
    )
    close_button.grid(row=2, column=0, pady=10)


def update_popup_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict[button_num].destroy()
        del source_label_dict[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict[button_num] = source_image
        else:
            update_pop_status("Face could not be detected in last upload!")
        return map


def create_preview(parent: ctk.CTkToplevel) -> ctk.CTkToplevel:
    global preview_label, preview_slider

    preview = ctk.CTkToplevel(parent)
    preview.withdraw()
    preview.title(_("Preview"))
    preview.configure()
    preview.protocol("WM_DELETE_WINDOW", lambda: toggle_preview())
    preview.resizable(width=True, height=True)

    preview_label = ctk.CTkLabel(preview, text=None)
    preview_label.pack(fill="both", expand=True)

    preview_slider = ctk.CTkSlider(
        preview, from_=0, to=0, command=lambda frame_value: update_preview(frame_value)
    )

    return preview


def update_status(text: str) -> None:
    status_label.configure(text=_(text))
    ROOT.update()


def update_pop_status(text: str) -> None:
    popup_status_label.configure(text=_(text))


def update_pop_live_status(text: str) -> None:
    popup_status_label_live.configure(text=_(text))


def update_tumbler(var: str, value: bool) -> None:
    modules.globals.fp_ui[var] = value
    save_switch_states()
    # If we're currently in a live preview, update the frame processors
    if PREVIEW.state() == "normal":
        global frame_processors
        frame_processors = get_frame_processors_modules(
            modules.globals.frame_processors
        )


def fetch_random_face() -> None:
    PREVIEW.withdraw()
    try:
        response = requests.get(
            "https://thispersondoesnotexist.com/",
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=10,
        )
        response.raise_for_status()
        temp_dir = tempfile.gettempdir()
        temp_path = os.path.join(temp_dir, "deep_live_cam_random_face.jpg")
        with open(temp_path, "wb") as f:
            f.write(response.content)
        modules.globals.source_path = temp_path
        image = render_image_preview(temp_path, (200, 200))
        source_label.configure(image=image, text="")
    except Exception as e:
        print(f"Failed to fetch random face: {e}")


def select_source_path() -> None:
    global RECENT_DIRECTORY_SOURCE, img_ft, vid_ft

    PREVIEW.withdraw()
    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )
    if is_image(source_path):
        modules.globals.source_path = source_path
        RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
        image = render_image_preview(modules.globals.source_path, (200, 200))
        source_label.configure(image=image, text="")
    else:
        modules.globals.source_path = None
        source_label.configure(image=None, text="No image selected")


def swap_faces_paths() -> None:
    global RECENT_DIRECTORY_SOURCE, RECENT_DIRECTORY_TARGET

    source_path = modules.globals.source_path
    target_path = modules.globals.target_path

    if not is_image(source_path) or not is_image(target_path):
        return

    modules.globals.source_path = target_path
    modules.globals.target_path = source_path

    RECENT_DIRECTORY_SOURCE = os.path.dirname(modules.globals.source_path)
    RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)

    PREVIEW.withdraw()

    source_image = render_image_preview(modules.globals.source_path, (200, 200))
    source_label.configure(image=source_image, text="")

    target_image = render_image_preview(modules.globals.target_path, (200, 200))
    target_label.configure(image=target_image, text="")


def select_target_path() -> None:
    global RECENT_DIRECTORY_TARGET, img_ft, vid_ft

    PREVIEW.withdraw()
    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image or video"),
        initialdir=RECENT_DIRECTORY_TARGET,
        filetypes=[img_ft, vid_ft],
    )
    if is_image(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        image = render_image_preview(modules.globals.target_path, (200, 200))
        target_label.configure(image=image, text="")
    elif is_video(target_path):
        modules.globals.target_path = target_path
        RECENT_DIRECTORY_TARGET = os.path.dirname(modules.globals.target_path)
        video_frame = render_video_preview(target_path, (200, 200))
        target_label.configure(image=video_frame, text="")
    else:
        modules.globals.target_path = None
        target_label.configure(image=None, text="No image or video selected")


def select_output_path(start: Callable[[], None]) -> None:
    global RECENT_DIRECTORY_OUTPUT, img_ft, vid_ft

    if is_image(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save image output file"),
            filetypes=[img_ft],
            defaultextension=".png",
            initialfile="output.png",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    elif is_video(modules.globals.target_path):
        output_path = ctk.filedialog.asksaveasfilename(
            title=_("save video output file"),
            filetypes=[vid_ft],
            defaultextension=".mp4",
            initialfile="output.mp4",
            initialdir=RECENT_DIRECTORY_OUTPUT,
        )
    else:
        output_path = None
    if output_path:
        modules.globals.output_path = output_path
        RECENT_DIRECTORY_OUTPUT = os.path.dirname(modules.globals.output_path)
        start()


def check_and_ignore_nsfw(target, destroy: Callable = None) -> bool:
    """Check if the target is NSFW.
    TODO: Consider to make blur the target.
    """
    from numpy import ndarray
    from modules.predicter import predict_image, predict_video, predict_frame

    if type(target) is str:  # image/video file path
        check_nsfw = predict_image if has_image_extension(target) else predict_video
    elif type(target) is ndarray:  # frame object
        check_nsfw = predict_frame
    if check_nsfw and check_nsfw(target):
        if destroy:
            destroy(
                to_quit=False
            )  # Do not need to destroy the window frame if the target is NSFW
        update_status("Processing ignored!")
        return True
    else:
        return False


def fit_image_to_size(image, width: int, height: int):
    if width is None and height is None:
        return image
    h, w, _ = image.shape
    ratio_h = 0.0
    ratio_w = 0.0
    if width > height:
        ratio_h = height / h
    else:
        ratio_w = width / w
    ratio = max(ratio_w, ratio_h)
    new_size = (int(ratio * w), int(ratio * h))
    return gpu_resize(image, dsize=new_size)


def render_image_preview(image_path: str, size: Tuple[int, int]) -> ctk.CTkImage:
    image = Image.open(image_path)
    if size:
        image = ImageOps.fit(image, size, Image.LANCZOS)
    return ctk.CTkImage(image, size=image.size)


def render_video_preview(
        video_path: str, size: Tuple[int, int], frame_number: int = 0
) -> ctk.CTkImage:
    capture = cv2.VideoCapture(video_path)
    if frame_number:
        capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    has_frame, frame = capture.read()
    if has_frame:
        image = Image.fromarray(gpu_cvt_color(frame, cv2.COLOR_BGR2RGB))
        if size:
            image = ImageOps.fit(image, size, Image.LANCZOS)
        return ctk.CTkImage(image, size=image.size)
    capture.release()
    cv2.destroyAllWindows()


def toggle_preview() -> None:
    if PREVIEW.state() == "normal":
        PREVIEW.withdraw()
    elif modules.globals.source_path and modules.globals.target_path:
        init_preview()
        update_preview()


def init_preview() -> None:
    if is_image(modules.globals.target_path):
        preview_slider.pack_forget()
    if is_video(modules.globals.target_path):
        video_frame_total = get_video_frame_total(modules.globals.target_path)
        preview_slider.configure(to=video_frame_total)
        preview_slider.pack(fill="x")
        preview_slider.set(0)


def update_preview(frame_number: int = 0) -> None:
    if modules.globals.source_path and modules.globals.target_path:
        update_status("Processing...")
        temp_frame = get_video_frame(modules.globals.target_path, frame_number)
        if modules.globals.nsfw_filter and check_and_ignore_nsfw(temp_frame):
            return
        for frame_processor in get_frame_processors_modules(
                modules.globals.frame_processors
        ):
            temp_frame = frame_processor.process_frame(
                get_one_face(cv2.imread(modules.globals.source_path)), temp_frame
            )
        image = Image.fromarray(gpu_cvt_color(temp_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(
            image, (PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT), Image.LANCZOS
        )
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)
        update_status("Processing succeed!")
        PREVIEW.deiconify()


def webcam_preview(root: ctk.CTk, camera_index: int):
    global POPUP_LIVE

    if POPUP_LIVE and POPUP_LIVE.winfo_exists():
        update_status("Source x Target Mapper is already open.")
        POPUP_LIVE.focus()
        return

    # Prevent double-start: if a preview is already running, ignore the click
    if modules.globals.webcam_preview_running:
        update_status("Live preview already running")
        return

    if not modules.globals.map_faces:
        if modules.globals.source_path is None:
            update_status("Please select a source image first")
            return
        from modules.processors.frame.face_swapper import get_face_swapper
        from modules.face_analyser import get_face_analyser
        get_face_analyser()
        get_face_swapper()
        create_webcam_preview(camera_index)
    else:
        modules.globals.source_target_map = []
        create_source_target_popup_for_webcam(
            root, modules.globals.source_target_map, camera_index
        )



def get_available_cameras():
    """Returns a list of available camera names and indices."""
    if platform.system() == "Windows":
        try:
            graph = FilterGraph()
            devices = graph.get_input_devices()

            # Create list of indices and names
            camera_indices = list(range(len(devices)))
            camera_names = devices

            # If no cameras found through DirectShow, try OpenCV fallback
            if not camera_names:
                # Try to open camera with index -1 and 0
                test_indices = [-1, 0]
                working_cameras = []

                for idx in test_indices:
                    cap = cv2.VideoCapture(idx)
                    if cap.isOpened():
                        working_cameras.append(f"Camera {idx}")
                        cap.release()

                if working_cameras:
                    return test_indices[: len(working_cameras)], working_cameras

            # If still no cameras found, return empty lists
            if not camera_names:
                return [], ["No cameras found"]

            return camera_indices, camera_names

        except Exception as e:
            print(f"Error detecting cameras: {str(e)}")
            return [], ["No cameras found"]
    else:
        # Unix-like systems (Linux/Mac) camera detection
        camera_indices = []
        camera_names = []

        if platform.system() == "Darwin":
            # Do NOT probe cameras with cv2.VideoCapture on macOS — probing
            # invalid indices triggers the OBSENSOR backend and causes SIGSEGV.
            # Default to indices 0 and 1 (covers FaceTime + one USB camera).
            # The user can select the correct index from the UI dropdown.
            camera_indices = [0, 1]
            camera_names = ["Camera 0", "Camera 1"]
        else:
            # Linux camera detection - test first 10 indices
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    camera_indices.append(i)
                    camera_names.append(f"Camera {i}")
                    cap.release()

        if not camera_names:
            return [], ["No cameras found"]

        return camera_indices, camera_names


def _capture_thread_func(cap, capture_queue, stop_event):
    """Capture thread: reads frames from camera and puts them into the queue.
    Drops frames when the queue is full to avoid backpressure on the camera."""
    while not stop_event.is_set():
        ret, frame = cap.read()
        if not ret:
            stop_event.set()
            break
        try:
            capture_queue.put_nowait(frame)
        except queue.Full:
            # Drop the oldest frame and enqueue the new one
            try:
                capture_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                capture_queue.put_nowait(frame)
            except queue.Full:
                pass


def _processing_thread_func(capture_queue, processed_queue, stop_event,
                            camera_fps: float = 30.0):
    """Processing thread: takes raw frames from capture_queue, runs face
    detection (throttled), applies face swap/enhancement, and puts results
    into processed_queue.

    Args:
        camera_fps: Actual camera frame rate — used to compute how many
            frames to skip between face detections (~80ms target).
    """
    frame_processors = get_frame_processors_modules(modules.globals.frame_processors)
    source_image = None
    last_source_path = None
    prev_time = time.time()
    fps_update_interval = 0.5
    frame_count = 0
    fps = 0
    det_count = 0
    cached_target_face = None
    cached_many_faces = None
    # Detect every N frames ≈ 80ms.  At 60fps → every 5 frames (83ms),
    # at 30fps → every 3 frames (100ms), at 15fps → every frame.
    det_interval = max(1, round(camera_fps * 0.08))

    while not stop_event.is_set():
        try:
            frame = capture_queue.get(timeout=0.05)
        except queue.Empty:
            continue

        temp_frame = frame

        if modules.globals.live_mirror:
            temp_frame = gpu_flip(temp_frame, 1)

        if not modules.globals.map_faces:
            if modules.globals.source_path and modules.globals.source_path != last_source_path:
                last_source_path = modules.globals.source_path
                source_image = get_one_face(cv2.imread(modules.globals.source_path))

            # Run detection every det_interval frames (~80ms).
            # Use fast detection (det-only, no landmark/recognition) for live mode.
            det_count += 1
            if det_count % det_interval == 0:
                if modules.globals.many_faces:
                    cached_target_face = None
                    cached_many_faces = detect_many_faces_fast(temp_frame)
                else:
                    cached_target_face = detect_one_face_fast(temp_frame)
                    cached_many_faces = None

            # Build face list for enhancers from cached detection
            _cached_faces = None
            if cached_many_faces:
                _cached_faces = cached_many_faces
            elif cached_target_face is not None:
                _cached_faces = [cached_target_face]

            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame(
                            None, temp_frame, detected_faces=_cached_faces)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN256":
                    if modules.globals.fp_ui.get("face_enhancer_gpen256", False):
                        temp_frame = frame_processor.process_frame(
                            None, temp_frame, detected_faces=_cached_faces)
                elif frame_processor.NAME == "DLC.FACE-ENHANCER-GPEN512":
                    if modules.globals.fp_ui.get("face_enhancer_gpen512", False):
                        temp_frame = frame_processor.process_frame(
                            None, temp_frame, detected_faces=_cached_faces)
                elif frame_processor.NAME == "DLC.FACE-SWAPPER":
                    # Use cached face positions from detection thread
                    swapped_bboxes = []
                    if modules.globals.many_faces and cached_many_faces:
                        result = temp_frame.copy()
                        for t_face in cached_many_faces:
                            result = frame_processor.swap_face(source_image, t_face, result)
                            if hasattr(t_face, 'bbox') and t_face.bbox is not None:
                                swapped_bboxes.append(t_face.bbox.astype(int))
                        temp_frame = result
                    elif cached_target_face is not None:
                        temp_frame = frame_processor.swap_face(source_image, cached_target_face, temp_frame)
                        if hasattr(cached_target_face, 'bbox') and cached_target_face.bbox is not None:
                            swapped_bboxes.append(cached_target_face.bbox.astype(int))
                    # Apply post-processing (sharpening, interpolation)
                    temp_frame = frame_processor.apply_post_processing(temp_frame, swapped_bboxes)
                else:
                    temp_frame = frame_processor.process_frame(source_image, temp_frame)
        else:
            modules.globals.target_path = None
            for frame_processor in frame_processors:
                if frame_processor.NAME == "DLC.FACE-ENHANCER":
                    if modules.globals.fp_ui["face_enhancer"]:
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                elif frame_processor.NAME in ("DLC.FACE-ENHANCER-GPEN256", "DLC.FACE-ENHANCER-GPEN512"):
                    fp_key = frame_processor.NAME.split(".")[-1].lower().replace("-", "_")
                    if modules.globals.fp_ui.get(fp_key, False):
                        temp_frame = frame_processor.process_frame_v2(temp_frame)
                else:
                    temp_frame = frame_processor.process_frame_v2(temp_frame)

        # Calculate and display FPS
        current_time = time.time()
        frame_count += 1
        if current_time - prev_time >= fps_update_interval:
            fps = frame_count / (current_time - prev_time)
            frame_count = 0
            prev_time = current_time

        if modules.globals.show_fps:
            cv2.putText(
                temp_frame,
                f"FPS: {fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        # Queue the processed frame as BGR; the display thread resizes to the
        # preview window first and then runs cvtColor on the (much smaller)
        # buffer — cheaper than converting the full 1080p frame here.
        try:
            processed_queue.put_nowait(temp_frame)
        except queue.Full:
            try:
                processed_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                processed_queue.put_nowait(temp_frame)
            except queue.Full:
                pass


def create_webcam_preview(camera_index: int):
    global preview_label, PREVIEW

    modules.globals.webcam_preview_running = True

    cap = VideoCapturer(camera_index)
    # Try resolutions from high to safe fallback.
    # Retry once after a short delay — the camera may still be releasing
    # from a previous session (macOS holds the lock for ~1s after release).
    started = False
    resolutions = [(1920, 1080, 30), (1280, 720, 30), (640, 480, 30)]
    for attempt in range(2):
        for _w, _h, _fps in resolutions:
            if cap.start(_w, _h, _fps):
                started = True
                break
        if started:
            break
        # First pass failed — wait for the OS to release the camera lock
        update_status("Camera starting… please wait")
        ROOT.update()
        time.sleep(1.5)

    if not started:
        modules.globals.webcam_preview_running = False
        update_status(
            "Camera unavailable — grant access in "
            "System Settings › Privacy & Security › Camera"
        )
        return

    camera_fps = cap.actual_fps
    print(f"[webcam] Camera running at {cap.actual_width}x{cap.actual_height}@{camera_fps:.0f}fps")

    preview_label.configure(width=PREVIEW_DEFAULT_WIDTH, height=PREVIEW_DEFAULT_HEIGHT)
    PREVIEW.deiconify()

    # Queues for decoupling capture from processing and processing from display.
    # Small maxsize ensures we always work on recent frames and drop stale ones.
    capture_queue = queue.Queue(maxsize=2)
    processed_queue = queue.Queue(maxsize=2)
    stop_event = threading.Event()

    # Start capture thread
    cap_thread = threading.Thread(
        target=_capture_thread_func,
        args=(cap, capture_queue, stop_event),
        daemon=True,
    )
    cap_thread.start()

    # Start processing thread
    proc_thread = threading.Thread(
        target=_processing_thread_func,
        args=(capture_queue, processed_queue, stop_event, camera_fps),
        daemon=True,
    )
    proc_thread.start()

    # Cleanup helper called from the display loop when preview closes
    def _cleanup():
        modules.globals.webcam_preview_running = False
        stop_event.set()
        cap_thread.join(timeout=2.0)
        proc_thread.join(timeout=2.0)
        cap.release()
        PREVIEW.withdraw()

    # Poll at ~2x camera FPS (Nyquist) so we pick up frames promptly
    # without burning CPU.  Clamped to [1, 16] ms.
    poll_ms = max(1, min(16, int(500 / camera_fps)))

    # Non-blocking display loop using ROOT.after() — avoids blocking the
    # Tk event loop which could cause UI freezes or re-entrancy issues.
    def _display_next_frame():
        if stop_event.is_set() or PREVIEW.state() == "withdrawn":
            _cleanup()
            return

        try:
            bgr_frame = processed_queue.get_nowait()
        except queue.Empty:
            ROOT.after(poll_ms, _display_next_frame)
            return

        # Resize the full-resolution BGR frame to the preview window first,
        # then convert colour on the smaller buffer.
        bgr_frame = fit_image_to_size(
            bgr_frame, PREVIEW.winfo_width(), PREVIEW.winfo_height()
        )
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb_frame)
        image = ctk.CTkImage(image, size=image.size)
        preview_label.configure(image=image)

        ROOT.after(poll_ms, _display_next_frame)

    # Kick off the non-blocking display loop
    ROOT.after(0, _display_next_frame)


def create_source_target_popup_for_webcam(
        root: ctk.CTk, map: list, camera_index: int
) -> None:
    global POPUP_LIVE, popup_status_label_live

    POPUP_LIVE = ctk.CTkToplevel(root)
    POPUP_LIVE.title(_("Source x Target Mapper"))
    POPUP_LIVE.geometry(f"{POPUP_LIVE_WIDTH}x{POPUP_LIVE_HEIGHT}")
    POPUP_LIVE.focus()

    def on_submit_click():
        if has_valid_map():
            simplify_maps()
            update_pop_live_status("Mappings successfully submitted!")
            create_webcam_preview(camera_index)  # Open the preview window
        else:
            update_pop_live_status("At least 1 source with target is required!")

    def on_add_click():
        add_blank_map()
        refresh_data(map)
        update_pop_live_status("Please provide mapping!")

    def on_clear_click():
        clear_source_target_images(map)
        refresh_data(map)
        update_pop_live_status("All mappings cleared!")

    popup_status_label_live = ctk.CTkLabel(POPUP_LIVE, text=None, justify="center")
    popup_status_label_live.grid(row=1, column=0, pady=15)

    add_button = ctk.CTkButton(POPUP_LIVE, text=_("Add"), command=lambda: on_add_click())
    add_button.place(relx=0.1, rely=0.92, relwidth=0.2, relheight=0.05)

    clear_button = ctk.CTkButton(POPUP_LIVE, text=_("Clear"), command=lambda: on_clear_click())
    clear_button.place(relx=0.4, rely=0.92, relwidth=0.2, relheight=0.05)

    close_button = ctk.CTkButton(
        POPUP_LIVE, text=_("Submit"), command=lambda: on_submit_click()
    )
    close_button.place(relx=0.7, rely=0.92, relwidth=0.2, relheight=0.05)



def clear_source_target_images(map: list):
    global source_label_dict_live, target_label_dict_live

    for item in map:
        if "source" in item:
            del item["source"]
        if "target" in item:
            del item["target"]

    for button_num in list(source_label_dict_live.keys()):
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    for button_num in list(target_label_dict_live.keys()):
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]


def refresh_data(map: list):
    global POPUP_LIVE

    scrollable_frame = ctk.CTkScrollableFrame(
        POPUP_LIVE, width=POPUP_LIVE_SCROLL_WIDTH, height=POPUP_LIVE_SCROLL_HEIGHT
    )
    scrollable_frame.grid(row=0, column=0, padx=0, pady=0, sticky="nsew")

    def on_sbutton_click(map, button_num):
        map = update_webcam_source(scrollable_frame, map, button_num)

    def on_tbutton_click(map, button_num):
        map = update_webcam_target(scrollable_frame, map, button_num)

    for item in map:
        id = item["id"]

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select source image"),
            command=lambda id=id: on_sbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=0, padx=30, pady=10)

        x_label = ctk.CTkLabel(
            scrollable_frame,
            text=f"X",
            width=MAPPER_PREVIEW_MAX_WIDTH,
            height=MAPPER_PREVIEW_MAX_HEIGHT,
        )
        x_label.grid(row=id, column=2, padx=10, pady=10)

        button = ctk.CTkButton(
            scrollable_frame,
            text=_("Select target image"),
            command=lambda id=id: on_tbutton_click(map, id),
            width=DEFAULT_BUTTON_WIDTH,
            height=DEFAULT_BUTTON_HEIGHT,
        )
        button.grid(row=id, column=3, padx=20, pady=10)

        if "source" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=id, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)

        if "target" in item:
            image = Image.fromarray(
                gpu_cvt_color(item["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{id}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=id, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)


def update_webcam_source(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global source_label_dict_live

    source_path = ctk.filedialog.askopenfilename(
        title=_("select an source image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "source" in map[button_num]:
        map[button_num].pop("source")
        source_label_dict_live[button_num].destroy()
        del source_label_dict_live[button_num]

    if source_path == "":
        return map
    else:
        cv2_img = cv2.imread(source_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["source"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["source"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            source_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"S-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            source_image.grid(row=button_num, column=1, padx=10, pady=10)
            source_image.configure(image=tk_image)
            source_label_dict_live[button_num] = source_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map


def update_webcam_target(
        scrollable_frame: ctk.CTkScrollableFrame, map: list, button_num: int
) -> list:
    global target_label_dict_live

    target_path = ctk.filedialog.askopenfilename(
        title=_("select an target image"),
        initialdir=RECENT_DIRECTORY_SOURCE,
        filetypes=[img_ft],
    )

    if "target" in map[button_num]:
        map[button_num].pop("target")
        target_label_dict_live[button_num].destroy()
        del target_label_dict_live[button_num]

    if target_path == "":
        return map
    else:
        cv2_img = cv2.imread(target_path)
        face = get_one_face(cv2_img)

        if face:
            x_min, y_min, x_max, y_max = face["bbox"]

            map[button_num]["target"] = {
                "cv2": cv2_img[int(y_min): int(y_max), int(x_min): int(x_max)],
                "face": face,
            }

            image = Image.fromarray(
                gpu_cvt_color(map[button_num]["target"]["cv2"], cv2.COLOR_BGR2RGB)
            )
            image = image.resize(
                (MAPPER_PREVIEW_MAX_WIDTH, MAPPER_PREVIEW_MAX_HEIGHT), Image.LANCZOS
            )
            tk_image = ctk.CTkImage(image, size=image.size)

            target_image = ctk.CTkLabel(
                scrollable_frame,
                text=f"T-{button_num}",
                width=MAPPER_PREVIEW_MAX_WIDTH,
                height=MAPPER_PREVIEW_MAX_HEIGHT,
            )
            target_image.grid(row=button_num, column=4, padx=20, pady=10)
            target_image.configure(image=tk_image)
            target_label_dict_live[button_num] = target_image
        else:
            update_pop_live_status("Face could not be detected in last upload!")
        return map

