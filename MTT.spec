# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

# PySide6 plugins + data files
pyside6_data = collect_data_files("PySide6", subdir="plugins")
pyside6_data += collect_data_files("PySide6", subdir="translations")

# soundcard needs its mediafoundation DLLs on Windows
soundcard_data = collect_data_files("soundcard")

# soundfile ships libsndfile
soundfile_data = collect_data_files("soundfile")

# certifi CA bundle for HTTPS (google-cloud, websockets, etc.)
certifi_data = collect_data_files("certifi")

a = Analysis(
    ["app.py"],
    pathex=[],
    binaries=[],
    datas=[
        *pyside6_data,
        *soundcard_data,
        *soundfile_data,
        *certifi_data,
    ],
    hiddenimports=[
        # Our own packages
        "ui",
        "ui.main_window",
        "ui.live_captions_widget",
        "ui.vu_meter_widget",
        "ui.job_status_widget",
        "services",
        "services.audio_capture",
        "services.session_store",
        "services.cloud_run_service",
        "services.realtime_cloud_service",
        "services.runtime_paths",
        "services.audio_processing",
        # Dependencies that are imported lazily
        "soundcard",
        "soundcard.mediafoundation",
        "soundfile",
        "websockets",
        "websockets.sync",
        "websockets.sync.client",
        "google.cloud.storage",
        "google.auth",
        "google.auth.transport.requests",
        "dotenv",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="MTT",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,   # windowed app, no terminal
    icon=None,       # add .ico path here if you have one
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="MTT",
)
