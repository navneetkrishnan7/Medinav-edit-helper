"""
launcher.py - wraps medinav_script_tool.py so that any crash on startup
shows a Windows message box instead of silently disappearing.
Run this instead of medinav_script_tool.py.
"""
import os
import sys
import traceback
from datetime import datetime


def show_error(msg):
    try:
        import ctypes

        ctypes.windll.user32.MessageBoxW(0, msg, "Medinav Script Tool - Error", 0x10)
    except Exception:
        print(msg, file=sys.stderr)


def write_log(tb):
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        with open(os.path.join(here, "medinav-error.log"), "a", encoding="utf-8") as f:
            f.write("\n\n[%s] Startup crash\n%s\n" % (datetime.now().isoformat(timespec="seconds"), tb))
    except Exception:
        pass


try:
    here = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, here)
    import medinav_script_tool

    medinav_script_tool.main()
except Exception:
    tb = traceback.format_exc()
    write_log(tb)
    show_error(
        "The app failed to start. Copy the error below and send to your admin:\n\n" + tb
    )
    sys.exit(1)
