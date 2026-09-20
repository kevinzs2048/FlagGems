"""Read-only Darwin process I/O/memory and public Foundation power-state samples."""

import argparse
import ctypes
import json
import os
import time
from functools import lru_cache


class RUsageV2(ctypes.Structure):
    # macOS SDK sys/resource.h: RUSAGE_INFO_V2. Time counters are left raw;
    # the latency analysis uses monotonic wall time, pageins and byte counters.
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "pkg_idle_wkups",
            "interrupt_wkups",
            "pageins",
            "wired_size",
            "resident_size",
            "phys_footprint",
            "proc_start_abstime",
            "proc_exit_abstime",
            "child_user_time",
            "child_system_time",
            "child_pkg_idle_wkups",
            "child_interrupt_wkups",
            "child_pageins",
            "child_elapsed_abstime",
            "diskio_bytesread",
            "diskio_byteswritten",
        )
    ]


@lru_cache(None)
def process_reader():
    lib = ctypes.CDLL("/usr/lib/libproc.dylib", use_errno=True)
    lib.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    lib.proc_pid_rusage.restype = ctypes.c_int
    return lib.proc_pid_rusage


@lru_cache(None)
def power_reader():
    foundation = ctypes.CDLL(
        "/System/Library/Frameworks/Foundation.framework/Foundation"
    )
    objc = ctypes.CDLL("/usr/lib/libobjc.A.dylib")
    objc.objc_getClass.argtypes = [ctypes.c_char_p]
    objc.objc_getClass.restype = ctypes.c_void_p
    objc.sel_registerName.argtypes = [ctypes.c_char_p]
    objc.sel_registerName.restype = ctypes.c_void_p
    call_id = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    call_integer = ctypes.CFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)(
        ("objc_msgSend", objc)
    )
    instance = call_id(
        objc.objc_getClass(b"NSProcessInfo"), objc.sel_registerName(b"processInfo")
    )
    thermal = objc.sel_registerName(b"thermalState")
    low_power = objc.sel_registerName(b"isLowPowerModeEnabled")

    def read():
        # Keep both framework handles alive with the closure.
        _ = (foundation, objc)
        return {
            "thermal_state": call_integer(instance, thermal),
            "low_power_mode": bool(call_integer(instance, low_power)),
        }

    return read


def sample(pid=None):
    pid = os.getpid() if pid is None else pid
    value = RUsageV2()
    if process_reader()(pid, 2, ctypes.byref(value)):
        code = ctypes.get_errno()
        raise OSError(code, os.strerror(code))
    result = {
        "unix_seconds": time.time(),
        "monotonic_seconds": time.perf_counter(),
        "pid": pid,
    }
    result.update(
        {
            key: int(getattr(value, key))
            for key in (
                "pageins",
                "resident_size",
                "phys_footprint",
                "diskio_bytesread",
                "diskio_byteswritten",
                "user_time",
                "system_time",
            )
        }
    )
    result.update(power_reader()())
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pid", type=int)
    parser.add_argument("--duration", type=float, default=0)
    parser.add_argument("--interval", type=float, default=15)
    args = parser.parse_args()
    if args.interval <= 0 or args.duration < 0:
        parser.error("Invalid sampling interval/duration")
    deadline = time.monotonic() + args.duration
    while True:
        try:
            print(json.dumps(sample(args.pid)), flush=True)
        except OSError as error:
            print(
                json.dumps(
                    {
                        "unix_seconds": time.time(),
                        "pid": args.pid,
                        "status": "process_read_error",
                        "errno": error.errno,
                        "message": str(error),
                    }
                ),
                flush=True,
            )
            if error.errno == 3:
                break  # Kernel confirms the PID no longer exists.
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        time.sleep(min(args.interval, remaining))
