"""The memory an engine process uses: proc_pid_rusage, the footprint macOS itself reports (the mapped
model file is not in it). Only macOS has this call; elsewhere there is no reading and the page shows no memory."""

from __future__ import annotations

import ctypes
import ctypes.util
import sys

RUSAGE_INFO_V4 = 4


class RusageInfoV4(ctypes.Structure):
    """struct rusage_info_v4 from <sys/resource.h> (macOS SDK 15). All fields after ri_uuid are uint64."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time", "ri_system_time", "ri_pkg_idle_wkups", "ri_interrupt_wkups", "ri_pageins",
            "ri_wired_size", "ri_resident_size", "ri_phys_footprint", "ri_proc_start_abstime",
            "ri_proc_exit_abstime", "ri_child_user_time", "ri_child_system_time", "ri_child_pkg_idle_wkups",
            "ri_child_interrupt_wkups", "ri_child_pageins", "ri_child_elapsed_abstime", "ri_diskio_bytesread",
            "ri_diskio_byteswritten", "ri_cpu_time_qos_default", "ri_cpu_time_qos_maintenance",
            "ri_cpu_time_qos_background", "ri_cpu_time_qos_utility", "ri_cpu_time_qos_legacy",
            "ri_cpu_time_qos_user_initiated", "ri_cpu_time_qos_user_interactive", "ri_billed_system_time",
            "ri_serviced_system_time", "ri_logical_writes", "ri_lifetime_max_phys_footprint", "ri_instructions",
            "ri_cycles", "ri_billed_energy", "ri_serviced_energy", "ri_interval_max_phys_footprint",
            "ri_runnable_time",
        )
    ]  # fmt: skip


def footprint_bytes(pid: int) -> int | None:
    """The footprint of process `pid` in bytes, or None when it is gone, the call fails, or this is not macOS."""
    if sys.platform != "darwin":
        return None
    libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    libc.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    libc.proc_pid_rusage.restype = ctypes.c_int
    info = RusageInfoV4()
    if libc.proc_pid_rusage(pid, RUSAGE_INFO_V4, ctypes.byref(info)) != 0:
        return None
    return info.ri_phys_footprint
