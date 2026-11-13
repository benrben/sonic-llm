//! Process memory accounting: resident set and physical footprint via `proc_pid_rusage`.
//!
//! `ri_phys_footprint` is the number Activity Monitor calls "Memory" and the one llama.cpp's accounting
//! reports: clean file-backed mmap pages count in `ri_resident_size` but not in the footprint.

use serde::Serialize;

#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct MemSnapshot {
    pub resident_bytes: u64,
    pub phys_footprint_bytes: u64,
    /// `ri_lifetime_max_phys_footprint`: the highest footprint the process has had (what `/usr/bin/time -l` calls "peak memory footprint").
    pub lifetime_max_phys_footprint_bytes: u64,
}

pub fn snapshot() -> MemSnapshot {
    let mut info = std::mem::MaybeUninit::<libc::rusage_info_v4>::zeroed();
    // SAFETY: proc_pid_rusage fills a rusage_info_v4 for our own pid; the buffer is zeroed and correctly sized.
    let rc = unsafe {
        libc::proc_pid_rusage(
            libc::getpid(),
            libc::RUSAGE_INFO_V4,
            info.as_mut_ptr() as *mut libc::rusage_info_t,
        )
    };
    if rc != 0 {
        return MemSnapshot::default();
    }
    // SAFETY: rc == 0, so the struct is initialised.
    let info = unsafe { info.assume_init() };
    MemSnapshot {
        resident_bytes: info.ri_resident_size,
        phys_footprint_bytes: info.ri_phys_footprint,
        lifetime_max_phys_footprint_bytes: info.ri_lifetime_max_phys_footprint,
    }
}

/// Installed physical memory in bytes (`hw.memsize`).
pub fn physical_memory_bytes() -> u64 {
    let mut v: u64 = 0;
    let mut len = std::mem::size_of::<u64>();
    // SAFETY: sysctlbyname writes at most `len` bytes into v.
    let rc = unsafe {
        libc::sysctlbyname(
            c"hw.memsize".as_ptr(),
            &mut v as *mut u64 as *mut libc::c_void,
            &mut len,
            std::ptr::null_mut(),
            0,
        )
    };
    if rc == 0 {
        v
    } else {
        0
    }
}

pub fn page_size() -> usize {
    // SAFETY: sysconf is always safe to call.
    unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize }
}
