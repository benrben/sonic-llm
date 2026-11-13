//! Minimal `.npy` (format 1.0) writer for a C-order f32 matrix.

use std::io::{self, Write};

/// The header bytes for an `<f4` array of shape `[rows, cols]`; the data follows directly.
pub fn header_f32_2d(rows: usize, cols: usize) -> Vec<u8> {
    let dict = format!("{{'descr': '<f4', 'fortran_order': False, 'shape': ({rows}, {cols}), }}");
    // magic (6) + version (2) + header length (2) + dict + padding + '\n' must be a multiple of 64.
    let unpadded = 10 + dict.len() + 1;
    let pad = (64 - unpadded % 64) % 64;
    let mut h = Vec::with_capacity(unpadded + pad);
    h.extend_from_slice(b"\x93NUMPY\x01\x00");
    h.extend_from_slice(&((dict.len() + pad + 1) as u16).to_le_bytes());
    h.extend_from_slice(dict.as_bytes());
    h.extend(std::iter::repeat(b' ').take(pad));
    h.push(b'\n');
    h
}

pub fn write_header_f32_2d(w: &mut impl Write, rows: usize, cols: usize) -> io::Result<()> {
    w.write_all(&header_f32_2d(rows, cols))
}

/// Write f32 values little endian (the platform is little endian; this keeps it explicit).
pub fn write_f32s(w: &mut impl Write, data: &[f32]) -> io::Result<()> {
    #[cfg(target_endian = "little")]
    {
        // SAFETY: f32 has no padding; reinterpreting as bytes is sound.
        let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, std::mem::size_of_val(data)) };
        w.write_all(bytes)
    }
    #[cfg(not(target_endian = "little"))]
    {
        for v in data {
            w.write_all(&v.to_le_bytes())?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_is_aligned_and_well_formed() {
        for (r, c) in [(1, 1), (2000, 151936), (2000, 262144), (123456789, 3)] {
            let h = header_f32_2d(r, c);
            assert_eq!(h.len() % 64, 0);
            assert_eq!(&h[..8], b"\x93NUMPY\x01\x00");
            let hl = u16::from_le_bytes([h[8], h[9]]) as usize;
            assert_eq!(10 + hl, h.len());
            assert_eq!(*h.last().unwrap(), b'\n');
            let s = std::str::from_utf8(&h[10..]).unwrap();
            assert!(s.starts_with("{'descr': '<f4', 'fortran_order': False, 'shape': ("));
            assert!(s.contains(&format!("({r}, {c})")));
        }
    }
}
