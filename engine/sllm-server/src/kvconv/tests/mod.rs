use super::*;

#[test]
fn hadamard_is_orthonormal_and_its_own_inverse() {
    let x: Vec<f32> = (0..256).map(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0).collect();
    let mut y = x.clone();
    hadamard(&mut y, 128);
    let (nx, ny): (f32, f32) = (x.iter().map(|v| v * v).sum(), y.iter().map(|v| v * v).sum());
    assert!((nx - ny).abs() < 1e-3 * nx);
    hadamard(&mut y, 128);
    assert!(x.iter().zip(&y).all(|(a, b)| (a - b).abs() < 1e-4));
    // the first row of H is all +1/sqrt(n): y[0] is the block sum over sqrt(n)
    let mut z = x[..64].to_vec();
    hadamard(&mut z, 64);
    assert!((z[0] - x[..64].iter().sum::<f32>() / 8.0).abs() < 1e-4);
}

/// A synthetic state: one cache, `n_stream` streams (cells in the first), 2 layers of 64-value rows.
fn state(n_stream: u32, cells: u32, ty: i32, row: u64, n_rot: (u32, u32)) -> Vec<u8> {
    state_at(n_stream, 0, cells, ty, row, n_rot)
}

fn state_at(n_stream: u32, pos0: u32, cells: u32, ty: i32, row: u64, n_rot: (u32, u32)) -> Vec<u8> {
    let mut b = Vec::new();
    b.extend_from_slice(&0xdead_beefu32.to_le_bytes());
    b.extend_from_slice(&3i32.to_le_bytes());
    b.extend_from_slice(&n_stream.to_le_bytes());
    b.extend_from_slice(&cells.to_le_bytes());
    for p in pos0..pos0 + cells {
        for x in [p, 1, 3] {
            b.extend_from_slice(&x.to_le_bytes());
        }
    }
    for x in [0, 2, n_rot.0, n_rot.1] {
        b.extend_from_slice(&(x as u32).to_le_bytes());
    }
    for l in 0..4 {
        b.extend_from_slice(&ty.to_le_bytes());
        b.extend_from_slice(&row.to_le_bytes());
        for i in 0..row * cells as u64 {
            b.push(if ty == T_F16 { if i % 2 == 1 { 0x3c } else { 0 } } else { (l * 17 + i as usize) as u8 });
        }
    }
    for _ in 1..n_stream {
        b.extend_from_slice(&0u32.to_le_bytes());
    }
    b
}

#[test]
fn f16_state_converts_to_the_target_layout() {
    const Q4_0: i32 = 2;
    let target = Layout::probe(&state(4, 1, Q4_0, 36, (64, 64))).unwrap();
    assert_eq!(target.ext, 0);
    assert_eq!(target.cell_bytes(), 4 * 36);
    let src = state(1, 5, T_F16, 128, (0, 0)); // every value 1.0
    let out = convert(&src, &target, 7, 3, None).unwrap();
    let (_, kvs) = parse(&out, 0).unwrap();
    let s = kvs[0].stream.as_ref().unwrap();
    assert_eq!((kvs[0].n_stream, s.cells, s.n_rot_k, s.n_rot_v, s.k.len()), (4, 5, 64, 64, 2));
    assert_eq!(&out[4..8], &7i32.to_le_bytes());
    assert_eq!(&out[s.meta.clone()], &src[16..16 + 5 * 12]);
    // H of a block of 64 ones is (8, 0, 0, ...); q4_0 (d = max / -8 = -1): value 8 -> nibble 0, value 0 -> nibble 8
    let r = &out[s.k[0].data.clone()][..18];
    assert_eq!(half_to_f32(u16::from_le_bytes([r[0], r[1]])), -1.0);
    assert_eq!((r[2], r[3]), (0x80, 0x88));
}

#[test]
fn kept_cells_stay_as_they_are() {
    const Q4_0: i32 = 2;
    let target = Layout::probe(&state(4, 1, Q4_0, 36, (64, 64))).unwrap();
    let main = state(4, 3, Q4_0, 36, (64, 64)); // cells at positions 0, 1, 2
    let src = state_at(1, 2, 2, T_F16, 128, (0, 0)); // cells at positions 2, 3
    let out = convert(&src, &target, 7, 2, Some((&main, 2))).unwrap();
    let (_, kvs) = parse(&out, 0).unwrap();
    let s = kvs[0].stream.as_ref().unwrap();
    assert_eq!(s.cells, 4);
    assert_eq!(s.cell_meta.iter().map(|c| c.0).collect::<Vec<_>>(), vec![0, 1, 2, 3]);
    let (_, mk) = parse(&main, 0).unwrap();
    let m = mk[0].stream.as_ref().unwrap();
    for l in 0..2 {
        // the first two rows of every layer are the main state's own bytes
        assert_eq!(&out[s.k[l].data.start..s.k[l].data.start + 72], &main[m.k[l].data.start..m.k[l].data.start + 72]);
        assert_eq!(&out[s.v[l].data.start..s.v[l].data.start + 72], &main[m.v[l].data.start..m.v[l].data.start + 72]);
    }
    // the converted rows: a block of 64 ones (see above)
    assert_eq!(&out[s.k[0].data.start + 72 + 2..s.k[0].data.start + 72 + 4], &[0x80, 0x88]);
    assert!(convert(&state(1, 2, T_F16, 128, (0, 0)), &target, 7, 2, Some((&main, 2))).is_err(), "a source cell below the kept position");
}

fn half_to_f32(h: u16) -> f32 {
    let mut f = 0f32;
    unsafe { ggml_fp16_to_fp32_row(&h, &mut f, 1) };
    f
}
