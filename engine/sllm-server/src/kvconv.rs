//! `--kv-read f16`: a sequence's state read in an f16 cache, rewritten for the engine's quantized cache.
//!
//! A quantized cache (b11669) holds K and V rotated: libllama multiplies every block of `n_rot_k` (`n_rot_v`) values of a
//! K (V) row by the orthonormal Walsh-Hadamard matrix before it quantizes the row (`llama_mul_mat_hadamard`, the
//! `attn_rot` of llama-kv-cache.cpp), and rotates Q the same way. An f16 cache holds them as they are. The conversion
//! does on the CPU what the quantized cache's write does on the GPU: f16 -> f32, the same rotation, ggml's own quantizer
//! (`ggml_quantize_chunk`) into the target row type. The cell metadata (positions, sequence ids, cell extras) is copied
//! as it is: both caches belong to the same model.
//!
//! The state format is `llama_state_seq_get_data`'s: magic, sequence id, then per KV cache (two for a sliding-window
//! model: base, then window) `n_stream` and per stream `cell_count`, the cells' metadata and the data (v_trans, n_layer,
//! n_rot_k, n_rot_v, per layer K type, row size and rows, then the same for V). The target layout (stream count, row
//! types and sizes, rotations) is read from a one-cell state of the target context itself (`Layout::probe`).

use std::ops::Range;

const T_F32: i32 = 0;
const T_F16: i32 = 1;

unsafe extern "C" {
    fn ggml_quantize_chunk(ty: i32, src: *const f32, dst: *mut u8, start: i64, nrows: i64, n_per_row: i64, imatrix: *const f32) -> usize;
    fn ggml_row_size(ty: i32, ne: i64) -> usize;
    fn ggml_fp16_to_fp32_row(x: *const u16, y: *mut f32, n: i64);
}

struct Rd<'a> {
    b: &'a [u8],
    at: usize,
}

impl Rd<'_> {
    fn take(&mut self, n: usize) -> Result<Range<usize>, String> {
        if self.at + n > self.b.len() {
            return Err(format!("state ends at {} (wanted {n} more bytes at {})", self.b.len(), self.at));
        }
        self.at += n;
        Ok(self.at - n..self.at)
    }
    fn u32(&mut self) -> Result<u32, String> {
        let r = self.take(4)?;
        Ok(u32::from_le_bytes(self.b[r].try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        let r = self.take(8)?;
        Ok(u64::from_le_bytes(self.b[r].try_into().unwrap()))
    }
}

/// One layer's K or V rows of one stream.
#[derive(Clone, Debug)]
struct Rows {
    ty: i32,
    row: u64,
    data: Range<usize>,
}

/// The non-empty stream of one KV cache in a sequence state.
#[derive(Clone, Debug)]
struct Stream {
    cells: u32,
    meta: Range<usize>,
    /// Each cell's position and metadata bytes, in state order (the row order).
    cell_meta: Vec<(i32, Range<usize>)>,
    v_trans: u32,
    n_rot_k: u32,
    n_rot_v: u32,
    k: Vec<Rows>,
    v: Vec<Rows>,
}

impl Stream {
    /// Layer `l`'s K rows (`part` 0) or V rows (1).
    fn layer(&self, part: usize, l: usize) -> &Rows {
        if part == 0 { &self.k[l] } else { &self.v[l] }
    }
}

#[derive(Clone, Debug)]
struct Kv {
    n_stream: u32,
    /// None: the sequence has no cells in this cache.
    stream: Option<Stream>,
}

/// Parse a sequence state; `ext` is the size of the per-cell extra (0 unless the model has M-RoPE or per-layer embeddings),
/// which the format does not record. Every layer has a V (true of the models the engine runs; the parse must end exactly
/// at the end of the state, which catches a model where it is not).
fn parse(b: &[u8], ext: usize) -> Result<(u32, Vec<Kv>), String> {
    let mut r = Rd { b, at: 0 };
    let magic = r.u32()?;
    let _seq = r.u32()?;
    let mut kvs = Vec::new();
    while r.at < b.len() {
        kvs.push(parse_kv(&mut r, ext)?);
    }
    Ok((magic, kvs))
}

/// One KV cache of a state: its stream count and its one stream with cells.
fn parse_kv(r: &mut Rd, ext: usize) -> Result<Kv, String> {
    let n_stream = r.u32()?;
    if n_stream == 0 || n_stream > 4096 {
        return Err(format!("implausible stream count {n_stream}"));
    }
    let mut stream = None;
    for _ in 0..n_stream {
        let cells = r.u32()?;
        if cells == 0 {
            continue;
        }
        if stream.is_some() {
            return Err("a sequence with cells in two streams".into());
        }
        stream = Some(parse_stream(r, cells, ext)?);
    }
    Ok(Kv { n_stream, stream })
}

/// A stream of `cells` cells: their metadata, then the data (every layer's K rows, then every layer's V rows).
fn parse_stream(r: &mut Rd, cells: u32, ext: usize) -> Result<Stream, String> {
    let m0 = r.at;
    let cell_meta = parse_cells(r, cells, ext)?;
    let meta = m0..r.at;
    let [v_trans, n_layer, n_rot_k, n_rot_v] = data_header(r)?;
    if n_layer == 0 || n_layer > 1024 {
        return Err(format!("implausible layer count {n_layer}"));
    }
    if v_trans != 0 {
        return Err("a transposed V cache (flash attention off) is not supported".into());
    }
    let k = (0..n_layer).map(|_| parse_rows(r, cells)).collect::<Result<Vec<_>, _>>()?;
    let v = (0..n_layer).map(|_| parse_rows(r, cells)).collect::<Result<Vec<_>, _>>()?;
    Ok(Stream { cells, meta, cell_meta, v_trans, n_rot_k, n_rot_v, k, v })
}

/// A stream's data header: v_trans, n_layer, n_rot_k, n_rot_v.
fn data_header(r: &mut Rd) -> Result<[u32; 4], String> {
    Ok([r.u32()?, r.u32()?, r.u32()?, r.u32()?])
}

/// Each cell's position and metadata bytes (position, sequence count, sequence ids, the extra of `ext` bytes).
fn parse_cells(r: &mut Rd, cells: u32, ext: usize) -> Result<Vec<(i32, Range<usize>)>, String> {
    let mut cell_meta = Vec::with_capacity(cells as usize);
    for _ in 0..cells {
        let c0 = r.at;
        let pos = r.u32()? as i32;
        let n_seq = r.u32()? as usize;
        r.take(ext + 4 * n_seq)?;
        cell_meta.push((pos, c0..r.at));
    }
    Ok(cell_meta)
}

/// One layer's K or V rows of `cells` cells: row type, row size, rows.
fn parse_rows(r: &mut Rd, cells: u32) -> Result<Rows, String> {
    let ty = r.u32()? as i32;
    let row = r.u64()?;
    if !(0..64).contains(&ty) || row == 0 || row > 1 << 24 {
        return Err(format!("implausible row type {ty} size {row}"));
    }
    let data = r.take(row as usize * cells as usize)?;
    Ok(Rows { ty, row, data })
}

/// The target context's cache layout.
#[derive(Clone, Debug)]
pub struct Layout {
    ext: usize,
    kvs: Vec<Kv>,
}

impl Layout {
    /// From a state of one sequence with cells in every cache (one cell is enough) of the target context.
    pub fn probe(state: &[u8]) -> Result<Layout, String> {
        let mut last = String::new();
        for ext in [0usize, 4, 8, 12, 16, 20, 24, 28, 32] {
            match parse(state, ext) {
                Ok((_, kvs)) if kvs.iter().all(|k| k.stream.is_some()) => return Ok(Layout { ext, kvs }),
                Ok(_) => last = "the probe sequence has no cells in some cache".into(),
                Err(e) => last = e,
            }
        }
        Err(format!("unrecognized sequence state: {last}"))
    }

    /// The bytes one cell takes in the target cache (every layer's K and V row).
    pub fn cell_bytes(&self) -> u64 {
        self.kvs.iter().filter_map(|k| k.stream.as_ref()).map(|s| s.k.iter().chain(&s.v).map(|r| r.row).sum::<u64>()).sum()
    }

    pub fn row_types(&self) -> Vec<i32> {
        let mut t: Vec<i32> = self.kvs.iter().filter_map(|k| k.stream.as_ref()).flat_map(|s| s.k.iter().chain(&s.v).map(|r| r.ty)).collect();
        t.dedup();
        t
    }
}

/// The orthonormal Walsh-Hadamard transform of every block of `n` values (natural order, as `ggml_gen_hadamard` builds
/// the matrix: H x for each block, H symmetric with entries +-1/sqrt(n)).
fn hadamard(x: &mut [f32], n: usize) {
    let s = 1.0 / (n as f32).sqrt();
    for b in x.chunks_exact_mut(n) {
        let mut h = 1;
        while h < n {
            for i in (0..n).step_by(2 * h) {
                for j in i..i + h {
                    let (a, c) = (b[j], b[j + h]);
                    b[j] = a + c;
                    b[j + h] = a - c;
                }
            }
            h *= 2;
        }
        for v in b.iter_mut() {
            *v *= s;
        }
    }
}

/// One layer's rows: f16 `src` -> the target row type, rotated by blocks of `rot` first (0: no rotation).
fn convert_rows(src: &[u8], ne: usize, rot: usize, ty: i32, dst: &mut [u8]) {
    let cells = src.len() / (2 * ne);
    let dst_row = dst.len() / cells.max(1);
    let mut f = vec![0f32; ne];
    for c in 0..cells {
        let s = &src[c * 2 * ne..(c + 1) * 2 * ne];
        let d = &mut dst[c * dst_row..(c + 1) * dst_row];
        if ty == T_F16 && rot == 0 {
            d.copy_from_slice(s);
            continue;
        }
        unsafe { ggml_fp16_to_fp32_row(s.as_ptr() as *const u16, f.as_mut_ptr(), ne as i64) };
        if rot > 0 {
            hadamard(&mut f, rot);
        }
        unsafe { ggml_quantize_chunk(ty, f.as_ptr(), d.as_mut_ptr(), 0, 1, ne as i64, std::ptr::null()) };
    }
}

/// One layer's rows to convert: the source rows, values per row, rotation, target type, and where they go in the output.
struct Job {
    src: Range<usize>,
    ne: usize,
    rot: usize,
    ty: i32,
    dst: Range<usize>,
}

/// The output state as it is laid out (every offset known before any row is converted), and the conversions it waits for.
#[derive(Default)]
struct Out {
    bytes: Vec<u8>,
    jobs: Vec<Job>,
}

/// What a conversion reads: the source state, the kept state's bytes (when cells are kept), and the position below
/// which the cells are the kept state's.
struct Sources<'a> {
    src: &'a [u8],
    kept: Option<&'a [u8]>,
    below: i32,
}

/// Rewrite `src` (a state of an f16 context of the same model) for the target layout, as a state of sequence `seq`.
/// `keep = (state, p)`: the cells below position `p` come from `state` (a state of the target context, rows as they are)
/// and `src` holds only cells from `p` on. `threads` workers convert the rows.
pub fn convert(src: &[u8], to: &Layout, seq: i32, threads: usize, keep: Option<(&[u8], i32)>) -> Result<Vec<u8>, String> {
    let (magic, kvs) = parse(src, to.ext)?;
    if kvs.len() != to.kvs.len() {
        return Err(format!("{} caches in the state, {} in the target", kvs.len(), to.kvs.len()));
    }
    let (kept, below) = match keep {
        Some((b, p)) => (Some(parse(b, to.ext)?.1), p),
        None => (None, 0),
    };
    if kept.as_ref().is_some_and(|k| k.len() != kvs.len()) {
        return Err("the kept state has another number of caches".into());
    }
    // the output layout first (every offset known), then the rows, converted in parallel
    let mut out = Out::default();
    out.bytes.extend_from_slice(&magic.to_le_bytes());
    out.bytes.extend_from_slice(&seq.to_le_bytes());
    let empty = Stream { cells: 0, meta: 0..0, cell_meta: Vec::new(), v_trans: 0, n_rot_k: 0, n_rot_v: 0, k: Vec::new(), v: Vec::new() };
    let from = Sources { src, kept: keep.map(|k| k.0), below };
    for (i, (kv, t)) in kvs.iter().zip(&to.kvs).enumerate() {
        let ks = kept.as_ref().and_then(|k| k[i].stream.as_ref());
        from.write_cache(t, ks, kv.stream.as_ref().unwrap_or(&empty), &mut out)?;
    }
    let Out { mut bytes, jobs } = out;
    run_jobs(src, &mut bytes, &jobs, threads.max(1));
    Ok(bytes)
}

impl Sources<'_> {
    /// One cache of the output: its stream count, then one stream with the kept cells (target state `ks`, below
    /// `below`, rows copied) and the source's cells `s` (rows to convert), the other streams empty.
    fn write_cache(&self, t: &Kv, ks: Option<&Stream>, s: &Stream, out: &mut Out) -> Result<(), String> {
        let ts = t.stream.as_ref().expect("probe streams are non-empty");
        out.bytes.extend_from_slice(&t.n_stream.to_le_bytes());
        // the kept cells: the target state's cells below `below`, rows copied
        let kc: Vec<usize> = ks.map_or(Vec::new(), |k| (0..k.cell_meta.len()).filter(|&c| k.cell_meta[c].0 < self.below).collect());
        if s.cell_meta.iter().any(|c| c.0 < self.below) {
            return Err(format!("the source holds a cell below the kept position {}", self.below));
        }
        let cells = kc.len() + s.cells as usize;
        if cells == 0 {
            for _ in 0..t.n_stream {
                out.bytes.extend_from_slice(&0u32.to_le_bytes());
            }
            return Ok(());
        }
        if s.cells > 0 && s.k.len() != ts.k.len() {
            return Err(format!("{} layers in the state, {} in the target", s.k.len(), ts.k.len()));
        }
        out.bytes.extend_from_slice(&(cells as u32).to_le_bytes());
        self.write_meta(ts, ks.map(|k| (k, &kc[..])), s, out);
        self.write_layers(ts, ks.map(|k| (k, &kc[..])), s, out)?;
        for _ in 1..t.n_stream {
            out.bytes.extend_from_slice(&0u32.to_le_bytes());
        }
        Ok(())
    }

    /// The cells' metadata (the kept cells', then the source's) and the data header of the output stream.
    fn write_meta(&self, ts: &Stream, ks: Option<(&Stream, &[usize])>, s: &Stream, out: &mut Out) {
        if let Some((k, kc)) = ks {
            for &c in kc {
                out.bytes.extend_from_slice(&self.kept.expect("kept")[k.cell_meta[c].1.clone()]);
            }
        }
        out.bytes.extend_from_slice(&self.src[s.meta.clone()]);
        for x in [ts.v_trans, ts.k.len() as u32, ts.n_rot_k, ts.n_rot_v] {
            out.bytes.extend_from_slice(&x.to_le_bytes());
        }
    }

    /// Every layer of the output stream (target stream `ts`): the K layers, then the V layers.
    fn write_layers(&self, ts: &Stream, ks: Option<(&Stream, &[usize])>, s: &Stream, out: &mut Out) -> Result<(), String> {
        for (part, tr) in [(0, &ts.k), (1, &ts.v)] {
            let rot = if part == 0 { ts.n_rot_k } else { ts.n_rot_v } as usize;
            for (l, b) in tr.iter().enumerate() {
                let kept = ks.map(|(k, kc)| (k.layer(part, l), kc));
                let conv = (s.cells > 0).then(|| (s.layer(part, l), s.cells));
                self.write_layer(b, rot, kept, conv, out)?;
            }
        }
        Ok(())
    }

    /// One layer of the output stream (target rows `b`, rotation `rot`): its type and row size, the kept cells' rows
    /// `kept` copied, and room for the source rows `conv` (of that many cells) with the job that converts them.
    fn write_layer(&self, b: &Rows, rot: usize, kept: Option<(&Rows, &[usize])>, conv: Option<(&Rows, u32)>, out: &mut Out) -> Result<(), String> {
        out.bytes.extend_from_slice(&b.ty.to_le_bytes());
        out.bytes.extend_from_slice(&b.row.to_le_bytes());
        if let Some((kr, kc)) = kept {
            if (kr.ty, kr.row) != (b.ty, b.row) {
                return Err(format!("the kept rows are type {} size {}, the target's {} {}", kr.ty, kr.row, b.ty, b.row));
            }
            let kb = self.kept.expect("kept");
            for &c in kc {
                let at = kr.data.start + c * kr.row as usize;
                out.bytes.extend_from_slice(&kb[at..at + kr.row as usize]);
            }
        }
        let Some((a, cells)) = conv else { return Ok(()) };
        if a.ty != T_F16 {
            return Err(format!("the source rows are type {}, not f16", a.ty));
        }
        let ne = (a.row / 2) as usize;
        if unsafe { ggml_row_size(b.ty, ne as i64) } as u64 != b.row || (rot > 0 && ne % rot != 0) || b.ty == T_F32 {
            return Err(format!("row of {ne} values: type {} row {} rotation {rot} does not fit", b.ty, b.row));
        }
        let at = out.bytes.len();
        out.bytes.resize(at + b.row as usize * cells as usize, 0);
        out.jobs.push(Job { src: a.data.clone(), ne, rot, ty: b.ty, dst: at..out.bytes.len() });
        Ok(())
    }
}

/// A piece of one layer's rows for one worker: source rows, values per row, rotation, target type, destination rows.
type Piece<'a> = (&'a [u8], usize, usize, i32, &'a mut [u8]);

/// Convert every job's rows into `out` on `threads` workers.
fn run_jobs(src: &[u8], out: &mut [u8], jobs: &[Job], threads: usize) {
    let pieces = split_pieces(src, out, jobs, threads);
    let per = pieces.len().div_ceil(threads).max(1);
    std::thread::scope(|sc| {
        let mut it = pieces.into_iter();
        loop {
            let group: Vec<_> = it.by_ref().take(per).collect();
            if group.is_empty() {
                break;
            }
            sc.spawn(move || {
                for (s, ne, rot, ty, d) in group {
                    convert_rows(s, ne, rot, ty, d);
                }
            });
        }
    });
}

/// Every layer's rows split into `threads` pieces of whole cells.
fn split_pieces<'a>(src: &'a [u8], out: &'a mut [u8], jobs: &[Job], threads: usize) -> Vec<Piece<'a>> {
    let mut pieces: Vec<Piece<'a>> = Vec::new();
    let mut rest: &mut [u8] = out;
    let mut taken = 0usize;
    for j in jobs {
        let (_, tail) = std::mem::take(&mut rest).split_at_mut(j.dst.start - taken);
        let (mut d, tail) = tail.split_at_mut(j.dst.len());
        rest = tail;
        taken = j.dst.end;
        let cells = j.src.len() / (2 * j.ne);
        let per = cells.div_ceil(threads).max(1);
        let (src_row, dst_row) = (2 * j.ne, j.dst.len() / cells.max(1));
        let mut c = 0;
        while c < cells {
            let n = per.min(cells - c);
            let (head, tail) = std::mem::take(&mut d).split_at_mut(n * dst_row);
            d = tail;
            pieces.push((&src[j.src.start + c * src_row..j.src.start + (c + n) * src_row], j.ne, j.rot, j.ty, head));
            c += n;
        }
    }
    pieces
}

#[cfg(test)]
mod tests;
