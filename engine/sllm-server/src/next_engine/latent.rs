//! Latent turns (`latent_steps`): silent steps fed vectors (a rescaled hidden state, or soft words) instead of tokens.

use super::*;

impl NextEngine {
    /// A hidden state (after the final norm) as the next latent input: scaled to the token table's mean row norm (LatentMAS's
    /// default; its realignment matrix is the identity for a tied output layer).
    /// Soft words for many output rows at once: one copy of their scores, the best words picked on several threads, one
    /// fetch of all their input rows. None when soft words are off or a copy fails (the caller falls back row by row).
    pub(super) fn soft_inputs(&self, ks: &[usize]) -> Option<Vec<Vec<f32>>> {
        let (temp, top) = self.soft?;
        if ks.is_empty() {
            return Some(Vec::new());
        }
        let (nv, n) = (self.n_vocab, self.n_embd);
        let mut rows = vec![0f32; ks.len() * nv];
        let kk: Vec<c_int> = ks.iter().map(|&k| k as c_int).collect();
        if let Err(m) = call(|b, l| unsafe { sn_logits_rows(self.e, kk.len() as c_int, kk.as_ptr(), rows.as_mut_ptr(), b, l) }) {
            eprintln!("latent: soft words: {m}; row by row instead");
            return None;
        }
        let eog = self.vocab.eog_ids();
        let k = top.min(nv);
        // per row: its K best (non end-of-turn) words and their weights softmax(score / T)
        let picks: Vec<(Vec<i32>, Vec<f64>)> = std::thread::scope(|sc| {
            let hs: Vec<_> = rows
                .chunks(nv)
                .map(|row| {
                    sc.spawn(move || {
                        let ids: Vec<i32> = (0..nv as i32).filter(|t| eog.binary_search(t).is_err()).collect();
                        soft_pick(row, ids, k, temp)
                    })
                })
                .collect();
            hs.into_iter().map(|h| h.join().expect("soft words thread")).collect()
        });
        let all: Vec<i32> = picks.iter().flat_map(|(ids, _)| ids.iter().copied()).collect();
        let mut emb = vec![0f32; all.len() * n];
        for (c, (ids, out)) in all.chunks(1024).zip(emb.chunks_mut(1024 * n)).enumerate() {
            let _ = c;
            if let Err(m) = call(|b, l| unsafe { sn_embed_rows(self.e, ids.len() as c_int, ids.as_ptr(), out.as_mut_ptr(), b, l) }) {
                eprintln!("latent: soft words: {m}; row by row instead");
                return None;
            }
        }
        let mut off = 0;
        Some(picks.iter().map(|(ids, w)| {
            let mut v = vec![0f64; n];
            for (i, p) in w.iter().enumerate() {
                for (x, &r) in v.iter_mut().zip(&emb[(off + i) * n..(off + i + 1) * n]) {
                    *x += p * r as f64;
                }
            }
            off += ids.len();
            v.into_iter().map(|x| x as f32).collect()
        }).collect())
    }

    /// The next silent input of output row k: soft words when on, else the rescaled hidden state h.
    pub(super) fn latent_input(&self, k: usize, h: &[f32]) -> Vec<f32> {
        let Some((temp, top)) = self.soft else { return self.latent_vec(h) };
        let mut row = vec![0f32; self.n_vocab];
        if let Err(m) = call(|b, n| unsafe { sn_logits(self.e, k as c_int, row.as_mut_ptr(), b, n) }) {
            eprintln!("latent: soft words: {m}; the hidden state instead");
            return self.latent_vec(h);
        }
        for &t in self.vocab.eog_ids() {
            if let Some(x) = row.get_mut(t as usize) {
                *x = f32::NEG_INFINITY;  // an end-of-turn word is not a thought
            }
        }
        let (ids, w) = soft_pick(&row, (0..row.len() as i32).collect(), top, temp);
        let kk = ids.len();
        let n = self.n_embd;
        let mut rows = vec![0f32; kk * n];
        if let Err(m) = call(|b, l| unsafe { sn_embed_rows(self.e, kk as c_int, ids.as_ptr(), rows.as_mut_ptr(), b, l) }) {
            eprintln!("latent: soft words: {m}; the hidden state instead");
            return self.latent_vec(h);
        }
        let mut v = vec![0f64; n];
        for (i, &p) in w.iter().enumerate() {
            for (x, &r) in v.iter_mut().zip(&rows[i * n..(i + 1) * n]) {
                *x += p * r as f64;
            }
        }
        v.into_iter().map(|x| x as f32).collect()
    }

    pub(super) fn latent_vec(&self, h: &[f32]) -> Vec<f32> {
        let aligned;
        let h = if self.realign.is_empty() {
            h
        } else {
            let n = self.n_embd;
            let mut out = vec![0f32; n];
            // out = h W_a (row vector times row-major matrix) = W_a^T h
            unsafe { cblas_sgemv(101, 112, n as c_int, n as c_int, 1.0, self.realign.as_ptr(), n as c_int, h.as_ptr(), 1, 0.0, out.as_mut_ptr(), 1) };
            aligned = out;
            &aligned[..]
        };
        let n = h.iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt().max(1e-6);
        let s = (self.emb_norm as f64 / n) as f32;
        h.iter().map(|&x| x * s).collect()
    }

    /// A latent turn (`latent_steps`): refused where silent cells could be lost or with a grammar; the first one measures the
    /// token table's norm and reads the latent settings (SLLM_LATENT_STOP, SLLM_LATENT_SOFT, SLLM_LATENT_REALIGN).
    pub(super) fn latent_setup(&mut self, req: &StartRequest) -> Result<(), EngineError> {
        // silent cells cannot be read again from tokens: not where a cut may lose cells (sliding windows, recurrent states)
        let windows = unsafe { sn_win_need(self.e, self.cfg.n_ctx as c_int) } > 0;
        if self.recurrent || windows || req.params.grammar.is_some() {
            return Err(err(501, "not_supported_error", "latent_steps: not for models with sliding-window layers or recurrent states, nor with a grammar"));
        }
        call(|b, n| unsafe { sn_latent(self.e, 0, std::ptr::null(), std::ptr::null(), std::ptr::null_mut(), b, n) })
            .map_err(|m| err(501, "not_supported_error", m))?;
        if self.emb_norm == 0.0 {
            self.emb_norm = unsafe { sn_embd_norm(self.e) };
            self.latent_settings()?;
        }
        Ok(())
    }

    /// The latent settings, read once: SLLM_LATENT_STOP, SLLM_LATENT_SOFT (and SLLM_LATENT_SOFT_GPU), SLLM_LATENT_REALIGN.
    fn latent_settings(&mut self) -> Result<(), EngineError> {
        if let Ok(v) = std::env::var("SLLM_LATENT_STOP") {
            self.lat_stop = v.split(',').filter_map(|x| x.trim().parse().ok()).collect();
            eprintln!("latent: a silent turn ends at its best word in {:?} or an end of turn", self.lat_stop);
        }
        if let Ok(v) = std::env::var("SLLM_LATENT_SOFT") {
            self.soft_words(&v);
        }
        if let Ok(path) = std::env::var("SLLM_LATENT_REALIGN") {
            let b = std::fs::read(&path).map_err(|e| err(500, "server_error", format!("SLLM_LATENT_REALIGN {path}: {e}")))?;
            if b.len() != self.n_embd * self.n_embd * 4 {
                return Err(err(500, "server_error", format!("SLLM_LATENT_REALIGN {path}: {} bytes, want {n}x{n} f32", b.len(), n = self.n_embd)));
            }
            self.realign = b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect();
            eprintln!("latent: realign matrix {path} ({n}x{n})", n = self.n_embd);
        }
        Ok(())
    }

    /// Soft words (SLLM_LATENT_SOFT="T,K"), made on the GPU unless SLLM_LATENT_SOFT_GPU=0 (or the GPU cannot).
    fn soft_words(&mut self, v: &str) {
        let mut it = v.split(',');
        let t: f32 = it.next().and_then(|x| x.trim().parse().ok()).unwrap_or(1.0);
        let k: usize = it.next().and_then(|x| x.trim().parse().ok()).unwrap_or(32);
        self.soft = Some((t.max(1e-4), k.clamp(1, 1024)));
        eprintln!("latent: soft words, temperature {t}, best {k}");
        if std::env::var("SLLM_LATENT_SOFT_GPU").map_or(true, |v| v != "0") {
            let eog = self.vocab.eog_ids();
            let r = call(|b, n| unsafe { sn_soft_config(self.e, t.max(1e-4), k.clamp(1, 1024) as c_int, eog.as_ptr(), eog.len() as c_int, b, n) });
            match r {
                Ok(()) => {
                    self.soft_gpu = true;
                    eprintln!("latent: soft words on the GPU");
                }
                Err(m) => eprintln!("latent: soft words on the CPU ({m})"),
            }
        }
    }

    /// Soft words made on the CPU for every silent output row of the step at once (by output row; empty when soft words are
    /// off, made on the GPU, or the batch failed: then row by row).
    pub(super) fn soft_pre(&self, rows: &StepRows) -> HashMap<usize, Vec<f32>> {
        let mut pre: HashMap<usize, Vec<f32>> = HashMap::new();
        if self.soft.is_some() && !self.soft_gpu {
            let mut ks = Vec::new();
            for &(r, k0, _) in &rows.groups {
                if self.jobs.get(&r).is_some_and(|j| j.lat > 0) {
                    ks.push(k0);
                }
            }
            for &(r, k) in &rows.out_req {
                if self.jobs.get(&r).is_some_and(|j| j.lat > 0) {
                    ks.push(k);
                }
            }
            if let Some(vs) = self.soft_inputs(&ks) {
                pre.extend(ks.into_iter().zip(vs));
            }
        }
        pre
    }
}

/// Soft words of one row of scores: of the candidate words `ids`, the `top` best, and their weights softmax(score / `temp`).
fn soft_pick(row: &[f32], mut ids: Vec<i32>, top: usize, temp: f32) -> (Vec<i32>, Vec<f64>) {
    let kk = top.min(ids.len());
    ids.select_nth_unstable_by(kk - 1, |a, b| row[*b as usize].total_cmp(&row[*a as usize]));
    ids.truncate(kk);
    let mx = ids.iter().map(|&t| row[t as usize]).fold(f32::NEG_INFINITY, f32::max);
    let w: Vec<f64> = ids.iter().map(|&t| (((row[t as usize] - mx) / temp) as f64).exp()).collect();
    let z: f64 = w.iter().sum();
    (ids, w.into_iter().map(|x| x / z).collect())
}
