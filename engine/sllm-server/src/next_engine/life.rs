//! The lifecycle: a slot's cells to an SSD file and back.

use super::*;

impl NextEngine {
    /// A resume's cells are here (or could not be read): load them into `slot` and bring the agent back, or leave it an
    /// empty agent (its history is what is left).
    pub(super) fn restore(&mut self, id: &str, slot: i32, data: Result<&[u8], String>) -> Result<f64, String> {
        let e = self.e;
        let a = self.agents.get_mut(id).ok_or("the agent is gone")?;
        let off = a.off.take().ok_or("the agent is not suspended")?;
        let n = off.tokens.len();
        let want = unsafe { sn_kv_bytes(e, n as c_int) };
        let res = data.and_then(|d| {
            if d.len() != want {
                return Err(format!("the snapshot has {} bytes, {want} expected", d.len()));
            }
            call(|b, l| unsafe { sn_kv_load(e, slot, n as c_int, d.as_ptr(), b, l) })
        });
        if off.host.is_none() {
            let _ = std::fs::remove_file(&off.path); // a write still running is removed when it ends
        }
        let reading = SlotOwner::Reading(id.to_string());
        match res {
            Ok(()) => {
                a.slot = Some(slot);
                self.cells[slot as usize] = off.tokens;
                if let Err(m) = self.owners.transfer(slot, &reading, SlotOwner::Agent(id.to_string())) {
                    eprintln!("next: slot bookkeeping: {m}");
                }
                Ok(off.resumed.map_or(0.0, |t| t.elapsed().as_secs_f64() * 1e3))
            }
            Err(m) => {
                a.slot = None;
                a.pending.clear();
                let _ = call(|b, l| unsafe { sn_clear(e, slot, 0, n as c_int, b, l) });
                self.cells[slot as usize].clear();
                self.slot_back(slot, reading);
                Err(m)
            }
        }
    }

    pub(super) fn suspend(&mut self, id: &str) -> Result<SuspendReport, EngineError> {
        let dir = self.spill_dir.clone();
        let a = self.agent_at_rest(id, "queued or running", || err(409, "conflict_error", format!("agent {id} is already suspended")))?;
        let t0 = Instant::now();
        let slot = a.slot;
        let tokens = slot.map_or(Vec::new(), |s| self.cells[s as usize].clone());
        let n = tokens.len();
        let mut host = vec![0u8; unsafe { sn_kv_bytes(self.e, n as c_int) }];
        if let Some(s) = slot {
            call(|b, l| unsafe { sn_kv_save(self.e, s, n as c_int, host.as_mut_ptr(), b, l) }).map_err(server_err)?;
        }
        let stage_ms = t0.elapsed().as_secs_f64() * 1e3;
        std::fs::create_dir_all(&dir).map_err(|e| server_err(format!("{}: {e}", dir.display())))?;
        self.life.suspends += 1;
        let gen = self.life.suspends;
        let path = dir.join(format!("next-{id}-{gen}.kv"));
        let host = Arc::new(host);
        let bytes = host.len() as u64;
        // the write runs on its own thread; the slot is free now (its cells stay until another sequence takes it)
        let (tx, data, p, aid) = (self.io_tx.clone(), host.clone(), path.clone(), id.to_string());
        self.life.in_flight += 1;
        std::thread::spawn(move || {
            let t = Instant::now();
            let result = std::fs::write(&p, &data[..]).map(|_| (data.len() as u64, t.elapsed().as_secs_f64() * 1e3)).map_err(|e| format!("{}: {e}", p.display()));
            let _ = tx.send(Io::Written { agent: aid, gen, path: p, result });
        });
        let a = self.agents.get_mut(id).expect("checked above");
        a.slot = None;
        a.off = Some(Off { tokens, path, host: Some(host), reading: None, resumed: None, gen });
        // its pages go back now (those shared with other slots stay theirs): the room a suspend is asked for
        let own = slot.map_or(0, |s| self.held(s, true));
        if let Some(s) = slot {
            self.truncate(s, 0).map_err(server_err)?;
            self.slot_back(s, SlotOwner::Agent(id.to_string()));
        }
        let _ = bytes;
        if self.jobs.is_empty() {
            self.give_back(0);
        }
        Ok(SuspendReport { pages_freed: own, bytes_freed: own as u64 * unsafe { sn_page_bytes(self.e) } as u64, stage_ms })
    }

    pub(super) fn resume(&mut self, id: &str) -> Result<(), EngineError> {
        let t0 = Instant::now();
        let a = self.agent(id)?;
        let Some(off) = &a.off else {
            return Err(err(409, "conflict_error", format!("agent {id} is not suspended")));
        };
        if off.reading.is_some() {
            return Ok(()); // the read is on its way already
        }
        let tokens = off.tokens.clone();
        let host = off.host.clone();
        let path = off.path.clone();
        if self.resume_warm(id, &tokens, t0)? {
            return Ok(());
        }
        let reading = SlotOwner::Reading(id.to_string());
        let (slot, _) = self.place(&[], 0, false, reading.clone())?;
        if let Err(e) = self.make_room(chunks(tokens.len()), tokens.len(), slot) {
            self.slot_back(slot, reading);
            return Err(e);
        }
        let a = self.agents.get_mut(id).expect("checked above");
        let off = a.off.as_mut().expect("checked above");
        off.reading = Some(slot);
        off.resumed = Some(t0);
        match host {
            // the SSD write has not finished: load the host copy now
            Some(h) => {
                let result = self.restore(id, slot, Ok(&h));
                self.events.push(LifecycleEvent::Restored { agent: id.to_string(), result });
            }
            None => {
                let (tx, aid) = (self.io_tx.clone(), id.to_string());
                self.life.in_flight += 1;
                std::thread::spawn(move || {
                    let result = std::fs::read(&path).map_err(|e| format!("{}: {e}", path.display()));
                    let _ = tx.send(Io::Read { agent: aid, slot, result });
                });
            }
        }
        Ok(())
    }

    /// A free slot that still holds exactly the suspended agent's cells (`tokens`): the agent takes it, nothing to read.
    /// False when there is none.
    fn resume_warm(&mut self, id: &str, tokens: &[i32], t0: Instant) -> Result<bool, EngineError> {
        let Some(&s) = self.owners.free().iter().find(|&&s| self.cells[s as usize] == tokens) else { return Ok(false) };
        self.owners.take(s, SlotOwner::Agent(id.to_string())).map_err(server_err)?;
        let a = self.agents.get_mut(id).expect("checked above");
        let off = a.off.take().expect("checked above");
        a.slot = Some(s);
        if off.host.is_none() {
            let _ = std::fs::remove_file(&off.path);
        }
        self.life.warm_resumes += 1;
        self.events.push(LifecycleEvent::Restored { agent: id.to_string(), result: Ok(t0.elapsed().as_secs_f64() * 1e3) });
        Ok(true)
    }

    pub(super) fn drop_kv(&mut self, id: &str) -> Result<(), EngineError> {
        let a = self.agent(id)?;
        let slot = a.slot;
        if let Some(s) = slot {
            self.truncate(s, 0).map_err(server_err)?;
        }
        let a = self.agents.get_mut(id).expect("checked above");
        a.pending.clear();
        // (a slot a resume reads into stays the read's: it is free when the read ends, see `read_back`)
        if let Some(o) = a.off.take() {
            let _ = std::fs::remove_file(&o.path);
        }
        Ok(())
    }

    pub(super) fn poll_io(&mut self) -> Vec<LifecycleEvent> {
        while let Ok(io) = self.io_rx.try_recv() {
            self.life.in_flight -= 1;
            match io {
                Io::Written { agent, gen, path, result } => self.written(agent, gen, path, result),
                Io::Read { agent, slot, result } => self.read_back(agent, slot, result),
            }
        }
        std::mem::take(&mut self.events)
    }

    /// A suspend's SSD write ended: the host copy goes (a failed write keeps it for a resume), or the file when the agent
    /// no longer needs it.
    fn written(&mut self, agent: AgentId, gen: u64, path: PathBuf, result: Result<(u64, f64), String>) {
        let off = self.agents.get_mut(&agent).and_then(|a| a.off.as_mut()).filter(|o| o.gen == gen);
        let full = off.as_ref().map_or(0, |o| o.host.as_ref().map_or(0, |h| h.len() as u64));
        if off.is_none() {
            if let Err(m) = &result {
                // an earlier suspend's write: the agent's state now does not depend on it (the executor would drop its KV)
                eprintln!("next: {agent}: an earlier snapshot's write failed: {m}");
                let _ = std::fs::remove_file(&path);
                return;
            }
        }
        match off {
            Some(o) => {
                if let Ok((b, _)) = &result {
                    o.host = None; // the file has it now (a failed write keeps the host copy for a resume)
                    self.life.ssd_bytes += b;
                    self.life.writes += 1;
                }
            }
            // resumed, dropped or freed before the write ended: the file is not needed
            None => {
                let _ = std::fs::remove_file(&path);
            }
        }
        self.events.push(LifecycleEvent::Written { agent, result: result.map(|(b, ms)| (b, full, ms, true)) });
    }

    /// A resume's SSD read ended: the cells go into the slot it reads into (unless the agent was freed or dropped meanwhile).
    fn read_back(&mut self, agent: AgentId, slot: i32, result: Result<Vec<u8>, String>) {
        let reading = self.agents.get(&agent).and_then(|a| a.off.as_ref()).and_then(|o| o.reading);
        if reading != Some(slot) {
            // the agent was freed or dropped meanwhile: the read's slot is free now (once: it stayed the read's till here)
            self.slot_back(slot, SlotOwner::Reading(agent));
            return;
        }
        self.life.reads += 1;
        let result = match &result {
            Ok(data) => self.restore(&agent, slot, Ok(data)),
            Err(m) => self.restore(&agent, slot, Err(m.clone())),
        };
        self.events.push(LifecycleEvent::Restored { agent, result });
    }
}
