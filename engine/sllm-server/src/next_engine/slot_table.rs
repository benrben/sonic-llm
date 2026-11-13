//! Who owns each sequence slot. Every slot has exactly one owner, and it changes only through [`SlotTable::take`],
//! [`SlotTable::give_back`] and [`SlotTable::transfer`], each of which names the owner it expects: a slot freed twice (a
//! resume's read that ends after its agent was dropped, when a new sequence has taken the slot meanwhile) is refused
//! instead of putting a running slot on the free list. Pure bookkeeping: what the cells hold is `NextEngine::cells`.

use crate::engine::{AgentId, ReqId};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(super) enum SlotOwner {
    /// A new sequence may take it (its cells may still hold an ended sequence's prefix).
    Free,
    /// A plain completion's sequence.
    Job(ReqId),
    /// An agent's cells (its turns run in them).
    Agent(AgentId),
    /// A resume reads the agent's snapshot into it: the slot is the read's until the read ends, even when the agent was
    /// dropped or freed meanwhile.
    Reading(AgentId),
    /// A pinned start (a recurrent model's shared system prompt).
    Pinned,
}

/// The owner of every slot, and the free ones in the order they were freed (the order the engine has always picked them
/// in: the first of equals wins).
pub(super) struct SlotTable {
    owner: Vec<SlotOwner>,
    free: Vec<i32>,
}

impl SlotTable {
    /// `n` free slots.
    pub(super) fn new(n: usize) -> SlotTable {
        SlotTable { owner: vec![SlotOwner::Free; n], free: (0..n as i32).rev().collect() }
    }

    /// The free slots.
    pub(super) fn free(&self) -> &[i32] {
        &self.free
    }

    pub(super) fn is_free(&self, s: i32) -> bool {
        self.owner(s) == Some(&SlotOwner::Free)
    }

    pub(super) fn owner(&self, s: i32) -> Option<&SlotOwner> {
        usize::try_from(s).ok().and_then(|i| self.owner.get(i))
    }

    /// The free slot `s` goes to `to`.
    pub(super) fn take(&mut self, s: i32, to: SlotOwner) -> Result<(), String> {
        self.transfer(s, &SlotOwner::Free, to)
    }

    /// Slot `s`, held by `from`, is free again.
    pub(super) fn give_back(&mut self, s: i32, from: &SlotOwner) -> Result<(), String> {
        self.transfer(s, from, SlotOwner::Free)
    }

    /// Slot `s` passes from `from` to `to`; refused (nothing changes) when `from` does not hold it.
    pub(super) fn transfer(&mut self, s: i32, from: &SlotOwner, to: SlotOwner) -> Result<(), String> {
        match self.owner(s) {
            Some(o) if o == from => {}
            Some(o) => return Err(format!("slot {s} is held by {o:?}, not {from:?}: {to:?} refused")),
            None => return Err(format!("no slot {s}")),
        }
        if *from == SlotOwner::Free {
            self.free.retain(|&f| f != s);
        }
        if to == SlotOwner::Free {
            self.free.push(s);
        }
        self.owner[s as usize] = to;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn agent(id: &str) -> SlotOwner {
        SlotOwner::Agent(id.to_string())
    }

    #[test]
    fn slots_are_taken_in_the_order_they_were_freed() {
        let mut t = SlotTable::new(3);
        assert_eq!(t.free(), &[2, 1, 0]);
        t.take(1, SlotOwner::Job(7)).unwrap();
        t.take(2, agent("ag_1")).unwrap();
        assert_eq!(t.free(), &[0]);
        t.give_back(1, &SlotOwner::Job(7)).unwrap();
        assert_eq!(t.free(), &[0, 1]);
        assert!(t.is_free(1) && !t.is_free(2));
    }

    #[test]
    fn a_slot_is_freed_once() {
        let mut t = SlotTable::new(2);
        t.take(0, SlotOwner::Job(1)).unwrap();
        t.give_back(0, &SlotOwner::Job(1)).unwrap();
        assert!(t.give_back(0, &SlotOwner::Job(1)).is_err(), "a double free is refused");
        assert_eq!(t.free(), &[1, 0], "and the free list holds the slot once");
        assert!(t.take(5, SlotOwner::Pinned).is_err(), "no such slot");
    }

    /// The next engine's bug: an agent dropped (or freed) while its resume's read runs gave the reading slot back at once,
    /// and again when the read ended; a sequence that took the slot in between then sat on the free list while it ran.
    #[test]
    fn a_read_that_ends_after_its_agent_was_dropped_frees_its_slot_once() {
        let mut t = SlotTable::new(2);
        let reading = SlotOwner::Reading("ag_1".to_string());
        t.take(0, reading.clone()).unwrap();
        // the agent is dropped: the slot stays the read's, so a new sequence cannot take it
        assert!(!t.free().contains(&0));
        t.take(1, SlotOwner::Job(9)).unwrap();
        assert!(t.take(0, SlotOwner::Job(10)).is_err());
        // the read ends: its slot is free, once
        t.give_back(0, &reading).unwrap();
        t.take(0, SlotOwner::Job(10)).unwrap();
        assert!(t.give_back(0, &reading).is_err(), "a second end of the same read does not free the running job's slot");
        assert_eq!(t.owner(0), Some(&SlotOwner::Job(10)));
        assert!(t.free().is_empty());
    }

    #[test]
    fn a_restored_read_passes_its_slot_to_the_agent() {
        let mut t = SlotTable::new(1);
        let reading = SlotOwner::Reading("ag_1".to_string());
        t.take(0, reading.clone()).unwrap();
        t.transfer(0, &reading, agent("ag_1")).unwrap();
        assert!(t.give_back(0, &reading).is_err());
        t.give_back(0, &agent("ag_1")).unwrap();
        assert_eq!(t.free(), &[0]);
    }
}
