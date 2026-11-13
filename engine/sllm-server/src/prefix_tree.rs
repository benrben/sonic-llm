//! A token-id trie of the prefixes the cache holds, for the engine's prefix lookup.
//!
//! Every sequence registered with [`PrefixTree::extend`] contributes its token path. Looking up a prompt walks the trie one
//! token at a time ([`PrefixTree::longest_prefix`]) instead of comparing it with every sequence's tokens; the work is the
//! length of the match, not the number of sequences times their length. The engine keeps the tree in step with the cache:
//! `extend` when tokens are prefilled or decoded, `truncate` on a cut or a rollback, `remove` when a sequence is freed,
//! suspended or paused. For models whose cache cannot be partly copied (sliding-window attention, recurrent state) only
//! the frozen prefix holders are registered, and [`PrefixTree::longest_end`] finds the longest holder that is a whole
//! prefix of the prompt.

use std::collections::HashMap;

const ROOT: u32 = 0;

struct Node {
    token: i32,
    parent: u32,
    children: Vec<(i32, u32)>,
    /// Sequences whose path goes through this node (it included).
    cnt: u32,
    /// Sequences whose path ends exactly here.
    ends: Vec<i32>,
}

pub struct PrefixTree {
    nodes: Vec<Node>,
    free: Vec<u32>,
    /// A sequence's node per token, in position order.
    paths: HashMap<i32, Vec<u32>>,
}

impl Default for PrefixTree {
    fn default() -> Self {
        PrefixTree { nodes: vec![Node { token: -1, parent: ROOT, children: Vec::new(), cnt: 0, ends: Vec::new() }], free: Vec::new(), paths: HashMap::new() }
    }
}

impl PrefixTree {
    pub fn new() -> PrefixTree {
        PrefixTree::default()
    }

    /// Tokens registered for `sid`.
    pub fn len(&self, sid: i32) -> usize {
        self.paths.get(&sid).map_or(0, |p| p.len())
    }

    pub fn is_empty(&self) -> bool {
        self.paths.is_empty()
    }

    /// Sequences registered.
    pub fn sequences(&self) -> usize {
        self.paths.len()
    }

    /// Nodes in use (the root excluded).
    pub fn nodes_in_use(&self) -> usize {
        self.nodes.len() - 1 - self.free.len()
    }

    fn child(&self, n: u32, t: i32) -> Option<u32> {
        self.nodes[n as usize].children.iter().find(|c| c.0 == t).map(|c| c.1)
    }

    fn alloc(&mut self, token: i32, parent: u32) -> u32 {
        let node = Node { token, parent, children: Vec::new(), cnt: 0, ends: Vec::new() };
        match self.free.pop() {
            Some(i) => {
                self.nodes[i as usize] = node;
                i
            }
            None => {
                self.nodes.push(node);
                (self.nodes.len() - 1) as u32
            }
        }
    }

    fn unend(&mut self, sid: i32) {
        if let Some(&last) = self.paths.get(&sid).and_then(|p| p.last()) {
            self.nodes[last as usize].ends.retain(|&s| s != sid);
        }
    }

    /// Append `toks` to the path of `sid` (a new sequence if it is not registered).
    pub fn extend(&mut self, sid: i32, toks: &[i32]) {
        if toks.is_empty() {
            return;
        }
        self.unend(sid);
        let mut path = self.paths.remove(&sid).unwrap_or_default();
        let mut cur = path.last().copied().unwrap_or(ROOT);
        for &t in toks {
            let next = match self.child(cur, t) {
                Some(n) => n,
                None => {
                    let n = self.alloc(t, cur);
                    self.nodes[cur as usize].children.push((t, n));
                    n
                }
            };
            self.nodes[next as usize].cnt += 1;
            path.push(next);
            cur = next;
        }
        self.nodes[cur as usize].ends.push(sid);
        self.paths.insert(sid, path);
    }

    /// Cut the path of `sid` back to its first `keep` tokens.
    pub fn truncate(&mut self, sid: i32, keep: usize) {
        let Some(len) = self.paths.get(&sid).map(|p| p.len()) else { return };
        if keep >= len {
            return;
        }
        self.unend(sid);
        let mut path = self.paths.remove(&sid).expect("present above");
        while path.len() > keep {
            let n = path.pop().expect("longer than keep");
            let node = &mut self.nodes[n as usize];
            node.cnt -= 1;
            if node.cnt == 0 {
                let (parent, token) = (node.parent, node.token);
                self.nodes[parent as usize].children.retain(|c| c.0 != token);
                self.free.push(n);
            }
        }
        if let Some(&last) = path.last() {
            self.nodes[last as usize].ends.push(sid);
            self.paths.insert(sid, path);
        } else if keep > 0 {
            unreachable!("keep > 0 leaves a node");
        }
    }

    pub fn remove(&mut self, sid: i32) {
        self.truncate(sid, 0);
    }

    /// The path of `from` now belongs to `to` (a sequence moved to another id); whatever `to` held is dropped.
    pub fn rename(&mut self, from: i32, to: i32) {
        if from == to {
            return;
        }
        self.remove(to);
        let Some(path) = self.paths.remove(&from) else { return };
        if let Some(&last) = path.last() {
            self.nodes[last as usize].ends.iter_mut().filter(|e| **e == from).for_each(|e| *e = to);
        }
        self.paths.insert(to, path);
    }

    /// Replace the path of `sid` with `toks`.
    pub fn set(&mut self, sid: i32, toks: &[i32]) {
        self.remove(sid);
        self.extend(sid, toks);
    }

    /// `cnt` of a node as seen without `skip`'s own path.
    fn eff(&self, n: u32, depth: usize, skip_path: Option<&Vec<u32>>) -> u32 {
        let c = self.nodes[n as usize].cnt;
        if skip_path.is_some_and(|p| p.get(depth) == Some(&n)) {
            c - 1
        } else {
            c
        }
    }

    /// The longest prefix of `ids` (at most `max` tokens) that some registered sequence other than `skip` starts with:
    /// `(sid of such a sequence, tokens)`. Same length as the longest common prefix over a scan of all sequences.
    pub fn longest_prefix(&self, ids: &[i32], max: usize, skip: Option<i32>) -> Option<(i32, usize)> {
        let skip_path = skip.and_then(|s| self.paths.get(&s));
        let mut cur = ROOT;
        let mut depth = 0usize;
        for &t in ids.iter().take(max) {
            let Some(n) = self.child(cur, t) else { break };
            if self.eff(n, depth, skip_path) == 0 {
                break;
            }
            cur = n;
            depth += 1;
        }
        if depth == 0 {
            return None;
        }
        // a sequence through `cur`: follow the children that other sequences use down to one that ends
        let mut n = cur;
        let mut d = depth; // depth of node n (its path length)
        loop {
            if let Some(&s) = self.nodes[n as usize].ends.iter().find(|&&s| Some(s) != skip) {
                return Some((s, depth));
            }
            let next = self.nodes[n as usize].children.iter().map(|c| c.1).find(|&c| self.eff(c, d, skip_path) > 0);
            match next {
                Some(c) => {
                    n = c;
                    d += 1;
                }
                None => return None, // cannot happen when counts are consistent
            }
        }
    }

    /// The longest registered sequence (other than `skip`) that is a whole prefix of `ids` and at most `max` tokens long.
    pub fn longest_end(&self, ids: &[i32], max: usize, skip: Option<i32>) -> Option<(i32, usize)> {
        let mut cur = ROOT;
        let mut best = None;
        for (i, &t) in ids.iter().take(max).enumerate() {
            let Some(n) = self.child(cur, t) else { break };
            cur = n;
            if let Some(&s) = self.nodes[n as usize].ends.iter().find(|&&s| Some(s) != skip) {
                best = Some((s, i + 1));
            }
        }
        best
    }
}

#[cfg(test)]
mod tests;
