#[test]
fn rename_moves_a_path_to_another_id() {
    let mut t = PrefixTree::new();
    t.extend(7, &[1, 2, 3]);
    t.extend(2, &[1, 2, 9]);
    t.rename(7, 4);
    assert_eq!(t.len(7), 0);
    assert_eq!(t.len(4), 3);
    assert_eq!(t.longest_prefix(&[1, 2, 3, 5], 10, None), Some((4, 3)));
    assert_eq!(t.longest_end(&[1, 2, 3, 5], 10, None), Some((4, 3)));
    // it keeps working as a sequence: extend and remove leave no stale id behind
    t.extend(4, &[5]);
    assert_eq!(t.longest_prefix(&[1, 2, 3, 5], 10, None), Some((4, 4)));
    t.remove(4);
    t.remove(2);
    assert!(t.is_empty());
    assert_eq!(t.nodes_in_use(), 0);
    // renaming an unregistered id drops the target's path and registers nothing
    t.extend(1, &[8]);
    t.rename(6, 1);
    assert_eq!(t.len(1), 0);
}
use super::*;

/// The linear scan the engine used: the longest common prefix, capped at `max`, over every sequence but `skip`.
fn scan(seqs: &HashMap<i32, Vec<i32>>, ids: &[i32], max: usize, skip: Option<i32>) -> usize {
    seqs.iter().filter(|(&s, _)| Some(s) != skip).map(|(_, t)| t.iter().zip(ids).take(max).take_while(|(a, b)| a == b).count()).max().unwrap_or(0)
}

/// xorshift
struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn below(&mut self, n: u64) -> u64 {
        self.next() % n
    }
}

fn check(tree: &PrefixTree, seqs: &HashMap<i32, Vec<i32>>, ids: &[i32], max: usize, skip: Option<i32>) {
    let want = scan(seqs, ids, max, skip);
    match tree.longest_prefix(ids, max, skip) {
        None => assert_eq!(want, 0, "tree found nothing, the scan found {want}"),
        Some((sid, k)) => {
            assert_eq!(k, want);
            assert_ne!(Some(sid), skip);
            let t = &seqs[&sid];
            assert!(t.len() >= k && t[..k] == ids[..k], "the returned sequence really starts with the prefix");
        }
    }
}

#[test]
fn matches_the_linear_scan_on_random_data_through_inserts_cuts_and_removals() {
    let mut rng = Rng(0x9E3779B97F4A7C15);
    let mut tree = PrefixTree::new();
    let mut seqs: HashMap<i32, Vec<i32>> = HashMap::new();
    // a small alphabet and shared stems make long common prefixes likely
    let stems: Vec<Vec<i32>> = (0..4).map(|_| (0..40).map(|_| rng.below(3) as i32).collect()).collect();
    for step in 0..4000 {
        let sid = rng.below(24) as i32;
        match rng.below(10) {
            0..=4 => {
                // grow: a stem on a new sequence, else a few random tokens
                let add: Vec<i32> = if !seqs.contains_key(&sid) && rng.below(2) == 0 {
                    let s = &stems[rng.below(4) as usize];
                    s[..(5 + rng.below(35) as usize)].to_vec()
                } else {
                    (0..1 + rng.below(6)).map(|_| rng.below(3) as i32).collect()
                };
                tree.extend(sid, &add);
                seqs.entry(sid).or_default().extend(add);
            }
            5..=6 => {
                let keep = rng.below(60) as usize;
                tree.truncate(sid, keep);
                if let Some(t) = seqs.get_mut(&sid) {
                    t.truncate(keep);
                    if t.is_empty() {
                        seqs.remove(&sid);
                    }
                }
            }
            7 => {
                tree.remove(sid);
                seqs.remove(&sid);
            }
            _ => {
                let new: Vec<i32> = (0..rng.below(50)).map(|_| rng.below(3) as i32).collect();
                tree.set(sid, &new);
                seqs.remove(&sid);
                if !new.is_empty() {
                    seqs.insert(sid, new);
                }
            }
        }
        if step % 3 == 0 {
            let ids: Vec<i32> = if rng.below(2) == 0 {
                let s = &stems[rng.below(4) as usize];
                let mut v = s[..rng.below(40) as usize].to_vec();
                v.extend((0..rng.below(10)).map(|_| rng.below(3) as i32));
                v
            } else {
                (0..rng.below(70)).map(|_| rng.below(3) as i32).collect()
            };
            let max = if ids.is_empty() { 0 } else { ids.len() - 1 };
            let skip = if rng.below(2) == 0 { Some(rng.below(24) as i32) } else { None };
            check(&tree, &seqs, &ids, max, skip);
        }
    }
    // the trie is consistent: removing everything leaves nothing
    let all: Vec<i32> = seqs.keys().copied().collect();
    for s in all {
        tree.remove(s);
    }
    assert_eq!(tree.nodes_in_use(), 0);
    assert!(tree.is_empty());
}

#[test]
fn longest_end_finds_the_longest_whole_holder_prefix() {
    let mut t = PrefixTree::new();
    t.extend(100, &[1, 2, 3]);
    t.extend(101, &[1, 2, 3, 4, 5]);
    t.extend(102, &[1, 9]);
    assert_eq!(t.longest_end(&[1, 2, 3, 4, 5, 6, 7], 6, None), Some((101, 5)));
    assert_eq!(t.longest_end(&[1, 2, 3, 4, 6], 4, None), Some((100, 3)));
    assert_eq!(t.longest_end(&[1, 2, 3, 4, 5], 4, None), Some((100, 3))); // the 5-token holder is longer than max
    assert_eq!(t.longest_end(&[1, 2, 4], 2, None), None);
    assert_eq!(t.longest_end(&[1, 2, 3, 4, 5, 6], 5, Some(101)), Some((100, 3)));
}

#[test]
fn lookup_is_fast_with_many_sequences() {
    // 64 agents sharing a 2,000-token system prompt, each with 2,000 own tokens
    let mut t = PrefixTree::new();
    let sys: Vec<i32> = (0..2000).map(|i| (i * 7 % 1000) as i32).collect();
    let mut seqs: HashMap<i32, Vec<i32>> = HashMap::new();
    for a in 0..64 {
        let mut v = sys.clone();
        v.extend((0..2000).map(|i| ((i + a * 13) % 500) as i32 + 1000));
        t.extend(a, &v);
        seqs.insert(a, v);
    }
    let mut ids = sys.clone();
    ids.extend((0..500).map(|i| (i % 500) as i32 + 1000));
    let t0 = std::time::Instant::now();
    let r = t.longest_prefix(&ids, ids.len() - 1, None);
    let tree_us = t0.elapsed().as_secs_f64() * 1e6;
    let t0 = std::time::Instant::now();
    let want = scan(&seqs, &ids, ids.len() - 1, None);
    let scan_us = t0.elapsed().as_secs_f64() * 1e6;
    assert_eq!(r.map(|x| x.1), Some(want));
    eprintln!("prefix lookup, 64 sequences x 4000 tokens: trie {tree_us:.1} us, linear scan {scan_us:.1} us");
}
