use rand::Rng;

/// Select the best individual out of `k` randomly choosen.
/// This gives individuals with better fitness a higher chance to reproduce.
/// `n` is the total number of individuals.
///
/// NOTE: We are not using `sample(rng, 0..n, k)` as it is *very* expensive.
/// Instead we call `rng.gen_range()` k-times. The drawn items could be the same,
/// but the probability is very low if `n` is high compared to `k`.
#[inline]
#[allow(unused)]
pub fn tournament_selection_fast<R: Rng, F>(rng: &mut R,
                                            better_than: &F,
                                            n: usize,
                                            k: usize)
                                            -> usize
    where F: Fn(usize, usize) -> bool
{
    assert!(n > 0);
    assert!(k > 0);
    assert!(n >= k);

    let mut best: usize = rng.gen_range(0, n);

    for _ in 1..k {
        let i = rng.gen_range(0, n);
        if better_than(i, best) {
            best = i;
        }
    }

    best
}


#[allow(unused)]
#[inline]
/// Same as `tournament_selection_fast`, but return two individuls.
/// Retry `n_retries` times if they point to the same individual.
pub fn tournament_selection_fast2<R: Rng, F>(rng: &mut R,
                                             better_than: &F,
                                             n: usize,
                                             k: usize,
                                             n_retries: usize)
                                             -> (usize, usize)
    where F: Fn(usize, usize) -> bool
{
    let first = tournament_selection_fast(rng, better_than, n, k);

    let mut retries = 0;
    loop {
        let second = tournament_selection_fast(rng, better_than, n, k);

        if first != second || retries >= n_retries {
            return (first, second);
        }

        retries += 1;
    }
}
