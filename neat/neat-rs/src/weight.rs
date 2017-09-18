use rand::{self, Rand, Rng};

use rand::distributions::{IndependentSample, Range};
// use rand::distributions::normal::StandardNormal
/// Represent a connection weight.
#[derive(Debug, Clone, Copy, PartialOrd, PartialEq)]
pub struct Weight(pub f64);

impl Rand for Weight {
    fn rand<R: Rng>(rng: &mut R) -> Weight {
        let between = Range::new(0f64, 1f64);
        let mut rng = rand::thread_rng();
        Weight(between.ind_sample(&mut rng))
    }
}

impl Into<f64> for Weight {
    fn into(self) -> f64 {
        if self.0 > 1f64 {
            1f64
        } else if self.0 < 0f64 {
            0f64
        } else {
            self.0
        }
    }
}


#[test]

fn construct() {
    assert!(Weight::rand() < 1f64);
}
