use rand::{Rand, Rng};

#[derive(PartialEq, PartialOrd, Ord, Eq, Hash, Clone, Copy, Debug)]
pub struct Weight {
    value: i64,
}


impl Weight {
    pub fn value(self) -> i64 {
        self.value
    }

    pub fn abs(&self) -> Weight {
        Weight { value: self.value.abs() }
    }
}

impl From<i64> for Weight {
    fn from(num: i64) -> Weight {
        Weight { value: num }
    }
}

impl Into<i64> for Weight {
    fn into(self) -> i64 {
        self.value()
    }
}
impl Rand for Weight {
    fn rand<R: Rng>(rng: &mut R) -> Self {
        Weight { value: rng.gen::<i64>() as i64 }
    }
}


#[cfg(test)]
use std::i64;

#[test]

fn it_works() {}
