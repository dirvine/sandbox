#[derive(Default, Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Innovation {
    current: u64,
}

impl Innovation {
    /// Get next innovation number (increments on read)
    pub fn next(&mut self) -> u64 {
        let num = self.current;
        self.current += 1;
        num
    }
    /// Get current innovation number
    pub fn current(&self) -> u64 {
        self.current
    }
}
