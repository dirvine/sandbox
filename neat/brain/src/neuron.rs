use innovation::Innovation;
use rand;
use rayon::prelude::*;

/// We have an innovation number to see where in evolution this Neuron was
/// created. We also have
/// a bias that we will use in RelU activation.
#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Neuron {
    innovation: Innovation,
    activation: i64,
}

impl Neuron {
    /// Create a neuron
    pub fn new(innovation: Innovation) -> Neuron {
        Neuron {
            innovation: innovation,
            activation: rand::random::<i64>().abs(),
        }
    }
}

#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub enum NeuronType {
    Input(Neuron),
    Hidden(Neuron),
    Output(Neuron),
}

impl NeuronType {
    /// Use a form of ReLu here
    pub fn activate(&mut self, inputs: Vec<i64>) -> i64 {
        match *self {
            NeuronType::Input(n) => n.activation,
            NeuronType::Hidden(n) |
            NeuronType::Output(n) => {
                inputs
                    .par_iter()
                    .filter(|&x| x < &n.activation)
                    .map(|&x| x - &n.activation)
                    .sum()
            }
        }
    }

    /// Randomise activation bias
    pub fn perturb(&mut self, innovation: Innovation) {
        match *self {
            NeuronType::Input(_) => {}
            NeuronType::Hidden(mut n) |
            NeuronType::Output(mut n) => {
                n.activation = rand::random::<i64>().abs();
                n.innovation = innovation;
            }
        }
    }
}
