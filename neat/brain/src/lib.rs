extern crate rand;
extern crate petgraph;
extern crate rayon;


pub mod neuron;
pub mod innovation;
pub mod edge;
pub mod phenotype;
pub mod lobe;

pub use edge::Edge;
pub use innovation::Innovation;
pub use neuron::{Neuron, NeuronType};
pub use phenotype::Phenotype;
pub use lobe::Lobe;

#[cfg(test)]
mod tests {
    #[test]
    fn it_works() {}
}
