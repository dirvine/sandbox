use Edge;
use NeuronType;
use petgraph::{Directed, graphmap};

/// Mix really of genotype and phenotype
#[allow(unused)]
pub struct Phenotype {
    genome: graphmap::GraphMap<NeuronType, Edge, Directed>,
}

#[allow(unused)]
impl Phenotype {
    fn new(inputs: NeuronType) {}

    fn set_inputs(&mut self, inputs: Vec<i64>) {}

    fn run(&mut self) {}

    fn activate(&mut self) {}

    fn add_neuron(&mut self) {}

    fn add_edge(&mut self) {}

    fn outputs(&self) -> Vec<i64> {
        vec![1i64]
    }
}
// Mate
// Crossover
// perturb weights (mutation)
// Chenge edges (mutation)
//


// TESTS ////////////////////////////////////////////////

#[cfg(test)]
use petgraph::graphmap::DiGraphMap;

#[test]
fn basic() {

    let mut g = DiGraphMap::new();
    g.add_edge("x", "y", -1);

    assert_eq!(g.node_count(), 2);
    assert_eq!(g.edge_count(), 1);
    assert!(g.contains_edge("x", "y"));
    assert!(!g.contains_edge("y", "x"));
    g.add_edge("y", "x", -1);
    assert!(g.contains_edge("y", "x"));

    g.add_edge("x", "z", -2);
    assert_eq!(g.node_count(), 3);
    assert_eq!(g.edge_count(), 3);
    assert!(g.contains_edge("x", "y"));
    assert!(g.contains_edge("x", "z"));

}
