use Innovation;
use NeuronType;

#[derive(Clone, Copy, PartialOrd, PartialEq, Ord, Eq, Hash)]
pub struct Edge {
    id: Innovation,
    from: NeuronType,
    to: NeuronType,
    weight: i64,
}
