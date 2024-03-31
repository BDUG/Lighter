use std::collections::BinaryHeap;
use std::cmp::Ordering;

struct ValWithIndex {
    val: f32,
    index: usize,
}

impl PartialEq for ValWithIndex {
    fn eq(&self, other: &Self) -> bool {
        self.val == other.val
    }
}

impl Eq for ValWithIndex {
}


impl Ord for ValWithIndex {
    fn cmp(&self, other: &Self) -> Ordering {
        other.val.partial_cmp(&self.val).unwrap_or(Ordering::Equal)
    }
    
    fn max(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::max_by(self, other, Ord::cmp)
    }
    
    fn min(self, other: Self) -> Self
    where
        Self: Sized,
    {
        std::cmp::min_by(self, other, Ord::cmp)
    }
    
}

impl PartialOrd for ValWithIndex {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub struct TopK {
}

pub trait TopKTrait {
    fn top_k_positions(&self,nums: Vec<f32>, k: usize) -> Vec<usize>;
    fn new() -> Self;
}


impl TopKTrait for TopK {

    fn new() -> Self{
        Self {
        }
    }

    fn top_k_positions(&self, nums: Vec<f32>, k: usize) -> Vec<usize> {
        let mut heap = BinaryHeap::new();
    
        for (index, &val) in nums.iter().enumerate() {
            heap.push(ValWithIndex { val, index });
    
            if heap.len() > k {
                heap.pop();
            }
        }
    
        heap.into_iter().map(|vwi| vwi.index).collect()
    }
    
}
