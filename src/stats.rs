use crate::{TensorItem, Tensor};

impl<T: TensorItem> Tensor<T> {
    /// Sums all the elements in the tensor
    pub fn sum(&self) -> T {
        self.into_iter().map(|x| *x).sum()
    }
}
