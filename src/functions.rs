use crate::{Tensor, TensorItem};
use num::Float;

impl<T: TensorItem + Float> Tensor<T> {
    pub fn log10(&self) -> Tensor<T> {
        Tensor{data: self.into_iter().map(|x| x.log10()).collect()}
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log10_test() {
        let t = Tensor::from(vec![1.0, 10.0, 100.0]);
        let tlog = t.log10();
        let tref = Tensor::from(vec![0.0, 1.0, 2.0]);
        assert_eq!(tref, tlog);
    }
}
