use num::Integer;
use std::ops::Add;
use std::vec::IntoIter;


#[derive(PartialEq, Debug)]
struct Tensor<T: Integer> {
    data: Vec<T>,
}

impl<T: Integer> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter() 
    }
    
}

impl<T: Integer> Add for Tensor<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x + y).collect();
        Tensor { data }
    }
    
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn test_add() {
        let t1 = Tensor{data: vec![1, 2, 3]};
        let t2 = Tensor{data: vec![1, 1, 1]};

        let add = t1 + t2;
        let expect  = Tensor{data: vec![2, 3, 4]};
        assert_eq!(expect, add);
    }
}
