use std::slice::Iter;
use std::vec::IntoIter;
use num::Num;

// Abstract generic which every tensor item must satisfy
pub trait TensorItem: Num + Copy + {}
impl<T: Num + Copy> TensorItem for T {}

#[derive(PartialEq, Debug)]
pub struct Tensor<T: TensorItem> {
    data: Vec<T>,
}

impl<T: TensorItem> Tensor<T> {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }

    pub fn from(v: Vec<T>) -> Self {
        Self { data: v }
    }

    pub fn get_data(self) -> Vec<T> {
        self.data
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }
}

// Iterator Implementations
impl<'a, T: TensorItem> IntoIterator for &'a Tensor<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        (&self.data).into_iter()
    }
}

impl<T: TensorItem> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn iter_test() {
        let t = &Tensor{data: vec![3.0, 4.0, 7.0]};
        let mut sum = 0.;
        // Use once
        for i in t {
            sum += i;
        }

        // And another time, meaning it is not consumed.
        for j in t {
            sum += j
        }

        assert_eq!(28., sum);
    }

    #[test]
    fn len_test() {
        let t = Tensor{data: vec![1, 2, 3]};
        assert_eq!(3, t.len());

        let t: Tensor<i8> = Tensor{data: vec![]};
        assert_eq!(0, t.len());
    }

    #[test]
    fn new_test() {
        let t: Tensor<i8> = Tensor::new();
        assert_eq!(t.data, vec![]);
        assert_eq!(0, t.len());
    }

    #[test]
    fn from_vector_test() {
        let t1 = Tensor::from(vec![1, 2, 3, 4]);
        let t2 = Tensor{data: vec![1, 2, 3, 4]};
        assert_eq!(t1, t2);

        let t1 = Tensor::from(vec![1., 2., 3., 4.]);
        let t2 = Tensor{data: vec![1., 2., 3., 4.]};
        assert_eq!(t1, t2);
    }
}

// Operator implementations
mod op {
    use std::ops::{Add, Mul, Sub, Div, Rem};
    use crate::{TensorItem, Tensor};

    impl<T: TensorItem> Mul for Tensor<T> {
        type Output = Self;

        fn mul(self, other: Self) -> Self {
            let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x * y).collect();
            Tensor { data }
        }
    }

    impl<T: TensorItem> Rem for Tensor<T> {
        type Output = Self;

        fn rem(self, other: Self) -> Self {
            let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x % y).collect();
            Tensor { data }
        }
    }


    impl<T: TensorItem> Div for Tensor<T> {
        type Output = Self;

        fn div(self, other: Self) -> Self {
            let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x / y).collect();
            Tensor { data }
        }
    }

    impl<T: TensorItem> Add for Tensor<T> {
        type Output = Self;

        fn add(self, other: Self) -> Self {
            let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x + y).collect();
            Tensor { data }
        }
    }

    impl<T: TensorItem> Sub for Tensor<T> {
        type Output = Self;

        fn sub(self, other: Self) -> Self {
            let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x - y).collect();
            Tensor { data }
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;

        #[test]
        fn add_int_test() {
            let t1 = Tensor{data: vec![1, 2, 3]};
            let t2 = Tensor{data: vec![1, 1, -1]};

            let add = t1 + t2;
            let expect  = Tensor{data: vec![2, 3, 2]};
            assert_eq!(expect, add);
        }

        #[test]
        fn add_float_test() {
            let t1 = Tensor{data: vec![1.0, 2.0, 3.0]};
            let t2 = Tensor{data: vec![1.0, 1.0, -1.0]};

            let add = t1 + t2;
            let expect  = Tensor{data: vec![2.0, 3.0, 2.0]};
            assert_eq!(expect, add);
        }

        #[test]
        fn sub_int_test() {
            let t1 = Tensor{data: vec![1, 2, 3]};
            let t2 = Tensor{data: vec![1, 1, -1]};

            let sub = t1 - t2;
            let expect  = Tensor{data: vec![0, 1, 4]};
            assert_eq!(expect, sub);
        }

        #[test]
        fn sub_float_test() {
            let t1 = Tensor{data: vec![1.0, 2.0, 3.0]};
            let t2 = Tensor{data: vec![1.0, 1.0, -1.0]};

            let sub = t1 - t2;
            let expect  = Tensor{data: vec![0.0, 1.0, 4.0]};
            assert_eq!(expect, sub);
        }

        #[test]
        fn mul_float_test() {
            let t1 = Tensor{data: vec![1.0, 2.0, 3.0]};
            let t2 = Tensor{data: vec![2.0, 0.5, 3.0]};

            let mul = t1 * t2;
            let expect  = Tensor{data: vec![2.0, 1.0, 9.0]};
            assert_eq!(expect, mul);
        }

        #[test]
        fn mul_int_test() {
            let t1 = Tensor{data: vec![1, 2, 3]};
            let t2 = Tensor{data: vec![1, 2, 3]};

            let mul = t1 * t2;
            let expect  = Tensor{data: vec![1, 4, 9]};
            assert_eq!(expect, mul);
        }

        #[test]
        fn rem_int_test() {
            let t1 = Tensor{data: vec![3, 4, 7]};
            let t2 = Tensor{data: vec![2, 2, 7]};

            let rem = t1 % t2;
            let expect  = Tensor{data: vec![1, 0, 0]};
            assert_eq!(expect, rem);
        }

        #[test]
        fn rem_float_test() {
            let t1 = Tensor{data: vec![3.0, 4.0, 7.0]};
            let t2 = Tensor{data: vec![2.0, 2.0, 7.0]};

            let rem = t1 % t2;
            let expect  = Tensor{data: vec![1.0, 0.0, 0.0]};
            assert_eq!(expect, rem);
        }

        #[test]
        fn div_int_test() {
            let t1 = Tensor{data: vec![3, 4, 7]};
            let t2 = Tensor{data: vec![2, 2, 7]};

            let div = t1 / t2;
            let expect  = Tensor{data: vec![1, 2, 1]};
            assert_eq!(expect, div);
        }

        #[test]
        fn div_float_test() {
            let t1 = Tensor{data: vec![3.0, 4.0, 7.0]};
            let t2 = Tensor{data: vec![2.0, 2.0, 7.0]};

            let div = t1 / t2;
            let expect  = Tensor{data: vec![1.5, 2.0, 1.0]};
            assert_eq!(expect, div);
        }

    }
}
