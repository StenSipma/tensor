use std::ops::{Add, Mul, Sub, Div, Rem};
use std::vec::IntoIter;
use num::Num;


#[derive(PartialEq, Debug)]
struct Tensor<T: Num> {
    data: Vec<T>,
}

impl<T: Num> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter() 
    }
    
}

impl<T: Num> Mul for Tensor<T> {
    type Output = Self;

    fn mul(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x * y).collect();
        Tensor { data }
    }
}

impl<T: Num> Rem for Tensor<T> {
    type Output = Self;

    fn rem(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x % y).collect();
        Tensor { data }
    }
}


impl<T: Num> Div for Tensor<T> {
    type Output = Self;

    fn div(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x / y).collect();
        Tensor { data }
    }
}

impl<T: Num> Add for Tensor<T> {
    type Output = Self;

    fn add(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x + y).collect();
        Tensor { data }
    }
}

impl<T: Num> Sub for Tensor<T> {
    type Output = Self;

    fn sub(self, other: Self) -> Self {
        let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x - y).collect();
        Tensor { data }
    }
}

#[cfg(test)]
mod tests {
    use crate::Tensor;

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
