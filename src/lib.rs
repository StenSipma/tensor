use std::iter::Sum;
use std::slice::{Iter, IterMut};
use std::vec::IntoIter;
use num::Num;

pub mod stats;

// Abstract generic which every tensor item must satisfy
pub trait TensorItem: Num + Copy + Sum + {}
impl<T: Num + Copy + Sum> TensorItem for T {}

#[derive(PartialEq, Debug, Clone)]
pub struct Tensor<T: TensorItem> {
    data: Vec<T>,
    dim: Vec<usize>,
}

impl<T: TensorItem> Tensor<T> {

    /// Create a new empty Tensor
    pub fn new() -> Self {
        Self { data: Vec::new(), dim: Vec::new() }
    }

    /// Gets the underlying Vec data from the tensor. This will consume the Tensor
    pub fn get_data(self) -> Vec<T> {
        self.data
    }

    /// Returns the length of the Tensor, i.e. the number of elements in the Tensor
    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Returns the number of dimensions of the Tensor
    pub fn ndim(&self) -> usize {
        self.dim.len()
    }

    /// Set the new dimension sizes for the Tensor, modifies the Tensor in place
    /// Will be None when the size of the new dimension does not match the existing 
    /// data
    pub fn set_dim(&mut self, dim: Vec<usize>) -> Option<()> {
        // First check the if the dimension is correct:
        let dim_len = {
            let mut total = 1;
            for d in &dim {
                total *= d;
            }
            total
        };

        if dim_len != self.data.len() {
            return None
        }

        // Update the dim vector with the new values
        self.dim.clear();
        self.dim.extend(dim);

        Some(())
    }

    /// Iterate over the elements of the Tensor (flattened)
    pub fn iter(&self) -> Iter<'_, T> {
        self.data.iter()
    }

    pub fn index(&self, index: Vec<usize>) -> Option<T> {
        if index.len() != self.dim.len() {
            return None
        }

        // Check if each index is within bounds
        for (a, b) in index.iter().zip(&self.dim) {
            if a + 1 > *b {
                return None
            }
        }

        let mut data_index = 0;
        for (n, idx) in index.iter().rev().enumerate() {
            let dim_idx = self.dim.len() - (n + 1);

            data_index *= self.dim[dim_idx];
            data_index += idx
        }
        Some(self.data[data_index])
    }

    /// TODO: use the internal ndim instead!!!!!!!!!!, also make it column major order!
    ///
    /// Iterate over the tensor in a row-major order. The desired shape of the
    /// tensor is given in `shape` which should have size 2.
    /// Moreover the following should be satisfied:
    ///     shape[0]*shape[1] == tensor.len()
    pub fn iter_2d(&self) -> impl Iterator<Item = ((usize, usize), &T)> {
        let coordinates = (0..self.dim[0])
            .map(move |x| { (0..self.dim[1]).map(move |y| (x, y)) })
            .flatten();
        coordinates.zip(self.into_iter())
    }
}


/// Implements the From trait to initialize a Tensor from a Vec of the same type
impl<T: TensorItem> From<Vec<T>> for Tensor<T> {
    fn from(v: Vec<T>) -> Self {
        let dim = if v.len() == 0 {
            Vec::new()
        } else {
            vec![v.len()]
        };
        Self { data: v, dim }
    }
}

/// Implements the IntoIterator trait for a Tensor
impl<T: TensorItem> IntoIterator for Tensor<T> {
    type Item = T;
    type IntoIter = IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.data.into_iter()
    }
}

/// Implements the IntoIterator trait for reference to a Tensor
impl<'a, T: TensorItem> IntoIterator for &'a Tensor<T> {
    type Item = &'a T;
    type IntoIter = Iter<'a, T>;

    fn into_iter(self) -> Iter<'a, T> {
        (&self.data).into_iter()
    }
}

/// Implements the IntoIterator trait for reference to a mutable Tensor
impl<'a, T: TensorItem> IntoIterator for &'a mut Tensor<T> {
    type Item = &'a mut T;
    type IntoIter = IterMut<'a, T>;

    fn into_iter(self) -> IterMut<'a, T> {
        (&mut self.data).into_iter()
    }
}

#[allow(unused_macros)]
mod types {
    macro_rules! all_types {
        () => {
            u8, u16, u32, u64, u128, i8, i16, i32, i64, i128, f32, f64, usize, isize
        };
    }
}


#[cfg(test)]
mod tests {
    use crate::Tensor;

    #[test]
    fn iter_test() {
        // let t = &Tensor{data: };
        let t = &Tensor::from(vec![3.0, 4.0, 7.0]);
        let mut sum = 0.;
        // Use once
        for i in t {
            sum += i;
        }

        // And another time, meaning it is not consumed.
        for j in t.iter() {
            sum += j
        }

        // And another time, meaning it is not consumed.
        for j in t {
            sum += j
        }

        assert_eq!(3.*14., sum);
    }

    #[test]
    fn len_test() {
        let t = Tensor::from(vec![1, 2, 3]);
        assert_eq!(3, t.len());

        let t: Tensor<i8> = Tensor::from(Vec::new());
        assert_eq!(0, t.len());
    }

    #[test]
    fn new_test() {
        let t: Tensor<i8> = Tensor::new();
        assert_eq!(t.data, vec![]);
        assert_eq!(0, t.len());

        assert_eq!(0, t.ndim());
    }

    #[test]
    fn from_test() {
        let t1 = Tensor::from(vec![1, 2, 3, 4]);
        let t2 = Tensor{data: vec![1, 2, 3, 4], dim: vec![4]};
        assert_eq!(t1, t2);

        let t1 = Tensor::from(vec![1., 2., 3., 4.]);
        let t2 = Tensor{data: vec![1., 2., 3., 4.], dim: vec![4]};
        assert_eq!(t1, t2);

        let t1: Tensor<f32> = vec![1., 2., 3., 4.].into();
        let t2 = Tensor{data: vec![1., 2., 3., 4.], dim: vec![4]};
        assert_eq!(t1, t2);
    }

    #[test]
    fn set_ndim_test() {
        let t0 = Tensor::from(vec![1, 2, 3, 4]);
        let mut t2 = Tensor::from(vec![1, 2, 3, 4]);
        let res = t2.set_dim(vec![2, 2]);
        assert_eq!(res, Some(()));

        let res = t2.set_dim(vec![4]);
        assert_eq!(res, Some(()));

        assert_eq!(t0, t2);

        // Invalid dimensions!
        let res = t2.set_dim(vec![2, 1]);
        assert_eq!(res, None);
        let res = t2.set_dim(vec![5]);
        assert_eq!(res, None);

        assert_eq!(t0, t2);

        let res = t2.set_dim(vec![1, 1, 1, 4]);
        assert_eq!(res, Some(()));
    }

    #[test]
    fn index_1d_test() {
        // One dimension
        let t1 = Tensor::from(vec![1, 2, 3, 4]);
        assert_eq!(t1.index(vec![0]), Some(1));
        assert_eq!(t1.index(vec![3]), Some(4));
        assert_eq!(t1.index(vec![4]), None);
    }

    #[test]
    fn index_0d_test() {
        // Empty array will always be None
        let t0: Tensor<i32> = Tensor::new();
        assert_eq!(t0.index(vec![0]), None);
    }

    #[test]
    fn index_2d_test() {
        // Two dimensions
        let mut t2 = Tensor::from(vec![1, 2, 3, 4]);
        t2.set_dim(vec![2, 2]);

        assert_eq!(t2.index(vec![0, 0]), Some(1));
        assert_eq!(t2.index(vec![1, 0]), Some(2));
        assert_eq!(t2.index(vec![0, 1]), Some(3));
        assert_eq!(t2.index(vec![1, 1]), Some(4));
        assert_eq!(t2.index(vec![0, 2]), None);
    }

    #[test]
    fn index_nd_test() {
        // Two dimensions
        let mut tn = Tensor::from(vec![1, 2, 3, 4]);
        tn.set_dim(vec![1, 1, 1, 1, 4]);

        assert_eq!(tn.index(vec![0, 0, 0, 0, 2]), Some(3));
    }

    #[test]
    fn iter_2d_test() {
        let mut t = Tensor::from(vec![1, 2, 3, 4]);
        let shape: Vec<usize> = vec![2, 2];
        t.set_dim(shape);
        let mut it = t.iter_2d();

        assert_eq!(Some(((0, 0), &1)), it.next());
        assert_eq!(Some(((0, 1), &2)), it.next());
        assert_eq!(Some(((1, 0), &3)), it.next());
        assert_eq!(Some(((1, 1), &4)), it.next());
        assert_eq!(None, it.next());

        let mut t = Tensor::from(vec![1, 2, 3, 4, 5, 6]);
        let shape: Vec<usize> = vec![2, 3];
        t.set_dim(shape);
        let mut it = t.iter_2d();

        assert_eq!(Some(((0, 0), &1)), it.next());
        assert_eq!(Some(((0, 1), &2)), it.next());
        assert_eq!(Some(((0, 2), &3)), it.next());
        assert_eq!(Some(((1, 0), &4)), it.next());
        assert_eq!(Some(((1, 1), &5)), it.next());
        assert_eq!(Some(((1, 2), &6)), it.next());
        assert_eq!(None, it.next());
    }
}

// Operator implementations
mod operators {
    use std::ops::{Add, Mul, Sub, Div, Rem};
    use crate::{TensorItem, Tensor};

    /// Given a operator, implement the operation with a scalar T and a Tensor<T>
    /// for the following combinations:
    /// T OP Tensor<T>
    /// Tensor<T> OP T
    /// T OP &Tensor<T>
    /// &Tensor<T> OP T
    macro_rules! impl_op_tens_sca {
        ($trait:ident, $trait_fn:ident, $scalar_type:path) => {
            /// Implements the $trait operator trait for $scalar_type and &Tensor
            impl $trait<$scalar_type> for &Tensor<$scalar_type> {
                type Output = Tensor<$scalar_type>;

                fn $trait_fn(self, other: $scalar_type) -> Self::Output {
                    let data: Vec<$scalar_type> = self.into_iter().map(|x| (*x).$trait_fn(other)).collect();
                    Tensor::from( data )
                }
            }

            /// Implements the $trait operator trait for $scalar_type and Tensor
            impl $trait<$scalar_type> for Tensor<$scalar_type> {
                type Output = Tensor<$scalar_type>;

                fn $trait_fn(self, other: $scalar_type) -> Self::Output {
                    let data: Vec<$scalar_type> = self.into_iter().map(|x| x.$trait_fn(other)).collect();
                    Tensor::from( data )
                }
            }

            /// Implements the $trait operator trait for &Tensor and $scalar_type
            impl $trait<&Tensor<$scalar_type>> for $scalar_type {
                type Output = Tensor<$scalar_type>;

                fn $trait_fn(self, other: &Tensor<$scalar_type>) -> Self::Output {
                    let data: Vec<$scalar_type> = other.into_iter().map(|x| self.$trait_fn(*x)).collect();
                    Tensor::from( data )
                }
            }

            /// Implements the $trait operator trait for Tensor and $scalar_type
            impl $trait<Tensor<$scalar_type>> for $scalar_type {
                type Output = Tensor<$scalar_type>;

                fn $trait_fn(self, other: Tensor<$scalar_type>) -> Self::Output {
                    let data: Vec<$scalar_type> = other.into_iter().map(|x| self.$trait_fn(x)).collect();
                    Tensor::from( data )
                }
            }
        };
    }

    /// Implements the macro impl_op_tens_sca for all relevant types
    /// TODO: Change to use the `all_types` macro
    macro_rules! impl_op_tens_sca_all {
        ($trait:ident, $trait_fn:ident) => {
            impl_op_tens_sca!($trait, $trait_fn, u8);
            impl_op_tens_sca!($trait, $trait_fn, u16);
            impl_op_tens_sca!($trait, $trait_fn, u32);
            impl_op_tens_sca!($trait, $trait_fn, u64);
            impl_op_tens_sca!($trait, $trait_fn, u128);

            impl_op_tens_sca!($trait, $trait_fn, i8);
            impl_op_tens_sca!($trait, $trait_fn, i16);
            impl_op_tens_sca!($trait, $trait_fn, i32);
            impl_op_tens_sca!($trait, $trait_fn, i64);
            impl_op_tens_sca!($trait, $trait_fn, i128);

            impl_op_tens_sca!($trait, $trait_fn, f32);
            impl_op_tens_sca!($trait, $trait_fn, f64);

            impl_op_tens_sca!($trait, $trait_fn, usize);
            impl_op_tens_sca!($trait, $trait_fn, isize);
        };
    }


    /// Given a operator trait (e.g. Mul) and its associated op function
    /// (e.g. mul) will implement the operator for the Tensor for all
    /// combinations of Tensor<T> and &Tensor<T>:
    /// - &Tensor, &Tensor
    /// - Tensor , &Tensor
    /// - &Tensor, Tensor
    /// - Tensor , Tensor
    macro_rules! impl_op_tens_tens {
        ($trait:ident, $trait_fn:ident) => {
            // &Tensor, &Tensor
            impl<T: TensorItem> $trait for &Tensor<T> {
                type Output = Tensor<T>;

                fn $trait_fn(self, other: Self) -> Self::Output {
                    let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| (*x).$trait_fn(*y)).collect();
                    Tensor::from( data )
                }
            }

            // &Tensor, Tensor
            impl<T: TensorItem> $trait<Tensor<T>> for &Tensor<T> {
                type Output = Tensor<T>;

                fn $trait_fn(self, other: Tensor<T>) -> Self::Output {
                    let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| (*x).$trait_fn(y)).collect();
                    Tensor::from( data )
                }
            }

            // Tensor, &Tensor
            impl<T: TensorItem> $trait<&Tensor<T>> for Tensor<T> {
                type Output = Tensor<T>;

                fn $trait_fn(self, other: &Tensor<T>) -> Self::Output {
                    let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x.$trait_fn(*y)).collect();
                    Tensor::from( data )
                }
            }

            // Tensor, Tensor
            impl<T: TensorItem> $trait for Tensor<T> {
                type Output = Self;

                fn $trait_fn(self, other: Self) -> Self {
                    let data: Vec<T> = self.into_iter().zip(other.into_iter()).map(|(x, y)| x.$trait_fn(y)).collect();
                    Tensor ::from( data )
                }
            }
        };
    }

    /// Macro to implement the operator for all Tensor x Tensor and
    /// Tensor x Scalar combinations (mostly convenience)
    macro_rules! impl_op_tens {
        ($trait:ident, $trait_fn:ident) => {
            impl_op_tens_tens!($trait, $trait_fn);
            impl_op_tens_sca_all!($trait, $trait_fn);
        };
    }

    impl_op_tens!(Mul, mul);
    impl_op_tens!(Div, div);
    impl_op_tens!(Rem, rem);
    impl_op_tens!(Add, add);
    impl_op_tens!(Sub, sub);

    #[cfg(test)]
    mod tests {
        use super::*;

        // Basic tests in the form of:
        // t1 OP t2 == expected
        macro_rules! test_op_basic {
            ($name:ident, $trait_op:ident, $t1:expr, $t2:expr, $expected:expr) => {
                #[test]
                fn $name() {
                    let t1 = Tensor::from($t1);
                    let t2 = Tensor::from($t2);
                    let expected = Tensor::from($expected);
                    assert_eq!(expected, t1.$trait_op(t2));
                }
            };
        }

        test_op_basic!(add_int, add, vec![1, 2, 3], vec![1, 1, -1], vec![2, 3, 2]);
        test_op_basic!(add_float, add, vec![1.0, 2.0, 3.0], vec![1.0, 1.0, -1.0], vec![2.0, 3.0, 2.0]);

        test_op_basic!(sub_int, sub, vec![1, 2, 3], vec![1, 1, -1], vec![0, 1, 4]);
        test_op_basic!(sub_float, sub, vec![1.0, 2.0, 3.0], vec![1.0, 1.0, -1.0], vec![0.0, 1.0, 4.0]);

        test_op_basic!(mul_int, mul, vec![1, 2, 3], vec![1, 2, 3], vec![1, 4, 9]);
        test_op_basic!(mul_float, mul, vec![1.0, 2.0, 3.0], vec![2.0, 0.5, 3.0], vec![2.0, 1.0, 9.0]);

        test_op_basic!(rem_int, rem, vec![3, 4, 7], vec![2, 2, 7], vec![1, 0, 0]);
        test_op_basic!(rem_float, rem, vec![3.0, 4.0, 7.0], vec![2.0, 2.0, 7.0], vec![1.0, 0.0, 0.0]);

        test_op_basic!(div_int, div, vec![3, 4, 7], vec![2, 2, 7], vec![1, 2, 1]);
        test_op_basic!(div_float, div, vec![3.0, 4.0, 7.0], vec![2.0, 2.0, 7.0], vec![1.5, 2.0, 1.0]);

        #[test]
        fn mul_op_ref_combinations_tens_tens_test() {
            let t1 = Tensor::from(vec![3.0, 4.0, 7.0]);
            let t2 = Tensor::from(vec![2.0, 2.0, 7.0]);
            let expect = Tensor::from(vec![1.5, 2.0, 1.0]);
            assert_eq!(expect, &t1 / &t2);

            assert_eq!(expect, &t1 / t2);
            let t2 = Tensor::from(vec![2.0, 2.0, 7.0]);
            assert_eq!(expect, t1 / &t2);

            let t1 = Tensor::from(vec![3.0, 4.0, 7.0]);
            let t2 = Tensor::from(vec![2.0, 2.0, 7.0]);
            assert_eq!(expect, t1 / t2);
        }

        #[test]
        fn mul_op_ref_combinations_tens_sca_test() {
            let t1 = Tensor::from(vec![3.0, 4.0, 7.0]);
            let expect = Tensor::from(vec![1.5, 2.0, 3.5]);
            assert_eq!(expect, &t1 / 2.);
            assert_eq!(expect, t1 / 2.);

            let t1 = Tensor::from(vec![3.0, 4.0, 7.0]);
            let expect = Tensor::from(vec![28.0, 21.0, 12.0]);
            assert_eq!(expect, 84. / &t1);
            assert_eq!(expect, 84. / t1);
        }

    }
}
