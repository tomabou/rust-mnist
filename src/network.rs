use matrix::{Vector,Matrix};
use matrix;

pub struct Network {
    w1: Matrix,
    w2: Matrix,
    w3: Matrix,
    b1: Vector,
    b2: Vector,
    b3: Vector,
    grad_w1: Matrix,
    grad_w2: Matrix,
    grad_w3: Matrix,
    grad_b1: Vector,
    grad_b2: Vector,
    grad_b3: Vector,
}

impl Network{
    pub fn new(i:usize, h: usize, c: usize) -> Network{
        Network{
            w1: Matrix::new(i,h),
            w2: Matrix::new(h,h),
            w3: Matrix::new(h,c),
            b1: Vector::new(h),
            b2: Vector::new(h),
            b3: Vector::new(c),
            grad_w1: Matrix::new(i,h),
            grad_w2: Matrix::new(h,h),
            grad_w3: Matrix::new(h,c),
            grad_b1: Vector::new(h),
            grad_b2: Vector::new(h),
            grad_b3: Vector::new(c),
        }
    }

    pub fn forward(&self,v: Vector) -> Vector{
        let a1 = matrix::mat_mul(&self.w1, &v).add(&self.b1);
        let h1 = Vector::relu(&a1);
        let a2 = matrix::mat_mul(&self.w2, &h1).add(&self.b2);
        let h2 = Vector::relu(&a2);
        let y = matrix::mat_mul(&self.w3, &h2).add(&self.b3);
        Vector::softmax(&y)
    }
}
