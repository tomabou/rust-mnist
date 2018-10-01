use matrix::{Vector,Matrix};
use matrix;

struct Network {
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
    pub fn new(I:usize, H: usize, C: usize) -> Network{
        Network{
            w1: Matrix::new(I,H),
            w2: Matrix::new(H,H),
            w3: Matrix::new(H,C),
            b1: Vector::new(H),
            b2: Vector::new(H),
            b3: Vector::new(C),
            grad_w1: Matrix::new(I,H),
            grad_w2: Matrix::new(H,H),
            grad_w3: Matrix::new(H,C),
            grad_b1: Vector::new(H),
            grad_b2: Vector::new(H),
            grad_b3: Vector::new(C),
        }
    }

    pub fn forward(&self,v: Vector) -> Vector{
        let a1 = matrix::mat_mul(&self.w1, &v).add(&self.b1);
        let h1 = Vector::relu(&a1);
        let a2 = matrix::mat_mul(&self.w2, &h1);
        let h2 = Vector::relu(&a2);
        let a3 = matrix::mat_mul(&self.w3, &h2);
        let h3 = Vector::relu(&a3);
        Vector::softmax(&h3)
    }
}
