mod matrix;

struct Network {
    W1: matrix::Matrix,
    W2: matrix::Matrix,
    W3: matrix::Matrix,
    b1: matrix::Vector,
    b2: matrix::Vector,
    b3: matrix::Vector,
    grad_W1: matrix::Matrix,
    grad_W2: matrix::Matrix,
    grad_W3: matrix::Matrix,
    grad_b1: matrix::Vector,
    grad_b2: matrix::Vector,
    grad_b3: matrix::Vector,
}

impl Network{
    pub fn new(I:usize, H: usize, C: usize) -> Network{
        W1: Matrix::new(I,H),
        W2: Matrix::new(H,H),
        W3: Matrix::new(H,C),
        b1: Vector::new(H),
        b2: Vector::new(H),
        b3: Vector::new(C),
        grad_W1: Matrix::new(I,H),
        grad_W2: Matrix::new(H,H),
        grad_W3: Matrix::new(H,C),
        grad_b1: Vector::new(H),
        grad_b2: Vector::new(H),
        grad_b3: Vector::new(C),
    }
    #[test]
    fn test_new(){
        new(2,3,4);
    }
}
