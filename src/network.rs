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
    pub fn new() -> Network{

    }
}