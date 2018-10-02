use matrix::{Vector,Matrix};
use matrix;

pub struct Network {
    w1: Matrix,
    w2: Matrix,
    w3: Matrix,
    b1: Vector,
    b2: Vector,
    b3: Vector,
    a1: Vector,
    a2: Vector,
    dw1: Matrix,
    dw2: Matrix,
    dw3: Matrix,
    db1: Vector,
    db2: Vector,
    db3: Vector,
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
            a1: Vector::new(h),
            a2: Vector::new(h),
            dw1: Matrix::new(i,h),
            dw2: Matrix::new(h,h),
            dw3: Matrix::new(h,c),
            db1: Vector::new(h),
            db2: Vector::new(h),
            db3: Vector::new(c),
        }
    }

    pub fn forward(&mut self,v: Vector) -> Vector{
        self.a1 = matrix::mat_mul(&self.w1, &v).add(&self.b1);
        let h1 = Vector::relu(&self.a1);
        self.a2 = matrix::mat_mul(&self.w2, &h1).add(&self.b2);
        let h2 = Vector::relu(&self.a2);
        let y = matrix::mat_mul(&self.w3, &h2).add(&self.b3);
        Vector::softmax(&y)
    }
    pub fn backward(&self, gy: Vector){
        let gh2 = matrix::mat_tmul(&self.w3, &gy);
        let ga2 = gh2.back(&self.a2);
        let gh1 = matrix::mat_tmul(&self.w2, &ga2);
        let ga1 = gh1.back(&self.a1);
    }

    fn calc_d(&mut self, gw1:&Matrix,gw2:&Matrix,gw3:&Matrix,gb1:&Vector,gb2:&Vector,gb3:&Vector){
        self.db1.times(0.9);
        self.db1.mut_add(gb1);
        self.db2.times(0.9);
        self.db2.mut_add(gb2);
        self.db3.times(0.9);
        self.db3.mut_add(gb3);
        self.dw1.times(0.9);
        self.dw1.mut_add(gw1);
        self.dw2.times(0.9);
        self.dw2.mut_add(gw2);
        self.dw3.times(0.9);
        self.dw3.mut_add(gw3);
    }

    fn update(&mut self) {
        let lr = 0.001;
        self.b1.mut_madd(&self.db1, lr);
        self.b2.mut_madd(&self.db2, lr);
        self.b3.mut_madd(&self.db3, lr);
        self.w1.mut_madd(&self.dw1, lr);
        self.w2.mut_madd(&self.dw2, lr);
        self.w3.mut_madd(&self.dw3, lr);
    }
}
