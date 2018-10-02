use rand;
use rand::distributions::{Distribution, Normal};
use std::f32;

#[derive(Debug, PartialEq,Clone)]
pub struct Matrix {
    shape: (usize, usize),
    val: Vec<f32>,
}

#[derive(Debug, PartialEq,Clone)]
pub struct Vector {
    val: Vec<f32>,
}

impl Matrix {
    pub fn mut_add(&mut self, m: &Matrix) {
        for i in 0..self.val.len(){
            self.val[i] += m.val[i];
        }
    }
    pub fn mut_madd(&mut self, m: &Matrix,x: f32) {
        for i in 0..self.val.len(){
            self.val[i] += m.val[i]*x;
        }
    }
    pub fn times(&mut self, m: f32){
        for i in 0..self.val.len(){
            self.val[i] *= m;
        }
    }
    pub fn from_vec(a: &Vector,b: &Vector) -> Matrix{
        let (x, y) = (a.val.len(),b.val.len());
        let mut res = Matrix {
            shape: (x,y),
            val: vec![0.0; x * y],
        };
        for i in 0..x {
            for j in 0..y {
                res.val[i*y+j] = a.val[i] * b.val[j];
            }
        }
        res
    }
    pub fn new(x: usize, y: usize) -> Matrix {
        let normal = Normal::new(0.0, 1.0/x as f64);
        let v = (0..x * y)
            .map(|_| normal.sample(&mut rand::thread_rng()) as f32)
            .collect();
        Matrix {
            shape: (y, x),
            val: v,
        }
    }
}

#[test]
fn test_from_vec(){
    let x = Vector {
        val: vec![1.0, 3.0],
    };
    let y = Vector {
        val: vec![1.0, 2.0,3.0],
    };
    let res = Matrix::from_vec(&x, &y);
    let m = Matrix {
        shape: (2, 3),
        val: vec![1.0, 2.0,3.0,3.0,6.0,9.0],
    };
    assert_eq!(m,res);
}

#[test]
fn test_add() {
    let mut m = Matrix {
        shape: (2, 2),
        val: vec![1.0, 2.0, 3.0, 4.0],
    };
    let n = Matrix {
        shape: (2, 2),
        val: vec![1.0, 2.0, 3.0, 4.0],
    };
    println!("{:?}", m);
    assert_eq!(m, n);
    m.mut_add(&n);
    let ans = Matrix {
        shape: (2, 2),
        val: vec![2.0, 4.0, 6.0, 8.0],
    };
    assert_eq!(m, ans);
}
pub fn mat_tmul(m: &Matrix, v: &Vector) -> Vector {
    assert!(m.shape.0 == v.val.len());
    let mut res = Vector {
        val: vec![0.0; m.shape.1],
    };
    for i in 0..m.shape.0 {
        for j in 0..m.shape.1 {
            res.val[j] += m.val[i * m.shape.1 + j] * v.val[i];
        }
    }
    res
}
#[test]
fn test_tmut_mul() {
    let x = Matrix {
        shape: (2, 2),
        val: vec![1.0, 2.0, 3.0, 4.0],
    };
    let y = Vector {
        val: vec![1.0, 2.0],
    };
    assert_eq!(
        mat_tmul(&x, &y),
        Vector {
            val: vec![7.0, 10.0]
        }
    );
}

pub fn mat_mul(m: &Matrix, v: &Vector) -> Vector {
    assert!(m.shape.1 == v.val.len());
    let mut res = Vector {
        val: vec![0.0; m.shape.0],
    };
    for i in 0..m.shape.0 {
        for j in 0..m.shape.1 {
            res.val[i] += m.val[i * m.shape.1 + j] * v.val[j];
        }
    }
    res
}
#[test]
fn test_mut_mul() {
    let x = Matrix {
        shape: (2, 2),
        val: vec![1.0, 2.0, 3.0, 4.0],
    };
    let y = Vector {
        val: vec![1.0, 2.0],
    };
    assert_eq!(
        mat_mul(&x, &y),
        Vector {
            val: vec![5.0, 11.0]
        }
    );
}

impl Vector {
    pub fn mut_add(&mut self, m: &Vector) {
        for i in 0..self.val.len(){
            self.val[i] += m.val[i];
        }
    }
    pub fn mut_madd(&mut self, m: &Vector, x: f32) {
        for i in 0..self.val.len(){
            self.val[i] += m.val[i] * x;
        }
    }
    pub fn add(mut self, m: &Vector)->Vector {
        for i in 0..self.val.len(){
            self.val[i] += m.val[i];
        }
        self
    }
    pub fn times(&mut self, m: f32){
        for i in 0..self.val.len(){
            self.val[i] *= m;
        }
    }
    pub fn relu(v: &Vector) -> Vector {
        let mut res = Vector {
            val: vec![0.0; v.val.len()],
        };
        for i in 0..v.val.len() {
            res.val[i] = if v.val[i] > 0.0 { v.val[i] } else { 0.0 };
        }
        res
    }

    pub fn new(x: usize) -> Vector{
        let normal = Normal::new(0.0, 1.0);
        let v = (0..x)
            .map(|_| normal.sample(&mut rand::thread_rng()) as f32)
            .collect();
        Vector {
            val: v,
        }
    }
    pub fn softmax(x: &Vector) -> Vector {
        let mut v = x.val.clone();
        for i in 0..v.len(){
            v[i] = x.val[i].exp();           
        }
        let mut sum = 0.0;
        for t in &v{
            sum += *t;
        }
        for t in &mut v{
            *t /= sum;
        }
        Vector{val:v}
    }
    pub fn from_data(d: &[u8]  ) -> Vector{
        let v = d.into_iter().map(|x|{
            *x as f32
        }).collect();
        Vector{val:v}
    }
    pub fn back(&self, a: &Vector) -> Vector{
        assert_eq!(self.val.len(),a.val.len());
        let mut res = Vector {
            val: vec![0.0; a.val.len()],
        };
        for i in 0..a.val.len() {
            res.val[i] = if a.val[i] > 0.0 { self.val[i] } else { 0.0 };
        }
        res
    }
}
#[test]
fn test_softmax(){
    let x = Vector {
        val: vec![0.5, 0.5],
    };
    let y = Vector {
        val: vec![1.0, 1.0],
    };
    assert_eq!(Vector::softmax(&y),x);
    let z = Vector {
        val: vec![2.0, 1.0,0.0],
    };
    println!("{:?}",Vector::softmax(&z))
}

