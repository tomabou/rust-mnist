use std::ops::Add;
use std::ops::AddAssign;
use rand::distributions::{Normal, Distribution};
use rand;


#[derive(Debug,PartialEq)]
pub struct Matrix{
    shape: (usize,usize),
    val: Vec<f32>
}

#[derive(Debug,PartialEq)]
pub struct Vector{
    val: Vec<f32>
}

impl Matrix{
    fn add(&mut self,m: &Matrix){
        assert_eq!(self.shape,m.shape);
        for i in 0..self.shape.0*self.shape.1 {
            self.val[i] += m.val[i];
        }
    }
    fn transpose(m: &Matrix) -> Matrix{
        let (x,y) = m.shape;
        let mut res = Matrix{shape:(y,x), val:vec![0.0; x*y]};
        for i in 0..x{
            for j in 0..y{
                res.val[j*y+i] = m.val[i*x+j];
            }
        }
        res
    }    
    pub fn new(x: usize, y: usize)-> Matrix{
        let normal = Normal::new(0.0,1.0);
        let v = (0..x*y).map(|_|{
            normal.sample(&mut rand::thread_rng()) as f32
        }).collect();
        Matrix{shape:(y,x), val: v}
    }
}

#[test]
fn test_add(){
    let mut m = Matrix{shape:(2,2),val:vec![1.0,2.0,3.0,4.0]};
    let n = Matrix{shape:(2,2),val:vec![1.0,2.0,3.0,4.0]};
    println!("{:?}",m );
    assert_eq!(m,n);
    m.add(&n);
    let ans = Matrix{shape:(2,2),val:vec![2.0,4.0,6.0,8.0]};
    assert_eq!(m,ans);
}
#[test]
fn test_transpose(){
    let n = Matrix{shape:(2,2),val:vec![1.0,2.0,3.0,4.0]};
    assert_eq!(Matrix::transpose(&n),Matrix{shape:(2,2), val:vec![1.0,3.0,2.0,4.0]} );
}

pub fn mat_mul(m: &Matrix, v: &Vector) -> Vector{
    assert!(m.shape.1 == v.val.len());
    let mut res = Vector{val: vec![0.0;m.shape.0]};
    for i in 0..m.shape.0{
        for j in 0..m.shape.1{
            res.val[i] += m.val[i*m.shape.1+j] * v.val[j];
        }
    }
    res
}
#[test]
fn test_mut_mul(){
    let x = Matrix{shape:(2,2),val:vec![1.0,2.0,3.0,4.0]};
    let y = Vector{val:vec![1.0,2.0]};
    assert_eq!(mat_mul(&x, &y),Vector{val:vec![5.0,11.0]});
}

pub fn relu(v: &Vector) -> Vector{
    let mut res = Vector{val: vec![0.0;v.val.len()]};
    for i in 0..v.val.len(){
        res.val[i] = if v.val[i] > 0.0 {v.val[i]} else {0.0};
    }
    res
}