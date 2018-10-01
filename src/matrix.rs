use std::ops::Add;

#[derive(Debug,PartialEq)]
pub struct Matrix{
    shape: (i32,i32),
    val: Vec<f32>
}

impl Matrix{
    pub fn add(&mut self,m: &Matrix) -> &Matrix{
        assert_eq!(self.shape,m.shape);
        for i in 0..(self.shape.0*self.shape.1) as usize {
            self.val[i] += m.val[i];
        }
        self
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