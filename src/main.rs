
fn main(){
    println!("Hello World");
    read_test_label();
    read_train_label();
}

use std::{fs, slice};
use std::io::{BufReader, Read};
use std::path::Path;

fn read_label_file(path: &std::path::Path) {
    let mut f = BufReader::new(fs::File::open(path).unwrap());
    let mut n: i32 = 0;
    let buf = unsafe{slice::from_raw_parts_mut(&mut n as *mut i32 as *mut u8, 4) };
    f.read_exact(buf).unwrap();
    f.read_exact(buf).unwrap();
    n = i32::from_be(n);
    
    println!("number of lable: {}",n);
}

fn read_test_label(){
    let path = Path::new("./data/t10k-labels-idx1-ubyte");
    read_label_file(path)
}
fn read_train_label(){
    let path = Path::new("./data/train-labels-idx1-ubyte");
    read_label_file(path)
}