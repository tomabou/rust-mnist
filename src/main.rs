
fn main(){
    println!("Hello World");
    read_file();
}

use std::{fs, slice};
use std::io::{BufReader, Read};
use std::path::Path;

fn read_file() {
    let path = Path::new("../data/t10k-labels-idx1-ubyte");
    let mut f = BufReader::new(fs::File::open(&path).unwrap());
    let mut m: i32 = 0;
    let buf = unsafe{slice::from_raw_parts_mut(&mut m as *mut i32 as *mut u8, 4) };
    f.read_exact(buf).unwrap();
    m = i32::from_be(m);
    
    println!("{}",m);
}
