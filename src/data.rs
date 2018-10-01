pub struct DataSet {
    label: Vec<u8>,
    image: Vec<[u8; 28 * 28]>,
}

use std::io::{BufReader, Read};
use std::path::Path;
use std::{fs, slice};

fn read_label_file(path: &Path) -> Vec<u8> {
    let mut f = BufReader::new(fs::File::open(path).unwrap());
    let mut n: i32 = 0;
    let buf = unsafe { slice::from_raw_parts_mut(&mut n as *mut i32 as *mut u8, 4) };
    f.read_exact(buf).unwrap();
    f.read_exact(buf).unwrap();
    n = i32::from_be(n);
    let mut label = vec![0; n as usize];
    f.read_exact(&mut label).unwrap();
    println!("number of lable: {}", n);
    label
}

pub fn test_data() -> DataSet {
    DataSet {
        label: read_test_label(),
        image: read_test_image(),
    }
}

pub fn train_data() -> DataSet {
    DataSet {
        label: read_train_label(),
        image: read_train_image(),
    }
}

fn read_test_label() -> Vec<u8> {
    let path = Path::new("./data/t10k-labels-idx1-ubyte");
    read_label_file(path)
}
fn read_train_label() -> Vec<u8> {
    let path = Path::new("./data/train-labels-idx1-ubyte");
    read_label_file(path)
}

fn read_image_file(path: &Path) -> Vec<[u8; 28 * 28]> {
    let mut f = BufReader::new(fs::File::open(path).unwrap());
    let mut n: i32 = 0;
    let buf = unsafe { slice::from_raw_parts_mut(&mut n as *mut i32 as *mut u8, 4) };
    f.read_exact(buf).unwrap();
    f.read_exact(buf).unwrap();
    let number = i32::from_be(n);
    f.read_exact(buf).unwrap();
    n = i32::from_be(n);
    assert_eq!(n, 28);
    f.read_exact(buf).unwrap();
    n = i32::from_be(n);
    assert_eq!(n, 28);
    let mut images = vec![[0; 28 * 28]; number as usize];
    for i in 0..number as usize {
        f.read_exact(&mut images[i]).unwrap();
    }
    println!("number of lable: {}", number);
    images
}

fn read_test_image() -> Vec<[u8; 28 * 28]> {
    let path = Path::new("./data/t10k-images-idx3-ubyte");
    read_image_file(path)
}
fn read_train_image() -> Vec<[u8; 28 * 28]> {
    let path = Path::new("./data/train-images-idx3-ubyte");
    read_image_file(path)
}
