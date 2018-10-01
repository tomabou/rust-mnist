extern crate rand;

mod data;
mod matrix;
mod network;

fn main() {
    println!("Hello World");
    data::test_data();
    data::train_data();
}
