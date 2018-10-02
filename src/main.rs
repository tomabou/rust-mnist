extern crate rand;

mod data;
mod matrix;
mod network;

use network::Network;
use matrix::Vector;

fn main() {
    println!("Hello World");
    let test = data::test_data();
    let _train = data::train_data();

    let mut network = Network::new(28*28, 256, 10);
    let v = Vector::from_data(&test.image[0]);
    let res = network.forward(v);
    network.backward(res.clone());
    network.update();
    println!("{:?}",res );
}
