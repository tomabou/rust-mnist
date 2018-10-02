extern crate rand;

mod data;
mod matrix;
mod network;

use matrix::Vector;
use network::Network;

fn main() {
    println!("Hello World");
    let test = data::test_data();
    let _train = data::train_data();

    let mut network = Network::new(28 * 28, 256, 10);

    for _ in 0..100{
        let mut loss = 0.0;
        for i in 0..1000{
            let v = Vector::from_data(&test.image[0]);
            let res = network.forward(v);
            loss += res.val[test.label[0] as usize];
            let mut dres = res.clone();
            dres.sub_label(test.label[0] as usize);
            network.backward(dres);
            network.update();
        }
        println!("{:?}", loss/1000.0);
    }
}
