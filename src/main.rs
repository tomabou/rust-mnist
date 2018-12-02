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

    for _ in 0..1000 {
        let mut loss = 0.0;
        for i in 0..1000 {
            let v = Vector::from_data(&test.image[i]);
            let res = network.forward(v);
            //println!("{:?}",res);
            loss += res.val[test.label[i] as usize];
            let mut dres = res.clone();
            dres.sub_label(test.label[i] as usize);
            network.backward(dres);
            network.update();
        }
        println!("{:?}", loss / 1000.0);
    }
}
