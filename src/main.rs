#[macro_use]
extern crate error_chain;
extern crate reqwest;
extern crate tempdir;

use std::io::copy;
use std::fs::File;
use tempdir::TempDir;

error_chain! {
    foreign_links {
        Io(std::io::Error);
        HttpRequest(reqwest::Error);
    }
}

fn run() -> Result<()> {
    let tmp_dir = TempDir::new("example")?;
    let target = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz";
    let mut response = reqwest::get(target)?;

    let mut dest = {
        let fname = response
            .url()
            .path_segments()
            .and_then(|segments| segments.last())
            .and_then(|name| if name.is_empty() { None } else { Some(name) })
            .unwrap_or("tmp.bin");

        println!("file to download: '{}'", fname);
        let fname = tmp_dir.path().join(fname);
        println!("will be located under: '{:?}'", fname);
        File::create(fname)?
    };
    copy(&mut response, &mut dest)?;
    Ok(())
}

quick_main!(run);
