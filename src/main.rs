
#[allow(unused)]
use candlelighter::prelude::*;
use terminal_menu::{menu, label, button, list, scroll, run, mut_menu};
    
fn main() {
    let menu = menu(vec![
        label("Select an example ..."),
        scroll("examples", vec!["Simple DNN", "Simple CNN"]),
        button("exit")
    ]);
    run(&menu);
    {
        let mm = mut_menu(&menu);
        print!("{esc}c", esc = 27 as char);
        std::process::Command::new("clear").status().unwrap();
        println!("");
        if mm.selection_value("examples").eq("Simple DNN") {
            candlelighter::examples::simple_dnn::simple_dnn();
        }
        else if mm.selection_value("examples").eq("Simple CNN") {
            candlelighter::examples::simple_cnn::simple_cnn();
        }
    }
}
