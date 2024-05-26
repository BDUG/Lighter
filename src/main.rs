

#[allow(unused)]
use candlelighter::prelude::*;
use terminal_menu::{menu, label, button, scroll, run, mut_menu};


fn main() {
    //env::set_var("RUST_BACKTRACE", "1");   
    let menu = menu(vec![
        label("Select an example ..."),
        scroll("examples", 
            vec![
                "Simple DNN", 
                "Simple DNN2", 
                "Simple DNN3", 
                "Simple CNN",
                "Simple RNN",
                "Simple RNN2",
                "Simple S2S",
                "Simple TNN",
                "Simple ENN",
                "Simple PNN",
                "Simple LLM",
                "Simple LLM2",
                ]
            ),
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
        else if mm.selection_value("examples").eq("Simple DNN2") {
            candlelighter::examples::simple_dnn::simple_dnn2();
        }
        else if mm.selection_value("examples").eq("Simple DNN3") {
            candlelighter::examples::simple_dnn::simple_dnn3();
        }
        else if mm.selection_value("examples").eq("Simple CNN") {
            candlelighter::examples::simple_cnn::simple_cnn();
        }
        else if mm.selection_value("examples").eq("Simple RNN") {
            candlelighter::examples::simple_rnn::simple_rnn();
        }
        else if mm.selection_value("examples").eq("Simple RNN2") {
            candlelighter::examples::simple_rnn::simple_rnn2();
        }
        else if mm.selection_value("examples").eq("Simple S2S") {
            candlelighter::examples::simple_s2s::simple_s2s();
        }
        else if mm.selection_value("examples").eq("Simple TNN") {
            candlelighter::examples::simple_tnn::simple_tnn();
        }
        else if mm.selection_value("examples").eq("Simple ENN") {
            candlelighter::examples::simple_enn::simple_enn();
        }
        else if mm.selection_value("examples").eq("Simple PNN") {
            candlelighter::examples::simple_pnn::simple_pnn();
        }
        else if mm.selection_value("examples").eq("Simple LLM") {
            candlelighter::examples::simple_llm::simple_llm();
        }
        else if mm.selection_value("examples").eq("Simple LLM2") {
            candlelighter::examples::simple_llm::simple_llm2();
        }
    }
}
