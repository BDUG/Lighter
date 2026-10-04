//! cargo run --no-default-features --example graph2text -- [graph.json] [output.json]
use candlelighter::graph2text::{verbalize, Graph, Options, Triple};
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    if args.len() > 2 {
        return Err("usage: graph2text [graph.json] [output.json]".into());
    }
    let graph = if let Some(path) = args.first() {
        Graph::from_json(&std::fs::read_to_string(path)?)?
    } else {
        Graph::from_triples(&[
            Triple {
                subject: "Ada Lovelace".into(),
                predicate: "wrote notes about".into(),
                object: "Analytical Engine".into(),
            },
            Triple {
                subject: "Charles Babbage".into(),
                predicate: "designed".into(),
                object: "Analytical Engine".into(),
            },
        ])?
    };
    let output = verbalize(&graph, &Options::default())?;
    if let Some(path) = args.get(1) {
        std::fs::write(path, serde_json::to_string_pretty(&output)?)?;
    } else {
        println!("{}", output.text);
    }
    Ok(())
}
