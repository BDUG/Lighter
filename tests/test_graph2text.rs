use candlelighter::graph2text::*;
use serde_json::json;
fn graph() -> Graph {
    let mut a = Node::new("a", "Alice");
    a.properties.insert("age".into(), json!(42));
    a.properties
        .insert("metadata".into(), json!({"tags":["science"],"active":true}));
    Graph {
        nodes: vec![
            a,
            Node::new("b", "Bob"),
            Node::new("c", "Carol"),
            Node::new("d", "Isolated"),
        ],
        edges: vec![
            Edge::new("ab", "a", "knows", "b"),
            Edge::new("ba", "b", "admires", "a"),
            Edge::new("bc", "b", "knows", "c"),
            Edge::new("aa", "a", "knows", "a"),
        ],
    }
}
#[test]
fn all_facts_and_values_survive_roundtrip() {
    let g = graph();
    assert_eq!(Graph::from_json(&g.to_json().unwrap()).unwrap(), g);
    let result = verbalize(&g, &Options::default()).unwrap();
    assert_eq!(result.plan.facts.len(), 10);
    assert!(result.plan.omitted_fact_ids.is_empty());
    assert!(result.text.contains("42"));
    assert!(result.text.contains("science"));
    assert!(result.text.contains("Isolated"));
    assert!(result
        .text
        .contains("\"Bob\" (\"b\") — \"admires\" → \"Alice\""));
}
#[test]
fn stable_order_independent_of_input_order() {
    let g = graph();
    let mut reversed = g.clone();
    reversed.nodes.reverse();
    reversed.edges.reverse();
    assert_eq!(
        verbalize(&g, &Options::default()).unwrap().text,
        verbalize(&reversed, &Options::default()).unwrap().text
    );
}
#[test]
fn radius_cycles_and_fact_budget_have_explicit_omissions() {
    let g = graph();
    let p = plan(
        &g,
        &Options {
            root: Some("a".into()),
            radius: Some(1),
            max_facts: None,
        },
    )
    .unwrap();
    assert_eq!(p.facts.len(), 7);
    assert_eq!(p.omitted_fact_ids, vec!["edge:bc", "node:c", "node:d"]);
    let p = plan(
        &g,
        &Options {
            root: Some("c".into()),
            radius: None,
            max_facts: Some(2),
        },
    )
    .unwrap();
    assert_eq!(p.facts[0].id, "node:c");
    assert_eq!(p.facts.len(), 2);
    assert_eq!(p.omitted_fact_ids.len(), 8);
    let p = plan(
        &g,
        &Options {
            root: Some("a".into()),
            radius: Some(0),
            max_facts: None,
        },
    )
    .unwrap();
    assert!(p.facts.iter().any(|f| f.id == "edge:aa"));
    assert!(!p.facts.iter().any(|f| f.id == "edge:ab"));
}
#[test]
fn rejects_invalid_graphs_and_options() {
    assert!(Graph::default().validate().is_err());
    let mut g = graph();
    g.nodes.push(g.nodes[0].clone());
    assert!(g.validate().is_err());
    let mut g = graph();
    g.edges.push(g.edges[0].clone());
    assert!(g.validate().is_err());
    let mut g = graph();
    g.edges[0].target = "absent".into();
    assert!(g.validate().is_err());
    let mut g = graph();
    g.nodes[0].properties.insert(" ".into(), json!(null));
    assert!(g.validate().is_err());
    assert!(plan(
        &graph(),
        &Options {
            root: Some("absent".into()),
            ..Default::default()
        }
    )
    .is_err());
    assert!(plan(
        &graph(),
        &Options {
            radius: Some(1),
            ..Default::default()
        }
    )
    .is_err());
    assert!(plan(
        &graph(),
        &Options {
            max_facts: Some(0),
            ..Default::default()
        }
    )
    .is_err());
    assert!(Graph::from_json(r#"{"nodes":[],"edges":[],"unknown":true}"#).is_err());
}
#[test]
fn duplicate_labels_parallel_edges_unicode_and_escaping() {
    let g = Graph {
        nodes: vec![
            Node::new("a", "同じ\n\"label\""),
            Node::new("b", "同じ\n\"label\""),
        ],
        edges: vec![
            Edge::new("1", "a", "likes", "b"),
            Edge::new("2", "a", "likes", "b"),
        ],
    };
    let text = verbalize(&g, &Options::default()).unwrap().text;
    assert_eq!(text.lines().count(), 4);
    assert!(text.contains("\\n\\\"label\\\""));
    let p = prompt(&g, &Options::default()).unwrap();
    let facts = p
        .split("FACTS_JSON:\n")
        .nth(1)
        .unwrap()
        .split("\nDESCRIPTION:")
        .next()
        .unwrap();
    assert_eq!(serde_json::from_str::<Vec<Fact>>(facts).unwrap().len(), 4);
}
#[test]
fn triples_merge_entities_but_preserve_relations() {
    let triple = Triple {
        subject: "A".into(),
        predicate: "knows".into(),
        object: "B".into(),
    };
    let g = Graph::from_triples(&[triple.clone(), triple]).unwrap();
    assert_eq!(g.nodes.len(), 2);
    assert_eq!(g.edges.len(), 2);
    assert!(Graph::from_triples(&[]).is_err());
}
#[cfg(feature = "native")]
#[test]
fn native_pipeline_generates_and_cleans_engine() {
    use candlelighter::native::*;
    struct Backend;
    impl NativeBackend for Backend {
        fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
            Ok(vec![0])
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens
                .iter()
                .filter(|&&x| x == 1)
                .map(|_| "answer")
                .collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
            Ok(if position == 1 {
                vec![-100.0, 100.0, -100.0]
            } else {
                vec![-100.0, -100.0, 100.0]
            })
        }
    }
    let mut engine = NativeEngine::new(Backend, 1, vec![2]).unwrap();
    let (p, r) = generate(
        &mut engine,
        &graph(),
        &Options::default(),
        SamplingParams {
            temperature: 0.0,
            max_tokens: 4,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p.facts.len(), 10);
    assert_eq!(r.text, "answer");
    assert!(engine.is_idle());
}
