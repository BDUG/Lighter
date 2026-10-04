//! Model-free scheduler, sampling, constraints, and workflow examples.
#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::*;
    use candlelighter::native_prompt::*;
    use serde_json::json;

    // Deliberately tiny backend: token 0 = a, token 1 = b, token 2 = EOS.
    struct Demo;
    impl NativeBackend for Demo {
        fn encode(&self, _: &str) -> NativeResult<Vec<u32>> {
            Ok(vec![0])
        }
        fn decode(&self, tokens: &[u32]) -> NativeResult<String> {
            Ok(tokens
                .iter()
                .filter_map(|id| match id {
                    0 => Some('a'),
                    1 => Some('b'),
                    _ => None,
                })
                .collect())
        }
        fn logits(&mut self, _: &str, _: &[u32], position: usize) -> NativeResult<Vec<f32>> {
            Ok(if position < 3 {
                vec![4., 1., -10.]
            } else {
                vec![-10., -10., 4.]
            })
        }
    }
    let mut engine = NativeEngine::new(Demo, 2, vec![2])?;
    let sampling = SamplingParams {
        max_tokens: 8,
        temperature: 0.0,
        seed: 42,
        logprobs: Some(2),
        ..Default::default()
    };
    for id in ["first", "second", "cancel"] {
        engine.submit(GenerateRequest {
            id: id.into(),
            prompt: "demo".into(),
            sampling: sampling.clone(),
            constraint: None,
        })?;
    }
    assert!(engine.cancel("cancel"));
    let responses = engine.run_to_completion()?;
    assert_eq!(responses.len(), 2); // Queued cancellation removes the request without a response.
    assert_eq!(responses.iter().filter(|r| r.text == "aa").count(), 2);
    for response in responses {
        println!(
            "{}: {:?} {:?}",
            response.id, response.text, response.finish_reason
        );
    }

    engine.submit(GenerateRequest {
        id: "active-cancel".into(),
        prompt: "demo".into(),
        sampling: sampling.clone(),
        constraint: None,
    })?;
    assert!(engine.step()?.is_empty());
    assert!(engine.cancel("active-cancel"));
    assert_eq!(
        engine.run_to_completion()?[0].finish_reason,
        FinishReason::Cancelled
    );

    // Sampling controls are independent of the model implementation.
    SamplingParams {
        temperature: 0.7,
        top_k: Some(2),
        top_p: 0.9,
        min_p: 0.05,
        presence_penalty: 0.1,
        frequency_penalty: 0.1,
        repetition_penalty: 1.1,
        min_tokens: 1,
        max_tokens: 8,
        stop: vec!["ab".into()],
        stop_token_ids: vec![2],
        bad_token_ids: vec![1],
        logit_bias: [(0, 0.2)].into(),
        seed: 42,
        ..Default::default()
    }
    .validate()?;

    for spec in [
        ConstraintSpec::Choice {
            choices: vec!["a".into(), "b".into()],
        },
        ConstraintSpec::Regex {
            pattern: "a+".into(),
        },
        ConstraintSpec::Json,
        ConstraintSpec::Gbnf {
            grammar: "root ::= \"a\" | \"b\"".into(),
            root: "root".into(),
        },
    ] {
        let constraint = spec.compile()?;
        assert!(constraint.allows_prefix(""));
    }
    engine.submit(GenerateRequest {
        id: "constrained".into(),
        prompt: "demo".into(),
        sampling,
        constraint: Some(ConstraintSpec::Choice {
            choices: vec!["a".into()],
        }),
    })?;
    assert_eq!(engine.run_to_completion()?[0].text, "a");

    struct Executor;
    impl PromptExecutor for Executor {
        fn generate(
            &mut self,
            _: &str,
            _: usize,
            _: Option<&str>,
            _: Option<&dyn OutputConstraint>,
        ) -> NativeResult<String> {
            Ok("hello".into())
        }
        fn select(&mut self, _: &str, choices: &[String]) -> NativeResult<String> {
            Ok(choices[0].clone())
        }
    }
    let branch = Workflow {
        steps: vec![WorkflowStep::Transform {
            input: "answer".into(),
            output: "uppercase".into(),
            operation: Transform::Uppercase,
        }],
    };
    let workflow = Workflow {
        steps: vec![
            WorkflowStep::Prompt {
                template: PromptTemplate::parse("{{gen answer max_tokens=8}}")?,
            },
            WorkflowStep::Fork {
                branches: vec![branch],
            },
        ],
    };
    let mut state = Variables::new();
    workflow.execute(&mut Executor, &mut state)?;
    assert_eq!(state["branch_0"]["uppercase"], json!("HELLO"));
    let call = ToolCall::from_openai_json(r#"{"name":"search","arguments":{"query":"Rust"}}"#)?;
    // Each constraint exposes distinct prefix/completion semantics.
    let regex = RegexConstraint::new("a+")?;
    assert!(regex.is_complete("aaa"));
    assert!(!regex.is_complete("b"));
    assert!(regex.allows_prefix("b")); // Current regex API does not prune prefixes.
    assert!(JsonConstraint.is_complete(r#"{"ok":true}"#));
    let choice = ChoiceConstraint::new(vec!["yes".into(), "no".into()])?;
    assert!(choice.allows_prefix("ye"));
    assert!(!choice.allows_prefix("maybe"));
    let grammar = GbnfGrammar::parse("root ::= \"start\" | \"stop\"", "root")?;
    assert!(grammar.is_complete("start"));
    assert!(!grammar.allows_prefix("quit"));
    let mut json_state = Variables::from([("record".into(), json!({"name":"Rust"}))]);
    let transforms = Workflow {
        steps: vec![
            WorkflowStep::Transform {
                input: "record".into(),
                output: "name".into(),
                operation: Transform::JsonPointer("/name".into()),
            },
            WorkflowStep::Transform {
                input: "name".into(),
                output: "lower".into(),
                operation: Transform::Lowercase,
            },
        ],
    };
    transforms.execute(&mut Executor, &mut json_state)?;
    assert_eq!(json_state["lower"], json!("rust"));
    let definition = ToolDefinition {
        name: "search".into(),
        description: "Demo search schema".into(),
        input_schema: json!({"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
    };
    let request = McpRequest {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "tools/call".into(),
        params: json!({"name":definition.name,"arguments":call.arguments}),
    };
    let request_text = serde_json::to_string(&request)?;
    let parsed: McpRequest = serde_json::from_str(&request_text)?;
    assert_eq!(parsed.method, "tools/call");
    let response = McpResponse {
        jsonrpc: "2.0".into(),
        id: parsed.id,
        result: Some(json!({"content":[{"type":"text","text":"demo result"}]})),
        error: None,
    };
    let response_text = serde_json::to_string(&response)?;
    let decoded: McpResponse = serde_json::from_str(&response_text)?;
    assert!(decoded.result.is_some());
    let signature = Signature::parse("question, context -> answer")?;
    assert_eq!(signature.inputs, vec!["question", "context"]);
    assert_eq!(signature.outputs, vec!["answer"]);
    println!("parsed tool call: {call:?}; workflow: {state:?}");
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("run this example with the native feature enabled");
}
