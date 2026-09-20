//! Declarative prompting, grammar constraints, workflows, and tool protocols.

#[cfg(feature = "native")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use candlelighter::native::{NativeError, NativeResult};
    use candlelighter::native_prompt::*;
    use serde_json::json;

    struct DemoExecutor;
    impl PromptExecutor for DemoExecutor {
        fn generate(
            &mut self,
            _: &str,
            _: usize,
            _: Option<&str>,
            constraint: Option<&dyn OutputConstraint>,
        ) -> NativeResult<String> {
            let output = "hello".to_string();
            if constraint.is_some_and(|constraint| !constraint.is_complete(&output)) {
                return Err(NativeError("demo output violates constraint".into()));
            }
            Ok(output)
        }
        fn select(&mut self, _: &str, choices: &[String]) -> NativeResult<String> {
            choices
                .first()
                .cloned()
                .ok_or_else(|| NativeError("no choices".into()))
        }
    }

    let template = PromptTemplate::parse(
        "Question: {{question}}\nFormat: {{select format choices=json|text}}\nAnswer: {{gen answer max_tokens=32 regex=hello stop=\"\\n\"}}",
    )?;
    let mut variables = Variables::from([("question".into(), json!("Say hello"))]);
    let output = template.execute(&mut DemoExecutor, &mut variables)?;
    println!("{output}");

    let grammar = GbnfGrammar::parse("root ::= command\ncommand ::= \"start\" | \"stop\"", "root")?;
    assert!(grammar.allows_prefix("sta"));
    assert!(grammar.is_complete("start"));

    let call = ToolCall::from_react(
        "Thought: look it up\nAction: search\nAction Input: {\"query\":\"Rust\"}",
    )?;
    println!("tool call: {call:?}");

    let request = McpRequest {
        jsonrpc: "2.0".into(),
        id: json!(1),
        method: "tools/call".into(),
        params: json!({"name": call.name, "arguments": call.arguments}),
    };
    println!("MCP: {}", serde_json::to_string(&request)?);
    println!(
        "DSPy-style signature: {:?}",
        Signature::parse("question, context -> answer")?
    );
    Ok(())
}

#[cfg(not(feature = "native"))]
fn main() {
    eprintln!("enable this example with --no-default-features --features native");
}
