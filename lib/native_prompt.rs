//! Declarative prompts, constrained output, workflows, and tool protocols.

use crate::native::{NativeError, NativeResult};
use regex::Regex;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::collections::HashMap;

pub type Variables = HashMap<String, Value>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TemplatePart {
    Text(String),
    Variable(String),
    Generate {
        name: String,
        max_tokens: usize,
        stop: Option<String>,
        constraint: Option<ConstraintSpec>,
    },
    Select {
        name: String,
        choices: Vec<String>,
    },
}

/// A small Guidance/LMQL-style template language. Directives are enclosed in
/// `{{...}}`: variables use `{{name}}`, generation uses
/// `{{gen answer max_tokens=32 stop="\n"}}`, and deterministic selection uses
/// `{{select format choices="json|text"}}`.
#[derive(Debug, Clone)]
pub struct PromptTemplate {
    parts: Vec<TemplatePart>,
}

impl PromptTemplate {
    pub fn parse(source: &str) -> NativeResult<Self> {
        let mut parts = Vec::new();
        let mut rest = source;
        while let Some(start) = rest.find("{{") {
            if start > 0 {
                parts.push(TemplatePart::Text(rest[..start].into()));
            }
            let after = &rest[start + 2..];
            let end = after
                .find("}}")
                .ok_or_else(|| NativeError("unterminated prompt directive".into()))?;
            parts.push(parse_directive(after[..end].trim())?);
            rest = &after[end + 2..];
        }
        if !rest.is_empty() {
            parts.push(TemplatePart::Text(rest.into()));
        }
        Ok(Self { parts })
    }

    pub fn parts(&self) -> &[TemplatePart] {
        &self.parts
    }

    pub fn render_static(&self, variables: &Variables) -> NativeResult<String> {
        let mut output = String::new();
        for part in &self.parts {
            match part {
                TemplatePart::Text(text) => output.push_str(text),
                TemplatePart::Variable(name) => {
                    output.push_str(&value_string(required(variables, name)?))
                }
                TemplatePart::Select { name, choices } => {
                    let selected = value_string(required(variables, name)?);
                    if !choices.contains(&selected) {
                        return Err(NativeError(format!(
                            "{selected:?} is not allowed for {name}"
                        )));
                    }
                    output.push_str(&selected);
                }
                TemplatePart::Generate { name, .. } => {
                    output.push_str(&value_string(required(variables, name)?))
                }
            }
        }
        Ok(output)
    }

    /// Executes generation directives from left to right, binding each result
    /// for later template references (multi-part LMQL/Guidance semantics).
    pub fn execute<E: PromptExecutor>(
        &self,
        executor: &mut E,
        variables: &mut Variables,
    ) -> NativeResult<String> {
        let mut output = String::new();
        for part in &self.parts {
            match part {
                TemplatePart::Text(text) => output.push_str(text),
                TemplatePart::Variable(name) => {
                    output.push_str(&value_string(required(variables, name)?))
                }
                TemplatePart::Select { name, choices } => {
                    let selected = executor.select(&output, choices)?;
                    if !choices.contains(&selected) {
                        return Err(NativeError(
                            "executor returned a value outside select choices".into(),
                        ));
                    }
                    output.push_str(&selected);
                    variables.insert(name.clone(), Value::String(selected));
                }
                TemplatePart::Generate {
                    name,
                    max_tokens,
                    stop,
                    constraint,
                } => {
                    let compiled = constraint
                        .as_ref()
                        .map(ConstraintSpec::compile)
                        .transpose()?;
                    let generated = executor.generate(
                        &output,
                        *max_tokens,
                        stop.as_deref(),
                        compiled
                            .as_deref()
                            .map(|value| value as &dyn OutputConstraint),
                    )?;
                    output.push_str(&generated);
                    variables.insert(name.clone(), Value::String(generated));
                }
            }
        }
        Ok(output)
    }
}

pub trait PromptExecutor {
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: usize,
        stop: Option<&str>,
        constraint: Option<&dyn OutputConstraint>,
    ) -> NativeResult<String>;
    fn select(&mut self, prompt: &str, choices: &[String]) -> NativeResult<String>;
}

/// Tokenizers/backends use this interface while sampling: test each decoded
/// candidate prefix before accepting its token, and finish only on a complete
/// value. This is the integration point for GBNF/regex/schema logit masks.
pub trait OutputConstraint {
    fn allows_prefix(&self, text: &str) -> bool;
    fn is_complete(&self, text: &str) -> bool;
    fn description(&self) -> &str;
}

/// Serializable guided-decoding configuration accepted by [`NativeEngine`](crate::native::NativeEngine).
/// Unlike a boxed callback, this can be transported in an HTTP/JSON request.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum ConstraintSpec {
    Choice { choices: Vec<String> },
    Regex { pattern: String },
    Json,
    Gbnf { grammar: String, root: String },
}

impl ConstraintSpec {
    pub fn compile(&self) -> NativeResult<Box<dyn OutputConstraint + Send + Sync>> {
        match self {
            Self::Choice { choices } => Ok(Box::new(ChoiceConstraint::new(choices.clone())?)),
            Self::Regex { pattern } => Ok(Box::new(RegexConstraint::new(pattern)?)),
            Self::Json => Ok(Box::new(JsonConstraint)),
            Self::Gbnf { grammar, root } => Ok(Box::new(GbnfGrammar::parse(grammar, root)?)),
        }
    }
}

pub struct ChoiceConstraint {
    choices: Vec<String>,
}
impl ChoiceConstraint {
    pub fn new(choices: Vec<String>) -> NativeResult<Self> {
        if choices.is_empty() || choices.iter().any(String::is_empty) {
            return Err(NativeError("choices cannot be empty".into()));
        }
        Ok(Self { choices })
    }
}
impl OutputConstraint for ChoiceConstraint {
    fn allows_prefix(&self, text: &str) -> bool {
        self.choices.iter().any(|choice| choice.starts_with(text))
    }
    fn is_complete(&self, text: &str) -> bool {
        self.choices.iter().any(|choice| choice == text)
    }
    fn description(&self) -> &str {
        "choice"
    }
}

pub struct RegexConstraint {
    regex: Regex,
    source: String,
}
impl RegexConstraint {
    pub fn new(pattern: &str) -> NativeResult<Self> {
        let regex = Regex::new(&format!("^(?:{pattern})$"))
            .map_err(|e| NativeError(format!("invalid regex: {e}")))?;
        Ok(Self {
            regex,
            source: pattern.into(),
        })
    }
}
impl OutputConstraint for RegexConstraint {
    // Rust regexes do not expose partial DFA matches. Prefixes remain eligible;
    // completion is nevertheless strict and callers can backtrack/retry.
    fn allows_prefix(&self, _: &str) -> bool {
        true
    }
    fn is_complete(&self, text: &str) -> bool {
        self.regex.is_match(text)
    }
    fn description(&self) -> &str {
        &self.source
    }
}

pub struct JsonConstraint;
impl OutputConstraint for JsonConstraint {
    fn allows_prefix(&self, text: &str) -> bool {
        json_prefix_valid(text)
    }
    fn is_complete(&self, text: &str) -> bool {
        serde_json::from_str::<Value>(text).is_ok()
    }
    fn description(&self) -> &str {
        "JSON"
    }
}

/// Compact GBNF subset supporting named rules, quoted literals, references,
/// sequences, and `|` alternatives. It is sufficient for command/JSON-like
/// fixed schemas and rejects unsupported EBNF operators rather than weakening
/// the constraint silently.
#[derive(Debug, Clone)]
pub struct GbnfGrammar {
    alternatives: Vec<String>,
    source: String,
}
impl GbnfGrammar {
    pub fn parse(source: &str, root: &str) -> NativeResult<Self> {
        let mut rules = HashMap::<String, Vec<Vec<String>>>::new();
        for line in source
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
        {
            let (name, expression) = line
                .split_once("::=")
                .ok_or_else(|| NativeError("GBNF rule requires ::=".into()))?;
            let branches = split_unquoted(expression, '|')
                .into_iter()
                .map(tokenize_gbnf)
                .collect::<NativeResult<Vec<_>>>()?;
            rules.insert(name.trim().into(), branches);
        }
        let mut alternatives = Vec::new();
        expand_rule(
            root,
            &rules,
            &mut Vec::new(),
            String::new(),
            &mut alternatives,
            4096,
        )?;
        if alternatives.is_empty() {
            return Err(NativeError("GBNF root has no expansions".into()));
        }
        Ok(Self {
            alternatives,
            source: source.into(),
        })
    }
}
impl OutputConstraint for GbnfGrammar {
    fn allows_prefix(&self, text: &str) -> bool {
        self.alternatives
            .iter()
            .any(|value| value.starts_with(text))
    }
    fn is_complete(&self, text: &str) -> bool {
        self.alternatives.iter().any(|value| value == text)
    }
    fn description(&self) -> &str {
        &self.source
    }
}

#[derive(Debug, Clone)]
pub enum WorkflowStep {
    Prompt {
        template: PromptTemplate,
    },
    Transform {
        input: String,
        output: String,
        operation: Transform,
    },
    Fork {
        branches: Vec<Workflow>,
    },
}
#[derive(Debug, Clone)]
pub enum Transform {
    JsonPointer(String),
    Uppercase,
    Lowercase,
}
#[derive(Debug, Clone, Default)]
pub struct Workflow {
    pub steps: Vec<WorkflowStep>,
}

impl Workflow {
    /// Executes LCEL/SGLang-style prompt pipelines. Forked branches receive a
    /// snapshot of state and are joined into `branch_0`, `branch_1`, ... maps.
    pub fn execute<E: PromptExecutor>(
        &self,
        executor: &mut E,
        state: &mut Variables,
    ) -> NativeResult<()> {
        for step in &self.steps {
            match step {
                WorkflowStep::Prompt { template } => {
                    let text = template.execute(executor, state)?;
                    state.insert("output".into(), Value::String(text));
                }
                WorkflowStep::Transform {
                    input,
                    output,
                    operation,
                } => {
                    let value = required(state, input)?.clone();
                    state.insert(output.clone(), transform(value, operation)?);
                }
                WorkflowStep::Fork { branches } => {
                    for (index, branch) in branches.iter().enumerate() {
                        let mut child = state.clone();
                        branch.execute(executor, &mut child)?;
                        state.insert(
                            format!("branch_{index}"),
                            serde_json::to_value(child).map_err(|e| NativeError(e.to_string()))?,
                        );
                    }
                }
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct ToolCall {
    pub name: String,
    pub arguments: Value,
}
impl ToolCall {
    pub fn from_openai_json(text: &str) -> NativeResult<Self> {
        serde_json::from_str(text).map_err(|e| NativeError(format!("invalid tool call: {e}")))
    }
    pub fn from_react(text: &str) -> NativeResult<Self> {
        let action = text
            .lines()
            .find_map(|line| line.strip_prefix("Action:"))
            .ok_or_else(|| NativeError("ReAct output has no Action".into()))?
            .trim();
        let arguments = text
            .lines()
            .find_map(|line| line.strip_prefix("Action Input:"))
            .ok_or_else(|| NativeError("ReAct output has no Action Input".into()))?
            .trim();
        let arguments =
            serde_json::from_str(arguments).unwrap_or_else(|_| Value::String(arguments.into()));
        Ok(Self {
            name: action.into(),
            arguments,
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Value,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Value,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<Value>,
}

#[derive(Debug, Clone)]
pub struct Signature {
    pub inputs: Vec<String>,
    pub outputs: Vec<String>,
    pub instructions: String,
}
impl Signature {
    pub fn parse(value: &str) -> NativeResult<Self> {
        let (inputs, outputs) = value
            .split_once("->")
            .ok_or_else(|| NativeError("signature requires ->".into()))?;
        let parse = |side: &str| {
            side.split(',')
                .map(str::trim)
                .filter(|v| !v.is_empty())
                .map(str::to_owned)
                .collect::<Vec<_>>()
        };
        let inputs = parse(inputs);
        let outputs = parse(outputs);
        if inputs.is_empty() || outputs.is_empty() {
            return Err(NativeError(
                "signature inputs and outputs cannot be empty".into(),
            ));
        }
        Ok(Self {
            inputs,
            outputs,
            instructions: String::new(),
        })
    }
}

fn parse_directive(value: &str) -> NativeResult<TemplatePart> {
    let words = shell_words(value)?;
    match words.first().map(String::as_str) {
        Some("gen") => {
            let name = words
                .get(1)
                .ok_or_else(|| NativeError("gen requires a binding name".into()))?
                .clone();
            let mut max_tokens = 64;
            let mut stop = None;
            let mut constraint = None;
            for option in &words[2..] {
                if let Some(value) = option.strip_prefix("max_tokens=") {
                    max_tokens = value
                        .parse()
                        .map_err(|_| NativeError("invalid max_tokens".into()))?;
                } else if let Some(value) = option.strip_prefix("stop=") {
                    stop = Some(value.into());
                } else if let Some(value) = option.strip_prefix("regex=") {
                    set_constraint(
                        &mut constraint,
                        ConstraintSpec::Regex {
                            pattern: value.into(),
                        },
                    )?;
                } else if let Some(value) = option.strip_prefix("choices=") {
                    set_constraint(
                        &mut constraint,
                        ConstraintSpec::Choice {
                            choices: value.split('|').map(str::to_owned).collect(),
                        },
                    )?;
                } else if option == "json" {
                    set_constraint(&mut constraint, ConstraintSpec::Json)?;
                } else {
                    return Err(NativeError(format!("unknown gen option {option}")));
                }
            }
            Ok(TemplatePart::Generate {
                name,
                max_tokens,
                stop,
                constraint,
            })
        }
        Some("select") => {
            let name = words
                .get(1)
                .ok_or_else(|| NativeError("select requires a binding name".into()))?
                .clone();
            let choices = words
                .iter()
                .find_map(|word| word.strip_prefix("choices="))
                .ok_or_else(|| NativeError("select requires choices".into()))?
                .split('|')
                .map(str::to_owned)
                .collect();
            Ok(TemplatePart::Select { name, choices })
        }
        Some(_) if words.len() == 1 => Ok(TemplatePart::Variable(words[0].clone())),
        _ => Err(NativeError("invalid prompt directive".into())),
    }
}

fn set_constraint(slot: &mut Option<ConstraintSpec>, value: ConstraintSpec) -> NativeResult<()> {
    if slot.replace(value).is_some() {
        return Err(NativeError(
            "gen accepts only one of regex, choices, or json".into(),
        ));
    }
    Ok(())
}

fn shell_words(value: &str) -> NativeResult<Vec<String>> {
    let mut result = Vec::new();
    let mut current = String::new();
    let mut quote = None;
    for character in value.chars() {
        match (quote, character) {
            (Some(active), c) if c == active => quote = None,
            (None, '"' | '\'') => quote = Some(character),
            (None, c) if c.is_whitespace() => {
                if !current.is_empty() {
                    result.push(std::mem::take(&mut current));
                }
            }
            (_, c) => current.push(c),
        }
    }
    if quote.is_some() {
        return Err(NativeError("unterminated quoted directive value".into()));
    }
    if !current.is_empty() {
        result.push(current);
    }
    Ok(result)
}
fn required<'a>(values: &'a Variables, name: &str) -> NativeResult<&'a Value> {
    values
        .get(name)
        .ok_or_else(|| NativeError(format!("missing variable {name}")))
}
fn value_string(value: &Value) -> String {
    value
        .as_str()
        .map(str::to_owned)
        .unwrap_or_else(|| value.to_string())
}
fn transform(value: Value, operation: &Transform) -> NativeResult<Value> {
    match operation {
        Transform::JsonPointer(pointer) => value
            .pointer(pointer)
            .cloned()
            .ok_or_else(|| NativeError(format!("JSON pointer {pointer} not found"))),
        Transform::Uppercase => Ok(Value::String(value_string(&value).to_uppercase())),
        Transform::Lowercase => Ok(Value::String(value_string(&value).to_lowercase())),
    }
}

fn json_prefix_valid(text: &str) -> bool {
    match serde_json::from_str::<Value>(text) {
        Ok(_) => true,
        Err(error) => error.classify() == serde_json::error::Category::Eof,
    }
}
fn split_unquoted(value: &str, separator: char) -> Vec<&str> {
    let mut result = Vec::new();
    let mut quote = false;
    let mut start = 0;
    for (index, c) in value.char_indices() {
        if c == '"' {
            quote = !quote;
        } else if c == separator && !quote {
            result.push(value[start..index].trim());
            start = index + c.len_utf8();
        }
    }
    result.push(value[start..].trim());
    result
}
fn tokenize_gbnf(value: &str) -> NativeResult<Vec<String>> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    let mut quoted = false;
    let mut escaped = false;
    for character in value.chars() {
        if quoted {
            current.push(character);
            if escaped {
                escaped = false;
            } else if character == '\\' {
                escaped = true;
            } else if character == '"' {
                quoted = false;
            }
        } else if character == '"' {
            quoted = true;
            current.push(character);
        } else if character.is_whitespace() {
            if !current.is_empty() {
                tokens.push(std::mem::take(&mut current));
            }
        } else {
            current.push(character);
        }
    }
    if quoted {
        return Err(NativeError("unterminated GBNF literal".into()));
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    Ok(tokens)
}
fn expand_rule(
    name: &str,
    rules: &HashMap<String, Vec<Vec<String>>>,
    stack: &mut Vec<String>,
    prefix: String,
    output: &mut Vec<String>,
    limit: usize,
) -> NativeResult<()> {
    if stack.iter().any(|active| active == name) {
        return Err(NativeError(
            "recursive GBNF requires an external grammar backend".into(),
        ));
    }
    let branches = rules
        .get(name)
        .ok_or_else(|| NativeError(format!("unknown GBNF rule {name}")))?;
    stack.push(name.into());
    for branch in branches {
        expand_tokens(branch, 0, rules, stack, prefix.clone(), output, limit)?;
    }
    stack.pop();
    Ok(())
}
fn expand_tokens(
    tokens: &[String],
    index: usize,
    rules: &HashMap<String, Vec<Vec<String>>>,
    stack: &mut Vec<String>,
    prefix: String,
    output: &mut Vec<String>,
    limit: usize,
) -> NativeResult<()> {
    if output.len() >= limit {
        return Err(NativeError("GBNF expansion limit exceeded".into()));
    }
    if index == tokens.len() {
        output.push(prefix);
        return Ok(());
    }
    let token = &tokens[index];
    if token.starts_with('"') && token.ends_with('"') {
        let literal: String = serde_json::from_str(token)
            .map_err(|e| NativeError(format!("invalid GBNF literal: {e}")))?;
        expand_tokens(
            tokens,
            index + 1,
            rules,
            stack,
            prefix + &literal,
            output,
            limit,
        )
    } else {
        if stack.iter().any(|active| active == token) {
            return Err(NativeError(
                "recursive GBNF requires an external grammar backend".into(),
            ));
        }
        let branches = rules
            .get(token)
            .ok_or_else(|| NativeError(format!("unknown GBNF symbol {token}")))?;
        stack.push(token.clone());
        for branch in branches {
            let mut intermediate = Vec::new();
            expand_tokens(
                branch,
                0,
                rules,
                stack,
                String::new(),
                &mut intermediate,
                limit,
            )?;
            for expansion in intermediate {
                expand_tokens(
                    tokens,
                    index + 1,
                    rules,
                    stack,
                    prefix.clone() + &expansion,
                    output,
                    limit,
                )?;
            }
        }
        stack.pop();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn template_binds_variables_and_choices() {
        let template =
            PromptTemplate::parse("Hello {{name}}: {{select kind choices=short|long}}").unwrap();
        let values = Variables::from([
            (String::from("name"), Value::String("Ada".into())),
            (String::from("kind"), Value::String("short".into())),
        ]);
        assert_eq!(template.render_static(&values).unwrap(), "Hello Ada: short");
    }
    #[test]
    fn gbnf_masks_invalid_prefixes() {
        let grammar = GbnfGrammar::parse(
            "root ::= greeting \"!\"\ngreeting ::= \"hello\" | \"hi\"",
            "root",
        )
        .unwrap();
        assert!(grammar.allows_prefix("hel"));
        assert!(grammar.is_complete("hello!"));
        assert!(!grammar.allows_prefix("bye"));
    }
    #[test]
    fn parses_tool_protocols() {
        assert_eq!(
            ToolCall::from_react("Thought: x\nAction: search\nAction Input: {\"q\":\"rust\"}")
                .unwrap()
                .name,
            "search"
        );
        assert!(JsonConstraint.allows_prefix("{\"a\":[1,"));
        assert!(!JsonConstraint.allows_prefix("{]"));
    }
}
