//! Instruction-pair graph SFT data, answer-only labels and native LM-head checkpoints.
use crate::graph2text::{Graph, Triple};
use crate::native::{NativeError, NativeResult};
use crate::native_training::{LoraAdapter, LoraConfig};
use rand::seq::SliceRandom;
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct InstructionExample {
    pub instruction: String,
    pub input: String,
    pub output: String,
}
impl InstructionExample {
    pub fn triples(&self) -> NativeResult<Vec<Triple>> {
        if self.instruction.trim().is_empty() || self.output.trim().is_empty() {
            return Err(NativeError(
                "instruction and output must be nonblank".into(),
            ));
        }
        let mut triples = Vec::new();
        for item in self.input.split('|') {
            let item = item
                .trim()
                .strip_prefix("<H>")
                .ok_or_else(|| NativeError("each triple must start with <H>".into()))?;
            let (subject, rest) = item
                .split_once("<R>")
                .ok_or_else(|| NativeError("missing <R> delimiter".into()))?;
            let (predicate, object) = rest
                .split_once("<T>")
                .ok_or_else(|| NativeError("missing <T> delimiter".into()))?;
            let fields = [subject.trim(), predicate.trim(), object.trim()];
            if fields.iter().any(|s| {
                s.is_empty() || ["<H>", "<R>", "<T>"].iter().any(|token| s.contains(token))
            }) {
                return Err(NativeError(
                    "empty triple field or ambiguous reserved delimiter".into(),
                ));
            }
            triples.push(Triple {
                subject: fields[0].into(),
                predicate: fields[1].into(),
                object: fields[2].into(),
            });
        }
        Graph::from_triples(&triples).map_err(|e| NativeError(e.to_string()))?;
        Ok(triples)
    }
    /// Consistent prompt format shared by training and inference. Reserved markers
    /// are ordinary tokenizer text; no vocabulary resizing is required.
    pub fn prompt<R: rand::Rng + ?Sized>(
        &self,
        shuffle: bool,
        rng: &mut R,
    ) -> NativeResult<String> {
        let mut triples = self.triples()?;
        if shuffle {
            triples.shuffle(rng);
        }
        let input = triples
            .iter()
            .map(|t| format!("<H> {} <R> {} <T> {}", t.subject, t.predicate, t.object))
            .collect::<Vec<_>>()
            .join(" | ");
        Ok(format!(
            "You are an expert Graph-to-Text verbalizer.\n{}\n\nGraph Input:\n{}\n\nAnswer:\n",
            self.instruction.trim(),
            input
        ))
    }
}

#[derive(Debug)]
pub struct SupervisedTokens {
    pub input_ids: Vec<u32>,
    /// Already shifted next-token targets. Prompt targets are -100.
    pub labels: Vec<i64>,
    pub prompt_tokens: usize,
}
/// Tokenize prompt and completion separately so prompt prefixes match inference.
/// Reject overflow instead of silently deleting evidence or answer tokens.
pub fn answer_only_tokens(
    tokenizer: &Tokenizer,
    prompt: &str,
    answer: &str,
    eos: Option<u32>,
    context_limit: usize,
) -> NativeResult<SupervisedTokens> {
    if prompt.trim().is_empty() || answer.trim().is_empty() {
        return Err(NativeError("prompt and answer must be nonblank".into()));
    }
    let mut ids = tokenizer
        .encode(prompt, true)
        .map_err(|e| NativeError(e.to_string()))?
        .get_ids()
        .to_vec();
    let boundary = ids.len();
    if boundary == 0 {
        return Err(NativeError("prompt tokenized to no tokens".into()));
    }
    let target = tokenizer
        .encode(answer, false)
        .map_err(|e| NativeError(e.to_string()))?;
    if target.is_empty() {
        return Err(NativeError("answer tokenized to no tokens".into()));
    }
    ids.extend_from_slice(target.get_ids());
    if let Some(eos) = eos {
        ids.push(eos);
    }
    if ids.len() > context_limit {
        return Err(NativeError(format!(
            "{} tokens exceed context limit {context_limit}",
            ids.len()
        )));
    }
    let labels = ids
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, id)| {
            if index < boundary {
                -100
            } else {
                i64::from(*id)
            }
        })
        .collect();
    ids.pop();
    Ok(SupervisedTokens {
        input_ids: ids,
        labels,
        prompt_tokens: boundary,
    })
}

#[derive(Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HeadAdapterCheckpoint {
    pub format_version: u32,
    /// Informational identity; callers must retain the exact original base weights/tokenizer.
    pub base_model: String,
    pub input: usize,
    pub output: usize,
    pub config: LoraConfig,
    pub a: Vec<f32>,
    pub b: Vec<f32>,
}
impl HeadAdapterCheckpoint {
    pub fn from_adapter(base_model: String, adapter: &LoraAdapter) -> Self {
        let (input, output) = adapter.dimensions();
        let (a, b) = adapter.weights();
        Self {
            format_version: 1,
            base_model,
            input,
            output,
            config: adapter.config().clone(),
            a: a.to_vec(),
            b: b.to_vec(),
        }
    }
    pub fn into_adapter(self) -> NativeResult<LoraAdapter> {
        if self.format_version != 1
            || self.input == 0
            || self.output == 0
            || self.a.iter().chain(&self.b).any(|x| !x.is_finite())
        {
            return Err(NativeError("invalid output-head adapter checkpoint".into()));
        }
        if self.config.rank.checked_mul(self.input) != Some(self.a.len())
            || self.output.checked_mul(self.config.rank) != Some(self.b.len())
        {
            return Err(NativeError(
                "adapter checkpoint dimensions are inconsistent".into(),
            ));
        }
        LoraAdapter::from_weights(self.input, self.output, self.config, self.a, self.b)
    }
}

/// Entity mention recall is a lexical diagnostic, not a relation-fidelity metric.
pub fn entity_mention_recall(triples: &[Triple], text: &str) -> f32 {
    let entities: std::collections::BTreeSet<_> = triples
        .iter()
        .flat_map(|t| [t.subject.to_lowercase(), t.object.to_lowercase()])
        .collect();
    if entities.is_empty() {
        return 0.0;
    }
    let text = text.to_lowercase();
    entities
        .iter()
        .filter(|entity| text.contains(entity.as_str()))
        .count() as f32
        / entities.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{rngs::StdRng, SeedableRng};
    fn row() -> InstructionExample {
        InstructionExample {
            instruction: "Describe only these facts.".into(),
            input: "<H> Ada <R> wrote <T> notes | <H> Babbage <R> designed <T> Engine".into(),
            output: "Ada wrote notes.".into(),
        }
    }
    fn tokenizer() -> Tokenizer {
        Tokenizer::from_bytes(br#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"<unk>":0,"hello":1,"answer":2,"<eos>":3},"unk_token":"<unk>"}}"#).unwrap()
    }
    #[test]
    fn parser_rejects_ambiguous_or_incomplete_triples() {
        assert_eq!(row().triples().unwrap().len(), 2);
        for input in [
            "Ada wrote notes",
            "<H> Ada <R> wrote",
            "<H> <R> wrote <T> notes",
            "<H> Ada <R> wrote <T> notes <T> duplicate",
        ] {
            let mut data = row();
            data.input = input.into();
            assert!(data.triples().is_err());
        }
    }
    #[test]
    fn augmentation_preserves_facts_and_is_seeded() {
        let a = row().prompt(true, &mut StdRng::seed_from_u64(42)).unwrap();
        let b = row().prompt(true, &mut StdRng::seed_from_u64(42)).unwrap();
        assert_eq!(a, b);
        assert!(a.contains("<H> Ada <R> wrote <T> notes"));
        assert!(a.contains("<H> Babbage <R> designed <T> Engine"));
    }
    #[test]
    fn shifted_labels_mask_prompt_but_keep_first_answer_and_eos() {
        let tokens = answer_only_tokens(&tokenizer(), "hello hello", "answer", Some(3), 4).unwrap();
        assert_eq!(tokens.input_ids, vec![1, 1, 2]);
        assert_eq!(tokens.labels, vec![-100, 2, 3]);
        assert_eq!(tokens.prompt_tokens, 2);
        assert!(answer_only_tokens(&tokenizer(), "hello hello", "answer", Some(3), 3).is_err());
        assert!(answer_only_tokens(&tokenizer(), "", "answer", Some(3), 4).is_err());
    }
    #[test]
    fn checkpoint_roundtrip_and_validation() {
        let original = LoraAdapter::new(
            2,
            4,
            LoraConfig {
                rank: 2,
                alpha: 4.0,
                dropout: 0.0,
            },
            42,
        )
        .unwrap();
        let saved = HeadAdapterCheckpoint::from_adapter("base".into(), &original);
        let restored: HeadAdapterCheckpoint =
            serde_json::from_str(&serde_json::to_string(&saved).unwrap()).unwrap();
        assert_eq!(
            restored.into_adapter().unwrap().weights(),
            original.weights()
        );
        let mut invalid = HeadAdapterCheckpoint::from_adapter("base".into(), &original);
        invalid.a[0] = f32::NAN;
        assert!(invalid.into_adapter().is_err());
        let mut invalid = HeadAdapterCheckpoint::from_adapter("base".into(), &original);
        invalid.config.rank = usize::MAX;
        assert!(invalid.into_adapter().is_err());
        assert!(LoraConfig {
            rank: 1,
            alpha: f32::NAN,
            dropout: 0.0
        }
        .validate()
        .is_err());
    }
    #[test]
    fn entity_mentions_are_diagnostic_only() {
        let triples = row().triples().unwrap();
        assert_eq!(
            entity_mention_recall(&triples, "Ada wrote notes. Babbage designed the Engine."),
            1.0
        );
        assert_eq!(entity_mention_recall(&triples, "No matching entity"), 0.0);
        // Reversing relations still mentions entities; this cannot establish grounding.
        assert_eq!(
            entity_mention_recall(&triples, "notes wrote Ada. Engine designed Babbage."),
            1.0
        );
    }
}
