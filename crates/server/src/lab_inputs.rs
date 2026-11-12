//! Offline benchmark preparation through the serving template seam.
use serde_json::{json, Value};
use crate::template::{ChatMessage, TemplateProvider};
use crate::thinking::{self, ThinkingDefaults, ThinkingRequestFields};

pub fn prepare(provider: &dyn TemplateProvider, request: &Value) -> Result<Value, String> {
    let messages: Vec<ChatMessage> = serde_json::from_value(request["messages"].clone())
        .map_err(|e| format!("messages: {e}"))?;
    let mut messages = crate::instruction::InstructionPolicy::default().normalize(&messages)
        .map_err(|e|e.message)?;
    crate::loop_guard::repair_history(&mut messages);
    let tools = request.get("tools").map(|v| serde_json::from_value::<Vec<Value>>(v.clone()))
        .transpose().map_err(|e| format!("tools: {e}"))?;
    let (tools, _) = crate::api::resolve_tools(tools, request.get("tool_choice").cloned())
        .map_err(|_| "tool configuration rejected".to_string())?;
    let options = thinking::resolve(ThinkingRequestFields {
        enable_thinking: request.get("enable_thinking"),
        reasoning_effort: request.get("reasoning_effort"),
        preserve_thinking: request.get("preserve_thinking"),
        chat_template_kwargs: request.get("chat_template_kwargs"),
    }, &ThinkingDefaults::default(), &provider.thinking_capabilities()).map_err(|e|e.message().to_string())?;
    let rendered = provider.apply_chat_template(&messages, &options, &tools).map_err(|e|e.message)?;
    if rendered.tokens.is_empty() { return Err("empty prepared prompt".into()); }
    let marker = |text: &str| -> Result<u32, String> {
        let ids = provider.encode_literal(text).ok_or_else(|| format!("cannot encode {text}"))?;
        if ids.len() != 1 { return Err(format!("{text} is not one token")); }
        Ok(ids[0])
    };
    Ok(json!({"tokens":rendered.tokens, "input_tokens":rendered.tokens.len(),
        "thinking_open":provider.decoder_starts_in_reasoning(&options),
        "think_open":marker("<think>")?, "think_close":marker("</think>")?,
        "output_tokens":request["max_tokens"], "request":request}))
}

#[cfg(test)]
mod tests {
    use super::*;
    struct Probe;
    impl TemplateProvider for Probe {
        fn apply_chat_template(&self, messages:&[ChatMessage], options:&thinking::ThinkingOptions,
            tools:&[Value])->Result<crate::template::RenderedPrompt,crate::template::TemplateRejection> {
            assert!(!messages.is_empty());
            assert!(options.enable_thinking && options.preserve_thinking);
            assert_eq!(options.reasoning_effort.map(|v|v.as_str()),Some("max"));
            assert!(!tools.is_empty());
            for tool in tools { assert_eq!(tool["function"]["strict"],false); }
            Ok(vec![42,43].into())
        }
        fn encode_literal(&self,text:&str)->Option<Vec<ignis_core::TokenId>> {
            match text {"<think>"=>Some(vec![91]),"</think>"=>Some(vec![92]),_=>None}
        }
        fn render_tokens(&self,tokens:&[ignis_core::TokenId])->String {
            crate::template::SimpleTemplateProvider.render_tokens(tokens)
        }
        fn thinking_capabilities(&self)->thinking::ThinkingCapabilities {
            crate::template::SimpleTemplateProvider.thinking_capabilities()
        }
        fn token_decoder(&self)->Box<dyn crate::decoder::TokenDecoder> {
            crate::template::SimpleTemplateProvider.token_decoder()
        }
    }
    #[test]
    fn benchmark_tools_get_the_same_defaults_as_http_before_rendering() {
        for source in ignis_bench::coding_agent::FIXTURES {
            let body=ignis_bench::coding_agent::request(serde_json::from_str(source).unwrap(),false);
            let result=prepare(&Probe,&body).unwrap();
            assert_eq!(result["tokens"],json!([42,43]));
            assert_eq!(result["output_tokens"],8192);
            assert_eq!(result["think_close"],92);
        }
    }
    #[test]
    fn invalid_messages_are_not_silently_prepared() {
        let p = crate::template::SimpleTemplateProvider;
        assert!(prepare(&p, &json!({"messages":"bad"})).is_err());
    }
}
