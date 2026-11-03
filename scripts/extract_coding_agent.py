#!/usr/bin/env python3
"""Reproduce the three frozen requests; no agent or tool execution.

Requires PyYAML only when extracting. Download the manifest's source URLs
to --source-dir under their listed local_file names first.
"""
import argparse
import hashlib
import json
from pathlib import Path


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def messages(query):
    """SWE-agent v1.1.0 LiteLLMModel._history_to_messages, without logging.

    Copy only model-visible fields. In particular, do not use the current
    trajectory step's response, thought, action, observation or future history.
    """
    result = []
    for item in query:
        message = {"role": item["role"], "content": item["content"]}
        if item["role"] == "tool":
            assert len(item["tool_call_ids"]) == 1
            message["tool_call_id"] = item["tool_call_ids"][0]
        elif item.get("tool_calls") is not None:
            message["tool_calls"] = item["tool_calls"]
        if "cache_control" in item:
            message["cache_control"] = item["cache_control"]
        result.append(message)
    return result


def tool(name, config):
    """SWE-agent v1.1.0 Command.get_function_calling_tool's schema mapping."""
    properties, required = {}, []
    for arg in config.get("arguments", []):
        prop = {"type": arg["type"], "description": arg["description"]}
        for key in ("items", "enum"):
            if arg.get(key):
                prop[key] = arg[key]
        properties[arg["name"]] = prop
        if arg["required"]:
            required.append(arg["name"])
    return {"type": "function", "function": {
        "name": name, "description": config.get("docstring") or "",
        "parameters": {"type": "object", "properties": properties, "required": required},
    }}


def extract(source_dir, manifest, out):
    import yaml

    for source in manifest["sources"]:
        data = (source_dir / source["local_file"]).read_bytes()
        if sha256(data) != source["sha256"]:
            raise ValueError("source hash mismatch: " + source["local_file"])
    tools = [tool("bash", {
        "docstring": "runs the given command directly in bash",
        "arguments": [{"name": "command", "type": "string", "description": "The bash command to execute.", "required": True}],
    })]
    for name in ("tools_edit_anthropic_config.yaml", "tools_review_on_submit_m_config.yaml"):
        config = yaml.safe_load((source_dir / name).read_text())
        tools.extend(tool(k, v) for k, v in config["tools"].items())
    out.mkdir(parents=True, exist_ok=True)
    for fixture in manifest["fixtures"]:
        trajectory = json.loads((source_dir / (fixture["task_id"] + ".traj")).read_bytes())
        config = json.loads(trajectory["replay_config"])
        assert not config["agent"]["model"]["convert_system_to_user"]
        assert config["env"]["repo"]["base_commit"] == fixture["task_revision"]
        # The archived rendered tool docs independently confirm all names,
        # descriptions and arguments used to reconstruct the native schemas.
        docs = config["agent"]["tools"]["command_docs"]
        for entry in tools:
            fn = entry["function"]
            assert fn["name"] + ":" in docs and fn["description"] in docs
            for prop in fn["parameters"]["properties"].values():
                assert prop["description"] in docs
        query = trajectory["trajectory"][fixture["query_index"]]["query"]
        body = {"messages": messages(query), "tools": tools}
        data = encoded(body)
        if sha256(data) != fixture["sha256"]:
            raise ValueError("fixture hash mismatch: " + fixture["file"])
        (out / fixture["file"]).write_bytes(data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("crates/bench/fixtures/coding_agent/manifest.json"))
    args = parser.parse_args()
    extract(args.source_dir, json.loads(args.manifest.read_bytes()), args.out)
