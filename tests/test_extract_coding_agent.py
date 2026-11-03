import importlib.util
from pathlib import Path
import unittest


spec = importlib.util.spec_from_file_location("extract", Path(__file__).parents[1] / "scripts/extract_coding_agent.py")
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


class ExtractTests(unittest.TestCase):
    def test_projection_preserves_visible_fields_and_excludes_agent_annotations(self):
        query = [
            {"role": "system", "content": "system", "agent": "main"},
            {"role": "assistant", "content": "analysis", "tool_calls": [{"id": "call"}], "thought": "not a wire field"},
            {"role": "tool", "content": [{"type": "text", "text": "source\r\n"}], "tool_call_ids": ["call"], "cache_control": {"type": "ephemeral"}},
        ]
        messages = extract.messages(query)
        self.assertEqual(messages[0], {"role": "system", "content": "system"})
        self.assertNotIn("thought", messages[1])
        self.assertEqual(messages[1]["tool_calls"], query[1]["tool_calls"])
        self.assertEqual(messages[2]["content"], query[2]["content"])
        self.assertEqual(messages[2]["tool_call_id"], "call")
        self.assertEqual(messages[2]["cache_control"], query[2]["cache_control"])

    def test_tool_schema_keeps_required_enum_and_array_item_types(self):
        tool = extract.tool("edit", {"docstring": "description\n", "arguments": [
            {"name": "command", "type": "string", "description": "mode", "required": True, "enum": ["view"]},
            {"name": "range", "type": "array", "description": "lines", "required": False, "items": {"type": "integer"}},
        ]})["function"]
        self.assertEqual(tool["description"], "description\n")
        self.assertEqual(tool["parameters"]["required"], ["command"])
        self.assertEqual(tool["parameters"]["properties"]["command"]["enum"], ["view"])
        self.assertEqual(tool["parameters"]["properties"]["range"]["items"], {"type": "integer"})


if __name__ == "__main__":
    unittest.main()
