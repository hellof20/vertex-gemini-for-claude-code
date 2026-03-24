import json
import uuid
import logging
from typing import Any

from google.genai import types

logger = logging.getLogger(__name__)

# tool_use_id -> {name, thought_signature}
_tool_use_store: dict[str, dict[str, Any]] = {}

GEMINI_FINISH_REASON_MAP = {
    "STOP": "end_turn",
    "MAX_TOKENS": "max_tokens",
    "SAFETY": "end_turn",
    "RECITATION": "end_turn",
    "FINISH_REASON_UNSPECIFIED": "end_turn",
}

UNSUPPORTED_SCHEMA_KEYS = {
    "$schema", "propertyNames", "const", "anyOf", "oneOf", "allOf",
    "not", "if", "then", "else", "patternProperties", "additionalItems",
    "dependencies", "contentMediaType", "contentEncoding",
}


# ────────────────────────────────────────────
# Schema cleaning
# ────────────────────────────────────────────


def clean_schema(obj: Any) -> Any:
    """Remove JSON Schema fields not supported by Gemini."""
    if isinstance(obj, dict):
        return {k: clean_schema(v) for k, v in obj.items() if k not in UNSUPPORTED_SCHEMA_KEYS}
    if isinstance(obj, list):
        return [clean_schema(item) for item in obj]
    return obj


# ────────────────────────────────────────────
# Request conversion: Anthropic -> Gemini SDK
# ────────────────────────────────────────────


def convert_content_to_parts(content: str | list) -> list[types.Part]:
    """Convert Anthropic content blocks to Gemini parts."""
    if isinstance(content, str):
        if not content.strip():
            return []
        return [types.Part(text=content)]

    parts = []
    for block in content:
        block_type = block.get("type")

        if block_type == "text":
            text = block.get("text", "").strip()
            if text:
                parts.append(types.Part(text=text))

        elif block_type == "image":
            source = block["source"]
            parts.append(types.Part(
                inline_data=types.Blob(
                    mime_type=source["media_type"],
                    data=source["data"],
                )
            ))

        elif block_type == "tool_use":
            tool_id = block["id"]
            tool_name = block["name"]
            stored = _tool_use_store.get(tool_id, {})

            part = types.Part(
                function_call=types.FunctionCall(
                    name=tool_name,
                    args=block.get("input", {}),
                )
            )
            if "thought_signature" in stored:
                part.thought_signature = stored["thought_signature"]

            _tool_use_store[tool_id] = {**stored, "name": tool_name}
            parts.append(part)

        elif block_type == "tool_result":
            tool_use_id = block.get("tool_use_id", "")
            stored = _tool_use_store.get(tool_use_id, {})
            tool_name = stored.get("name", "unknown")

            result_content = block.get("content", "")
            if isinstance(result_content, list):
                result_parts = []
                for sub in result_content:
                    if sub.get("type") == "text":
                        result_parts.append(sub["text"])
                result_text = " ".join(result_parts)
            else:
                result_text = str(result_content)

            response_data: dict[str, Any] = {"result": result_text}
            if block.get("is_error"):
                response_data["error"] = result_text

            parts.append(types.Part(
                function_response=types.FunctionResponse(
                    name=tool_name,
                    response=response_data,
                )
            ))

    return parts


def convert_messages(messages: list[dict]) -> list[types.Content]:
    """Convert Anthropic messages to Gemini contents, merging consecutive roles and ensuring it starts with user."""
    contents = []
    for msg in messages:
        # Anthropic roles: user, assistant
        # Gemini roles: user, model
        role = "model" if msg["role"] == "assistant" else "user"
        parts = convert_content_to_parts(msg["content"])
        if not parts:
            continue

        if contents and contents[-1].role == role:
            # Merge consecutive messages with same role
            contents[-1].parts.extend(parts)
        else:
            contents.append(types.Content(role=role, parts=parts))

    # Gemini MUST start with 'user' role. If it starts with 'model' (pre-fill),
    # we prepend an empty user message (though not ideal, it's a fix for 400).
    if contents and contents[0].role == "model":
        contents.insert(0, types.Content(role="user", parts=[types.Part(text="...")]))

    return contents


def convert_tools(tools: list[dict]) -> list[types.Tool]:
    """Convert Anthropic tool definitions to Gemini tools."""
    declarations = []
    for tool in tools:
        tool_type = tool.get("type")
        if tool_type and tool_type != "custom":
            continue

        decl: dict[str, Any] = {"name": tool["name"]}
        if "description" in tool:
            decl["description"] = tool["description"]
        if "input_schema" in tool:
            decl["parameters"] = clean_schema(tool["input_schema"])
        declarations.append(types.FunctionDeclaration(**decl))

    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]


def convert_tool_choice(tool_choice: dict) -> types.ToolConfig:
    """Convert Anthropic tool_choice to Gemini toolConfig."""
    choice_type = tool_choice.get("type", "auto")
    if choice_type == "auto":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))
    elif choice_type == "any":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="ANY"))
    elif choice_type == "none":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE"))
    elif choice_type == "tool":
        return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(
            mode="ANY",
            allowed_function_names=[tool_choice["name"]],
        ))
    return types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="AUTO"))


def build_gemini_request(body: dict) -> dict[str, Any]:
    """Convert Anthropic request body to kwargs for genai generate_content."""
    kwargs: dict[str, Any] = {}

    kwargs["contents"] = convert_messages(body["messages"])

    # generation config
    config_kwargs: dict[str, Any] = {}

    # system prompt
    system = body.get("system")
    if system:
        if isinstance(system, str):
            config_kwargs["system_instruction"] = system
        elif isinstance(system, list):
            text_parts = []
            for block in system:
                if isinstance(block, str):
                    text_parts.append(block)
                elif isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block["text"])
            config_kwargs["system_instruction"] = "\n".join(text_parts)
    if "max_tokens" in body:
        # Some Gemini models return 400 if max_output_tokens is > 8192 or similar.
        # But we'll try to pass what's requested unless it's obviously extreme
        # or it's known to fail. For now, let's just pass it.
        config_kwargs["max_output_tokens"] = body["max_tokens"]
    if "temperature" in body:
        config_kwargs["temperature"] = body["temperature"]
    if "top_p" in body:
        config_kwargs["top_p"] = body["top_p"]
    if "top_k" in body:
        config_kwargs["top_k"] = body["top_k"]
    if "stop_sequences" in body and body["stop_sequences"]:
        # Gemini does not like empty stop_sequences list
        config_kwargs["stop_sequences"] = body["stop_sequences"]

    # tools
    if "tools" in body:
        gemini_tools = convert_tools(body["tools"])
        if gemini_tools:
            config_kwargs["tools"] = gemini_tools

    # tool_choice
    if "tool_choice" in body:
        config_kwargs["tool_config"] = convert_tool_choice(body["tool_choice"])

    if config_kwargs:
        kwargs["config"] = types.GenerateContentConfig(**config_kwargs)

    return kwargs


# ────────────────────────────────────────────
# Response conversion: Gemini SDK -> Anthropic
# ────────────────────────────────────────────


def convert_parts_to_content(parts: list[types.Part]) -> list[dict]:
    """Convert Gemini response parts to Anthropic content blocks."""
    content = []
    for part in parts:
        if part.text is not None:
            content.append({"type": "text", "text": part.text})
        elif part.function_call is not None:
            fc = part.function_call
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            store_entry: dict[str, Any] = {"name": fc.name}

            if hasattr(part, "thought_signature") and part.thought_signature:
                store_entry["thought_signature"] = part.thought_signature
                logger.info("saved thought_signature for tool_id=%s name=%s", tool_id, fc.name)

            _tool_use_store[tool_id] = store_entry
            content.append({
                "type": "tool_use",
                "id": tool_id,
                "name": fc.name,
                "input": dict(fc.args) if fc.args else {},
            })
    return content


def get_stop_reason(finish_reason: str, content: list[dict]) -> str:
    has_tool_use = any(b.get("type") == "tool_use" for b in content)
    if has_tool_use:
        return "tool_use"
    return GEMINI_FINISH_REASON_MAP.get(finish_reason, "end_turn")


def build_anthropic_response(gemini_resp, model: str) -> dict:
    """Convert Gemini SDK response to Anthropic messages format."""
    candidate = gemini_resp.candidates[0] if gemini_resp.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else []

    content = convert_parts_to_content(parts)

    finish_reason = str(candidate.finish_reason) if candidate and candidate.finish_reason else "STOP"
    stop_reason = get_stop_reason(finish_reason, content)

    usage = gemini_resp.usage_metadata
    usage_dict = {
        "input_tokens": usage.prompt_token_count if usage else 0,
        "output_tokens": usage.candidates_token_count if usage else 0,
    }

    if usage:
        if usage.cached_content_token_count:
            usage_dict["cache_read_input_tokens"] = usage.cached_content_token_count
        if usage.thoughts_token_count:
            usage_dict["thinking_tokens"] = usage.thoughts_token_count

    return {
        "id": f"msg_{uuid.uuid4().hex[:24]}",
        "type": "message",
        "role": "assistant",
        "content": content,
        "model": model,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": usage_dict,
    }


# ────────────────────────────────────────────
# Streaming conversion
# ────────────────────────────────────────────


class StreamState:
    def __init__(self, msg_id: str, model: str):
        self.msg_id = msg_id
        self.model = model
        self.message_started = False
        self.text_block_started = False
        self.block_index = 0
        self.has_tool_use = False


def build_anthropic_stream_events(gemini_chunk, state: StreamState) -> list[dict]:
    """Convert a Gemini SDK streaming chunk to Anthropic SSE events."""
    events = []
    candidate = gemini_chunk.candidates[0] if gemini_chunk.candidates else None
    parts = candidate.content.parts if candidate and candidate.content else []
    finish_reason = str(candidate.finish_reason) if candidate and candidate.finish_reason else None
    usage = gemini_chunk.usage_metadata

    if not state.message_started:
        state.message_started = True
        events.append({
            "type": "message_start",
            "message": {
                "id": state.msg_id,
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": state.model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {
                    "input_tokens": usage.prompt_token_count if usage else 0,
                    "output_tokens": 0,
                },
            },
        })

    for part in parts:
        if part.text is not None and part.text:
            if not state.text_block_started:
                state.text_block_started = True
                events.append({
                    "type": "content_block_start",
                    "index": state.block_index,
                    "content_block": {"type": "text", "text": ""},
                })
            events.append({
                "type": "content_block_delta",
                "index": state.block_index,
                "delta": {"type": "text_delta", "text": part.text},
            })

        elif part.function_call is not None:
            fc = part.function_call
            tool_id = f"toolu_{uuid.uuid4().hex[:24]}"
            store_entry: dict[str, Any] = {"name": fc.name}
            if hasattr(part, "thought_signature") and part.thought_signature:
                store_entry["thought_signature"] = part.thought_signature
                logger.info("saved thought_signature (stream) for tool_id=%s", tool_id)
            _tool_use_store[tool_id] = store_entry
            state.has_tool_use = True

            if state.text_block_started or state.block_index > 0:
                events.append({"type": "content_block_stop", "index": state.block_index})
                state.block_index += 1
                state.text_block_started = False

            events.append({
                "type": "content_block_start",
                "index": state.block_index,
                "content_block": {
                    "type": "tool_use",
                    "id": tool_id,
                    "name": fc.name,
                },
            })
            events.append({
                "type": "content_block_delta",
                "index": state.block_index,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": json.dumps(dict(fc.args) if fc.args else {}),
                },
            })

    if finish_reason and finish_reason not in ("FINISH_REASON_UNSPECIFIED", "FinishReason.FINISH_REASON_UNSPECIFIED", ""):
        events.append({"type": "content_block_stop", "index": state.block_index})
        # normalize finish_reason enum string
        reason_str = finish_reason.replace("FinishReason.", "")
        stop_reason = "tool_use" if state.has_tool_use else GEMINI_FINISH_REASON_MAP.get(reason_str, "end_turn")
        events.append({
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": usage.candidates_token_count if usage else 0},
        })
        events.append({"type": "message_stop"})

    return events
