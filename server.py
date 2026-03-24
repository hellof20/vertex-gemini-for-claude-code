"""
Anthropic -> Vertex AI Gemini proxy server.
Accepts Anthropic format requests, converts to Gemini format,
forwards to Vertex AI via google-genai SDK, and converts responses back.

Supports both ANTHROPIC_VERTEX_BASE_URL and ANTHROPIC_BASE_URL modes.

Usage:
    1. pip install fastapi uvicorn google-genai pyyaml
    2. export VERTEX_PROJECT_ID=your-project-id
    3. python server.py
    4. Set ANTHROPIC_BASE_URL=http://localhost:8765
       or ANTHROPIC_VERTEX_BASE_URL=http://localhost:8765/v1
"""

import json
import uuid
import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse

from config import (
    MODEL_MAP, DEFAULT_GEMINI_MODEL,
    VERTEX_PROJECT_ID, SERVER_PORT,
    gemini_client,
)
from converter import (
    build_gemini_request, build_anthropic_response,
    build_anthropic_stream_events, StreamState,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI()


# ────────────────────────────────────────────
# Shared forwarding logic
# ────────────────────────────────────────────


def _format_gemini_error(e: Exception) -> str:
    """Format verbose Gemini SDK errors for logging (keep only the summary)."""
    msg = str(e)
    # Most Gemini errors start with a short message followed by a period or comma
    # and then a huge JSON blob. We only want the short part.
    if "\n" in msg:
        return msg.split("\n")[0]
    if "{" in msg and "error" in msg:
        # If it looks like a JSON block, split at the first occurrence
        idx = msg.find("{")
        return msg[:idx].strip(". ,")
    return msg


async def forward_to_gemini(
    gemini_model: str,
    response_model: str,
    stream: bool,
    anthropic_body: dict,
):
    """Forward converted request to Gemini and return Anthropic-format response."""
    kwargs = build_gemini_request(anthropic_body)
    logger.info("forwarding to gemini model: %s stream=%s", gemini_model, stream)

    if not stream:
        try:
            resp = await gemini_client.aio.models.generate_content(
                model=gemini_model,
                **kwargs,
            )
        except Exception as e:
            msg = str(e)
            logger.error("gemini error: %s", _format_gemini_error(e))
            return JSONResponse(
                {"error": {"message": msg, "type": "api_error"}},
                status_code=500,
            )

        anthropic_resp = build_anthropic_response(resp, response_model)
        usage = anthropic_resp.get("usage", {})
        logger.info(
            "tokens: input=%d, output=%d, cache=%d, thinking=%d",
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
            usage.get("thinking_tokens", 0),
        )
        logger.info("anthropic response: %s", json.dumps(anthropic_resp, ensure_ascii=False)[:500])
        return JSONResponse(anthropic_resp)
    else:
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"

        async def event_stream():
            state = StreamState(msg_id, response_model)
            total_usage = {"input": 0, "output": 0, "cache": 0, "thinking": 0}
            try:
                async for chunk in await gemini_client.aio.models.generate_content_stream(
                    model=gemini_model,
                    **kwargs,
                ):
                    if chunk.usage_metadata:
                        u = chunk.usage_metadata
                        if u.prompt_token_count:
                            total_usage["input"] = u.prompt_token_count
                        if u.candidates_token_count:
                            total_usage["output"] = u.candidates_token_count
                        if u.cached_content_token_count:
                            total_usage["cache"] = u.cached_content_token_count
                        if u.thoughts_token_count:
                            total_usage["thinking"] = u.thoughts_token_count

                    events = build_anthropic_stream_events(chunk, state)
                    for evt in events:
                        yield f"event: {evt['type']}\ndata: {json.dumps(evt)}\n\n"

                logger.info(
                    "stream tokens: input=%d, output=%d, cache=%d, thinking=%d",
                    total_usage["input"],
                    total_usage["output"],
                    total_usage["cache"],
                    total_usage["thinking"],
                )
            except Exception as e:
                msg = str(e)
                logger.error("gemini stream error: %s", _format_gemini_error(e))
                error_event = {
                    "type": "error",
                    "error": {"type": "api_error", "message": msg},
                }
                yield f"event: error\ndata: {json.dumps(error_event)}\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")


# ────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────


@app.post("/v1/messages")
async def proxy_anthropic_direct(request: Request):
    if not VERTEX_PROJECT_ID:
        return JSONResponse(
            {"error": {"type": "configuration_error", "message": "VERTEX_PROJECT_ID is required in config"}},
            status_code=500,
        )

    anthropic_body = await request.json()

    raw_model = anthropic_body.get("model", "")
    base_model = raw_model.split("@")[0] if "@" in raw_model else raw_model

    # Try exact match, then try prefix match
    gemini_model = MODEL_MAP.get(base_model)
    if not gemini_model:
        # Try to find a prefix match (e.g. 'claude-haiku-4-5' matches 'claude-haiku-4-5-20251001')
        for k, v in MODEL_MAP.items():
            if base_model.startswith(k):
                gemini_model = v
                break
    if not gemini_model:
        gemini_model = DEFAULT_GEMINI_MODEL

    logger.info("model mapping: %s -> %s", raw_model, gemini_model)

    stream = anthropic_body.get("stream", False)

    return await forward_to_gemini(gemini_model, raw_model, stream, anthropic_body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=SERVER_PORT)
