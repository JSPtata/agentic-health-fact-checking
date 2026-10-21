"""Ollama backend with source-controlled quotation anchors."""
import hashlib
import json
import re
import urllib.request

from verify_claim_agent import dump
from verification_schemas import Finding, Critique


PROTOCOL = "quote-anchors-v3"


def compact_schema(value):
    if isinstance(value, dict):
        return {
            key: compact_schema(item)
            for key, item in value.items()
            if key not in ("title", "description", "default")
        }
    if isinstance(value, list):
        return [compact_schema(item) for item in value]
    return value


def quote_anchors(text):
    segments = re.split(
        r"(?<=[.!?])(?=\s)|(?<=\n)", text
    )
    units = []

    for segment in segments:
        while len(segment) > 220:
            cut = segment.rfind(" ", 12, 220)
            if cut < 12:
                cut = 220
            units.append(segment[:cut])
            segment = segment[cut:]

        if segment:
            units.append(segment)

    spans, pending = [], ""

    for unit in units:
        if (
            pending
            and len(pending) + len(unit) > 220
            and len(" ".join(pending.split())) >= 12
        ):
            spans.append(pending)
            pending = ""

        pending += unit

    if pending:
        if len(" ".join(pending.split())) >= 12:
            spans.append(pending)
        elif spans:
            spans[-1] += pending
        else:
            raise ValueError(
                "Passage too short for a verified quotation"
            )

    if not spans or "".join(spans) != text:
        raise ValueError(
            "Quotation anchors do not cover the passage"
        )

    return {
        f"Q{i}": value
        for i, value in enumerate(spans, 1)
    }


def prepare_request(schema, prompt):
    prefix, marker, body = prompt.partition("INPUT:\n")

    if not marker:
        raise ValueError(
            "Expected INPUT marker in verification prompt"
        )

    context = json.loads(body)
    anchors = quote_anchors(context["passage"])
    sid = context["source_id"]

    wire_schema = schema.model_json_schema()
    id_type = {
        "type": "string",
        "enum": list(anchors),
    }

    if schema is Finding:
        properties = wire_schema["properties"]
        properties.pop("quotes")
        properties["source_id"] = {
            "type": "string",
            "enum": [sid],
        }
        properties["quote_ids"] = {
            "type": "array",
            "items": id_type,
            "minItems": 1,
            "maxItems": 2,
            "uniqueItems": True,
        }
        wire_schema["required"] = [
            "quote_ids" if name == "quotes" else name
            for name in wire_schema["required"]
        ]

    elif schema is Critique:
        objection = wire_schema["$defs"]["Objection"]
        objection["properties"].pop("source_quote")

        objection["properties"]["quote_id"] = {
            "anyOf": [id_type, {"type": "null"}],
            "default": None,
        }
        objection["properties"]["source_id"] = {
            "anyOf": [
                {"type": "string", "enum": [sid]},
                {"type": "null"},
            ],
            "default": None,
        }

        if "required" in objection:
            objection["properties"]["source_id"] = {
                "type": "string",
                "enum": [sid],
            }

            objection["required"] = list(dict.fromkeys(
                [
                    "quote_id"
                    if name == "source_quote"
                    else name
                    for name in objection.get("required", [])
                ]
                + ["source_id"]
            ))
    else:
        raise ValueError(
            "Unsupported local verification schema"
        )

    # All passage text remains present and unchanged.
    context["passage"] = anchors

    instruction = (
        "\nCITATION FORMAT: The passage is numbered Q1, Q2, etc. "
        "For findings return quote_ids, not quotes. "
        "For review objections return quote_id or null, "
        "not source_quote. Select only supplied IDs. "
        "Do not write or paraphrase quotations. "
        "IDs identify text, not its validity.\n"
    )

    request_prompt = (
        prefix
        + instruction
        + "INPUT:\n"
        + json.dumps(
            context,
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )

    return (
        request_prompt,
        compact_schema(wire_schema),
        anchors,
        sid,
    )


def decode_result(schema, value, anchors, sid):
    if schema is Finding:
        ids = value.pop("quote_ids")

        if not isinstance(ids, list) or not 1 <= len(ids) <= 2:
            raise ValueError(
                "Expected one or two quotation IDs"
            )

        if any(
            not isinstance(qid, str) or qid not in anchors
            for qid in ids
        ):
            raise ValueError("Unknown quotation ID")

        if len(set(ids)) != len(ids):
            raise ValueError("Repeated quotation ID")

        value["quotes"] = [anchors[qid] for qid in ids]

    else:
        for objection in value["objections"]:
            qid = objection.pop("quote_id", None)

            if qid is not None:
                if (
                    not isinstance(qid, str)
                    or qid not in anchors
                ):
                    raise ValueError(
                        "Unknown review quotation ID"
                    )

                returned_source = objection.get("source_id")

                if returned_source not in (None, sid):
                    raise ValueError(
                        "Review quotation has an incorrect source ID"
                    )

                # This quotation ID belongs to the single
                # source supplied for this passage.
                objection["source_id"] = sid

            objection["source_quote"] = (
                anchors[qid] if qid is not None else None
            )

    return schema.model_validate(value)


class AnchoredBackend:
    def __init__(self, directory, state, model):
        self.directory = directory
        self.state = state
        self.model = model

    def call(self, key, schema, prompt, validator):
        # Reuse previously validated work.
        if key in self.state["outputs"]:
            value = schema.model_validate(
                self.state["outputs"][key]
            )
            validator(value)
            return value

        (
            request_prompt,
            wire_schema,
            anchors,
            sid,
        ) = prepare_request(schema, prompt)

        previous = [
            event
            for event in self.state["events"]
            if (
                event["key"] == key
                and event.get("protocol") == PROTOCOL
            )
        ]

        # At most two attempts per stage under this new protocol.
        for attempt in range(len(previous), 2):
            feedback = ""

            if attempt:
                errors = [
                    event.get("error", "")
                    for event in self.state["events"]
                    if (
                        event["key"] == key
                        and event.get("protocol") == PROTOCOL
                    )
                ]
                feedback = (
                    "\nCORRECT THIS VALIDATION ERROR: "
                    + (
                        errors[-1][:160]
                        if errors
                        else "Invalid JSON"
                    )
                )

            actual_prompt = request_prompt + feedback

            size = (
                len(actual_prompt.encode())
                + 2 * len(json.dumps(wire_schema).encode())
            )
            if size > 6100:
                raise ValueError(
                    "Anchored context exceeds the local input "
                    "budget; state preserved"
                )

            payload = {
                "model": self.model,
                "stream": False,
                "think": False,
                "format": wire_schema,
                "messages": [{
                    "role": "user",
                    "content": actual_prompt,
                }],
                "keep_alive": "5m",
                "options": {
                    "temperature": 0,
                    "num_ctx": 8192,
                    "num_predict": 1536,
                },
            }

            event = {
                "key": key,
                "protocol": PROTOCOL,
                "status": "started",
                "prompt_sha256": hashlib.sha256(
                    actual_prompt.encode()
                ).hexdigest(),
                "quote_anchors": anchors,
            }
            self.state["events"].append(event)
            dump(
                self.directory / "state.json",
                self.state,
            )

            try:
                request = urllib.request.Request(
                    "http://127.0.0.1:11434/api/chat",
                    data=json.dumps(payload).encode(),
                    headers={
                        "Content-Type": "application/json"
                    },
                )

                with urllib.request.urlopen(
                    request, timeout=600
                ) as response:
                    raw = json.load(response)

                filename = (
                    "responses/"
                    f"{len(self.state['events']):04d}.json"
                )
                dump(self.directory / filename, raw)

                event.update(
                    response_file=filename,
                    prompt_tokens=raw.get(
                        "prompt_eval_count"
                    ),
                    output_tokens=raw.get("eval_count"),
                )

                if (
                    not raw.get("done")
                    or raw.get("done_reason") == "length"
                ):
                    raise ValueError(
                        "Ollama response was incomplete"
                    )

                value = decode_result(
                    schema,
                    json.loads(raw["message"]["content"]),
                    anchors,
                    sid,
                )
                validator(value)

                self.state["outputs"][key] = (
                    value.model_dump()
                )
                event["status"] = "validated"
                return value

            except (ValueError, KeyError, TypeError) as error:
                event.update(
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                )
                if attempt == 1:
                    raise

            except OSError as error:
                event.update(
                    status="error",
                    error=f"{type(error).__name__}: {error}",
                )
                raise

            finally:
                dump(
                    self.directory / "state.json",
                    self.state,
                )

        raise RuntimeError(
            f"Two anchored attempts exhausted for {key}; "
            "state preserved"
        )