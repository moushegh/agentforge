"""
MCP Tool Registry
-----------------
Connects to MCP servers over SSE and registers their tools with autogen agents.

All sync→async bridging uses a background thread with its own event loop,
preventing any conflict with nest_asyncio or the main asyncio event loop.
"""

import asyncio
import inspect
import json
import re
import threading
import typing
from typing import Any, Callable, Optional

# ---------------------------------------------------------------------------
# Global tool registry — populated during register_mcp_tools()
# maps tool_name → server_url so text-format tool calls can be dispatched
# ---------------------------------------------------------------------------
_TOOL_REGISTRY: dict[str, str] = {}


def get_tool_registry() -> dict[str, str]:
    """Return the global {tool_name: server_url} registry."""
    return _TOOL_REGISTRY


def detect_text_tool_call(content: str) -> "dict | None":
    """
    Try to parse a text-formatted tool call from message content.
    Handles:
      - Valid JSON:  {"name": "fetch", "parameters": {"url": "..."}}
      - Python dicts: {"name": "fetch", "parameters": {"url": "...", "raw": False}}
      - Embedded JSON inside prose
    Returns {"name": ..., "arguments": {...}} or None.
    """
    import ast

    def _extract(data: dict) -> "dict | None":
        if not (isinstance(data, dict) and "name" in data):
            return None
        name = data["name"]
        args = data.get("parameters") or data.get("arguments") or {}
        if isinstance(args, dict):
            return {"name": name, "arguments": args}
        return None

    text = content.strip()

    # 1. Try direct JSON parse
    try:
        data = json.loads(text)
        result = _extract(data)
        if result:
            return result
    except Exception:
        pass

    # 2. Normalise Python literals → JSON, then parse
    # Replace standalone Python keywords only (word-boundary safe)
    import re as _re
    normalised = _re.sub(r'\bFalse\b', 'false', text)
    normalised = _re.sub(r'\bTrue\b', 'true', normalised)
    normalised = _re.sub(r'\bNone\b', 'null', normalised)
    try:
        data = json.loads(normalised)
        result = _extract(data)
        if result:
            return result
    except Exception:
        pass

    # 3. Use ast.literal_eval for full Python dict syntax
    try:
        data = ast.literal_eval(text)
        result = _extract(data)
        if result:
            return result
    except Exception:
        pass

    # 4. Find the first {...} object in mixed text and try the above
    for m in _re.finditer(r'\{', text):
        # Find the matching closing brace
        start = m.start()
        depth = 0
        for i, ch in enumerate(text[start:], start):
            if ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    chunk = text[start:i+1]
                    for parser in (
                        lambda s: json.loads(s),
                        lambda s: json.loads(_re.sub(r'\bFalse\b', 'false',
                                             _re.sub(r'\bTrue\b', 'true',
                                             _re.sub(r'\bNone\b', 'null', s)))),
                        lambda s: ast.literal_eval(s),
                    ):
                        try:
                            data = parser(chunk)
                            result = _extract(data)
                            if result:
                                return result
                        except Exception:
                            pass
                    break
    return None


def execute_text_tool_call(name: str, arguments: dict) -> "str | None":
    """
    Execute a tool by name using the global registry.
    Returns the tool result string, or None if the tool is not registered.
    Retries on transient SSE reconnect errors (supergateway restart window).
    """
    import time as _time
    server_url = _TOOL_REGISTRY.get(name)
    if not server_url:
        return None
    last_exc: BaseException = RuntimeError("unknown")
    for attempt in range(4):
        try:
            return _run_in_thread(_call_tool(server_url, name, arguments))
        except Exception as exc:
            last_exc = exc
            err_str = str(exc)
            if attempt < 3 and ("RemoteProtocol" in err_str or
                                "Server disconnected" in err_str or
                                "Connection" in err_str):
                wait = 2 * (attempt + 1)
                print(f"[MCP] text-tool '{name}' attempt {attempt+1} failed "
                      f"({exc!r}); retrying in {wait}s…")
                _time.sleep(wait)
                continue
            raise
    raise last_exc


# ---------------------------------------------------------------------------
# Internal helper: run a coroutine in a dedicated background thread
# ---------------------------------------------------------------------------

def _run_in_thread(coro):
    """
    Execute *coro* in a brand-new event loop running in a background thread.
    Safe to call from sync or async contexts, with or without nest_asyncio.
    """
    result_box: list = [None]
    error_box:  list = [None]

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            result_box[0] = loop.run_until_complete(coro)
        except BaseException as exc:
            # Python 3.11 asyncio.TaskGroup raises ExceptionGroup; unwrap to
            # surface the real underlying error instead of the opaque wrapper.
            if hasattr(exc, "exceptions") and exc.exceptions:
                error_box[0] = exc.exceptions[0]
            else:
                error_box[0] = exc
        finally:
            loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join(timeout=60)   # 60 s — long enough for large web pages

    if t.is_alive():
        raise TimeoutError("MCP call timed out after 60 s")
    if error_box[0] is not None:
        raise error_box[0]
    return result_box[0]


# ---------------------------------------------------------------------------
# Low-level async helpers
# ---------------------------------------------------------------------------

async def _list_tools(server_url: str) -> list:
    """Return the list of Tool objects advertised by an MCP server."""
    from mcp.client.sse import sse_client
    from mcp import ClientSession

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.list_tools()
            return result.tools


async def _call_tool(server_url: str, tool_name: str, arguments: dict) -> str:
    """Call *tool_name* on *server_url* and return the result as a string."""
    from mcp.client.sse import sse_client
    from mcp import ClientSession

    async with sse_client(server_url) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, arguments)
            if result.content:
                parts = []
                for item in result.content:
                    if hasattr(item, "text") and item.text:
                        parts.append(item.text)
                    elif hasattr(item, "data"):
                        parts.append(str(item.data))
                return "\n".join(parts) if parts else "Success (no output)"
            return "Success (no output)"


# ---------------------------------------------------------------------------
# Tool factory — builds a properly-typed wrapper from the MCP inputSchema
# ---------------------------------------------------------------------------

_JSON_TO_PYTHON = {"string": str, "number": float, "integer": int, "boolean": bool}


def _make_tool_func(server_url: str, tool_name: str, input_schema: dict) -> Callable:
    """
    Create a synchronous wrapper whose *signature* matches the MCP tool's
    inputSchema so autogen generates a correct JSON schema for the model.
    """
    props    = input_schema.get("properties", {}) if input_schema else {}
    required = set(input_schema.get("required", [])) if input_schema else set()

    # Build a list of inspect.Parameter objects in required-first order
    ordered_names = sorted(props.keys(), key=lambda k: (k not in required, k))
    params: list[inspect.Parameter] = []
    annotations: dict[str, Any] = {"return": str}

    for pname in ordered_names:
        pschema = props[pname]
        raw_type = _JSON_TO_PYTHON.get(pschema.get("type", "string"), str)
        if pname in required:
            py_type = raw_type
            param = inspect.Parameter(
                pname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                annotation=py_type,
            )
        else:
            py_type = Optional[raw_type]
            param = inspect.Parameter(
                pname,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                default=None,
                annotation=py_type,
            )
        params.append(param)
        annotations[pname] = py_type

    # Capture for closure
    _url  = server_url
    _name = tool_name
    _req  = required
    _all  = list(ordered_names)

    def tool_func(*args, **kwargs) -> str:  # type: ignore[override]
        # Merge positional args into kwargs by position
        merged: dict[str, Any] = {}
        for i, val in enumerate(args):
            if i < len(_all):
                merged[_all[i]] = val
        merged.update(kwargs)
        # Drop None optional args
        call_args = {k: v for k, v in merged.items() if v is not None or k in _req}
        # Retry on connection errors — supergateway briefly restarts between calls
        import time as _time
        last_exc: BaseException = RuntimeError("unknown")
        for attempt in range(4):
            try:
                return _run_in_thread(_call_tool(_url, _name, call_args))
            except Exception as exc:
                last_exc = exc
                err_str = str(exc)
                if attempt < 3 and ("RemoteProtocol" in err_str or
                                    "Server disconnected" in err_str or
                                    "Connection" in err_str):
                    wait = 2 * (attempt + 1)   # 2s, 4s, 6s
                    print(f"[MCP] '{_name}' attempt {attempt+1} failed ({exc!r}); "
                          f"retrying in {wait}s…")
                    _time.sleep(wait)
                    continue
                raise
        raise last_exc

    # Attach the proper signature so autogen generates the right JSON schema
    tool_func.__name__        = tool_name
    tool_func.__annotations__ = annotations
    tool_func.__signature__   = inspect.Signature(params, return_annotation=str)
    return tool_func


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_mcp_tools(config: dict, agents: dict, executor) -> None:
    """
    Read the `mcp_servers` block from *config*, connect to each server,
    discover its tools, and register them with the specified autogen agents.

    Parameters
    ----------
    config   : parsed agent_config.yaml dict
    agents   : {agent_key: AssistantAgent} dict built by LocalMultiAgentTeam
    executor : UserProxyAgent used to execute function calls
    """
    from autogen import register_function

    mcp_servers = config.get("mcp_servers", {})
    if not mcp_servers:
        print("[MCP] No mcp_servers found in config — skipping tool registration.")
        return

    print("\n[MCP] Registering tools from MCP servers...")

    for server_key, server_cfg in mcp_servers.items():
        url = server_cfg.get("url", "")
        agent_names = server_cfg.get("agents", [])

        if not url:
            print(f"  [{server_key}] WARNING: no 'url' defined — skipped.")
            continue

        # ---- Discover tools ------------------------------------------------
        tools = None
        for attempt in range(3):
            try:
                tools = _run_in_thread(_list_tools(url))
                break
            except BaseException as exc:
                if attempt < 2:
                    import time; time.sleep(2)
                else:
                    print(f"  [{server_key}] WARNING: cannot reach {url} — {exc}")
        if tools is None:
            continue

        if not tools:
            print(f"  [{server_key}] No tools advertised at {url}.")
            continue

        print(f"  [{server_key}] {url} — {len(tools)} tool(s): "
              f"{[t.name for t in tools]}")

        # ---- Register each tool with the configured agents -----------------
        for tool in tools:
            # Extract inputSchema: MCP Tool objects expose it as .inputSchema
            raw_schema = getattr(tool, "inputSchema", None)
            if hasattr(raw_schema, "model_dump"):
                input_schema = raw_schema.model_dump()  # Pydantic v2
            elif hasattr(raw_schema, "dict"):
                input_schema = raw_schema.dict()         # Pydantic v1
            elif isinstance(raw_schema, dict):
                input_schema = raw_schema
            else:
                input_schema = {}
            fn = _make_tool_func(url, tool.name, input_schema)
            description = (tool.description or tool.name).strip()
            registered_to = []

            for agent_name in agent_names:
                if agent_name not in agents:
                    print(f"    WARNING: agent '{agent_name}' not found — skipped.")
                    continue
                try:
                    register_function(
                        fn,
                        caller=agents[agent_name],
                        executor=executor,
                        name=tool.name,
                        description=description,
                    )
                    registered_to.append(agent_name)
                except Exception as exc:
                    print(f"    WARNING: could not register '{tool.name}' "
                          f"to '{agent_name}': {exc}")

            if registered_to:
                print(f"    ✓ '{tool.name}' → {registered_to}")
                # Also add to global text-tool-call registry
                _TOOL_REGISTRY[tool.name] = url

    print("[MCP] Tool registration complete.\n")
