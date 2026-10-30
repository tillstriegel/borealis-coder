# Extending Borealis

## Tool model

A tool has four responsibilities:

1. Name and explain one operation.
2. Define a strict JSON-object schema.
3. Declare an effect category.
4. Execute against `ToolContext` and return a bounded `ToolResult`.

The registry validates arguments before the tool sees them. All properties must be listed in `required`; logically optional fields use a nullable type and explicit default handling.

Minimal example:

```python
from borealis_coder.models import Effect, ToolResult
from borealis_coder.tools import Tool, ToolContext


class CountPythonFiles(Tool):
    name = "count_python_files"
    description = "Count Python source files inside the workspace."
    effect = Effect.READ
    parameters = {
        "type": "object",
        "properties": {
            "path": {"type": ["string", "null"]},
        },
        "required": ["path"],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict, context: ToolContext) -> ToolResult:
        base = context.roots.resolve(arguments.get("path") or ".", must_exist=True, kind="dir").path
        count = sum(1 for item in base.rglob("*.py") if item.is_file())
        return ToolResult(output=str(count), data={"count": count})
```

### Script bridge

Registered plugin and MCP tools are available as `await tools.NAME(arguments)`
inside Script. The same policy/approval/lifecycle path applies. Tools must declare
accurate effects and mutation scopes; tools with effects are serialized even if
`concurrent = True`. Cancellation must stop and drain owned work before returning.
Do not suppress host cancellation or budget exceptions.

Return `ToolResult(output=...)` for text, or optional JSON-compatible
`data={...}` for structured program results. Script resolves `data` when present,
otherwise `{output, is_error, metadata}`. Direct calls keep their text preview.
Script redacts and archives nested results before transport and rejects oversized,
overdeep or non-JSON results. Tool errors reject with `error.result` and count
against job status even if caught. Script/job control and delegation are excluded
from nested calls through `Tool.script_accessible = False`, including aliases;
read-only delegates keep their direct tool palette.

## Installed-package tools

Publish a package with an entry point in the `borealis.tools` group:

```toml
[project.entry-points."borealis.tools"]
count_python = "my_package.tools:CountPythonFiles"
```

The loaded object may be a `Tool` instance, a `Tool` subclass, a zero-argument factory, or an iterable of those forms. Installed packages are trusted code.

## Workspace plugins

A workspace may place Python files in `.borealis/plugins`. These are disabled by default because importing them grants arbitrary in-process code execution.

Enable explicitly:

```bash
export BOREALIS_ENABLE_WORKSPACE_PLUGINS=1
```

A plugin exports `register(registry)` and registers tools. Prefer installed packages or MCP for production extensions because they provide a clearer review and deployment boundary.

## Skills

A skill is a Markdown file with a short title/description and procedural guidance. Put skills in `.agents/skills` or `.borealis/skills`.

Borealis injects only compact skill metadata into the initial context. The model calls `read_skill` when the procedure is relevant. This reduces persistent context cost.

## MCP servers

MCP is the preferred boundary for tools that:

- require their own dependency graph
- connect to external services
- need a different language/runtime
- should be deployed separately
- maintain long-lived resources

Set accurate safety annotations and still configure Borealis effect policy conservatively.

## Provider registry

`ProviderRegistry` maps a configuration `type` to a factory. Custom factories receive the selected provider configuration and return a normalized async provider.

```python
from borealis_coder import build_runner
from borealis_coder.providers import ProviderRegistry

registry = ProviderRegistry.with_defaults()
registry.register("internal_gateway", make_internal_provider)
runner = await build_runner(workspace, provider_registry=registry)
```

Keep provider wire code out of tools and the agent loop.

## Event subscribers

Subscribe to `runner.events` for product integration:

```python
async def observe(event):
    await publish_to_ui(event.to_dict())

runner.events.subscribe(observe)
```

Observers should be idempotent and fast. The event bus isolates subscriber failures, but a remote product should queue or batch outbound telemetry.

## Custom approval UI

Pass an async approval callback to `build_runner`. It receives an `ApprovalRequest` with effect, tool, reason, arguments, and session context, and returns the selected decision. ACP uses the same callback abstraction through client JSON-RPC.

## Custom process isolation

Implement the process-driver interface and inject it during assembly when the native and Docker drivers do not match your infrastructure. A remote worker driver can execute commands in a VM/Kubernetes sandbox while preserving normalized results and cancellation.

## Testing extensions

Extension tests should cover:

- schema strictness and invalid arguments
- policy classification
- root/path behavior
- cancellation and timeout
- output truncation
- secret redaction
- deterministic fixture behavior
- server/provider failure cleanup
