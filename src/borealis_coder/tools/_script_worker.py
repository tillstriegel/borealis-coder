"""Private QuickJS worker. No Python objects or callbacks are exposed to JavaScript."""
from __future__ import annotations

import json
import math
import sys
import time
from typing import Any

# The dispatch function stays in Python. Programs cannot access its closure.
_BOOTSTRAP = r'''
(() => {
    const encode = JSON.stringify.bind(JSON), decode = JSON.parse.bind(JSON);
    const AsyncFunction = Object.getPrototypeOf(async function(){}).constructor;
    const pending = new Map();
    let calls = [], output = [], serial = 0, state = "running", error = "";
    const tools = new Proxy(Object.create(null), {get: (_, name) => args =>
        new Promise((resolve, reject) => {
            const id = ++serial;
            // Serialize here so arguments cannot change after admission.
            calls.push(encode({id, name, arguments: args}));
            pending.set(id, {resolve, reject});
        })
    });
    const text = (...values) => output.push(values.map(v =>
        typeof v === "string" ? v : encode(v)).join(" ") + "\n");
    return (operation, value) => {
        if (operation === "start") {
            new AsyncFunction("tools", "text", "print", value)(tools, text, text).then(
                () => { state = "completed"; },
                e => { state = "error"; error = String(e); }
            );
        } else if (operation === "result") {
            const response = decode(value), item = pending.get(response.id);
            if (!item) throw Error("Invalid result identity");
            pending.delete(response.id);
            if (response.result.is_error) {
                const failure = Error(response.result.output);
                failure.result = response.result;
                item.reject(failure);
            } else item.resolve(response.result.data ?? response.result);
        } else {
            const packet = encode({calls: calls.map(c => decode(c)), output, state, error,
                                   pending: pending.size});
            calls = []; output = [];
            return packet;
        }
    };
})()
'''


def main() -> None:
    try:
        import quickjs
    except ImportError as error:
        raise ValueError("QuickJS is unavailable; install the declared quickjs-ng dependency") from error
    # Parent sends trusted limits before any source is compiled.
    header = sys.stdin.buffer.readline(16_384)
    config = json.loads(header)
    limit = config["ipc_bytes"]
    source = sys.stdin.buffer.readline(limit + 1)
    if len(source) > limit or not source.endswith(b"\n"):
        raise ValueError("Source IPC limit exceeded")
    started = time.process_time()
    ctx = quickjs.Context()
    ctx.set_memory_limit(config["memory_bytes"])
    ctx.set_max_stack_size(config["stack_bytes"])
    if sys.platform != "win32":
        import resource
        cpu = math.ceil(config["cpu_seconds"]) + 1
        resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
    dispatch = ctx.eval(_BOOTSTRAP)

    def remaining() -> None:
        cpu_left = config["cpu_seconds"] - (time.process_time() - started)
        if cpu_left <= 0:
            raise ValueError("Script CPU limit exceeded")
        ctx.set_time_limit(cpu_left)

    remaining()
    dispatch("start", json.loads(source))
    while True:
        remaining()
        while ctx.execute_pending_job():
            remaining()
        packet_text = dispatch("drain", "")
        encoded = packet_text.encode("utf-8")
        if len(encoded) > limit:
            raise ValueError("Script IPC limit exceeded")
        packet = json.loads(packet_text)
        sys.stdout.buffer.write(encoded + b"\n")
        sys.stdout.buffer.flush()
        if not packet["pending"]:
            if packet["state"] == "running":
                raise ValueError("Script has an unresolved promise without active tools")
            break
        response = sys.stdin.buffer.readline(limit + 1)
        if len(response) > limit or not response.endswith(b"\n"):
            raise ValueError("Invalid result IPC")
        remaining()
        dispatch("result", response.decode("utf-8"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        # Never include engine stacks, source, file contents, or Python tracebacks.
        packet: dict[str, Any] = {"calls": [], "output": [], "state": "error",
                                  "error": str(error).splitlines()[0][:1000] if str(error) else "Script execution failed", "pending": 0}
        sys.stdout.write(json.dumps(packet, ensure_ascii=True) + "\n")
        sys.stdout.flush()
        raise SystemExit(1) from None
