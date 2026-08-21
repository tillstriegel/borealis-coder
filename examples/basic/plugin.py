"""Example opted-in workspace plugin: .borealis/plugins/project_tools.py."""
from borealis_coder.models import Effect, ToolResult
from borealis_coder.tools import FunctionTool, object_schema


def register(registry):
    async def project_name(arguments, context):
        return ToolResult(context.workspace.name)

    registry.register(FunctionTool(
        name="project_name",
        description="Return this workspace's directory name.",
        parameters=object_schema({}),
        function=project_name,
        effect=Effect.READ,
        concurrent=True,
    ))
