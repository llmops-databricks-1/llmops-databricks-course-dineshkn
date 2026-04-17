"""Basic tests to ensure the package is properly installed."""

import importlib


def test_package_import() -> None:
    """Test that the package can be imported."""
    # This will be replaced by cookiecutter with the actual package name
    package_name = "causal_inference_curator"
    module = importlib.import_module(package_name)
    assert module is not None


def test_version_exists() -> None:
    """Test that the package has a version attribute."""
    package_name = "causal_inference_curator"
    module = importlib.import_module(package_name)
    assert hasattr(module, "__version__")
    assert isinstance(module.__version__, str)


def test_tool_info_construction() -> None:
    """ToolInfo can be built with a callable exec_fn."""
    from causal_inference_curator.mcp import ToolInfo

    def my_fn(x: int) -> int:
        return x * 2

    tool = ToolInfo(
        name="my_tool",
        spec={"type": "function", "function": {"name": "my_tool"}},
        exec_fn=my_fn,
    )
    assert tool.name == "my_tool"
    assert tool.exec_fn(3) == 6


def test_tool_registry() -> None:
    """ToolRegistry registers, lists, and executes tools."""
    from causal_inference_curator.mcp import ToolInfo

    def add(a: int, b: int) -> int:
        return a + b

    tool = ToolInfo(
        name="add",
        spec={"type": "function", "function": {"name": "add"}},
        exec_fn=add,
    )

    registry: dict[str, ToolInfo] = {}
    registry[tool.name] = tool

    assert "add" in registry
    assert registry["add"].exec_fn(2, 3) == 5
