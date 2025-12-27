"""Dependency resolver for cross-file statute references.

Resolves references between .rac files, builds a dependency graph,
and provides execution order via topological sort.

Example:
    resolver = DependencyResolver(statute_root=Path("cosilico-us"))
    modules = resolver.resolve_all("statute/26/32/a/1/earned_income_credit")
    # modules is ordered: dependencies before dependents
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from .dsl_parser import Module, parse_file, parse_dsl


class ModuleNotFoundError(Exception):
    """Raised when a referenced module cannot be found."""
    pass


class CircularDependencyError(Exception):
    """Raised when circular dependencies are detected."""
    pass


def extract_dependencies(module: Module) -> list[str]:
    """Extract dependency paths from a module's references block.

    Args:
        module: Parsed DSL module

    Returns:
        List of statute paths referenced by this module
    """
    if not module.imports:
        return []

    return [ref.statute_path for ref in module.imports.references]


@dataclass
class DependencyGraph:
    """Directed acyclic graph of module dependencies.

    Supports:
    - Adding modules with their dependencies
    - Querying dependencies
    - Topological sorting for execution order
    - Circular dependency detection
    """

    _adjacency: dict[str, list[str]] = field(default_factory=dict)

    def add_module(self, name: str, dependencies: list[str]) -> None:
        """Add a module and its dependencies to the graph.

        Args:
            name: Module identifier (usually statute path)
            dependencies: List of modules this one depends on
        """
        self._adjacency[name] = dependencies
        # Ensure all dependencies are in the graph
        for dep in dependencies:
            if dep not in self._adjacency:
                self._adjacency[dep] = []

    def get_dependencies(self, name: str) -> list[str]:
        """Get direct dependencies of a module.

        Args:
            name: Module identifier

        Returns:
            List of dependency module names
        """
        return self._adjacency.get(name, [])

    def topological_sort(self) -> list[str]:
        """Return modules in topological order (dependencies first).

        Returns:
            List of module names, ordered so dependencies come before dependents

        Raises:
            CircularDependencyError: If circular dependencies exist
        """
        # Kahn's algorithm
        # Calculate in-degrees
        in_degree = {node: 0 for node in self._adjacency}
        for node, deps in self._adjacency.items():
            for dep in deps:
                if dep in in_degree:
                    in_degree[node] += 0  # node depends on dep, not the reverse
                # Actually we need reverse: if A depends on B, B must come first
                # So we need to track who depends on each node

        # Rebuild: for each node, who depends on it?
        dependents = {node: [] for node in self._adjacency}
        for node, deps in self._adjacency.items():
            for dep in deps:
                if dep in dependents:
                    dependents[dep].append(node)

        # Recalculate in-degrees (number of unprocessed dependencies)
        in_degree = {node: len(self._adjacency.get(node, [])) for node in self._adjacency}

        # Start with nodes that have no dependencies
        queue = [node for node, degree in in_degree.items() if degree == 0]
        result = []

        while queue:
            node = queue.pop(0)
            result.append(node)

            # For each node that depends on this one, decrement its in-degree
            for dependent in dependents.get(node, []):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    queue.append(dependent)

        if len(result) != len(self._adjacency):
            # Not all nodes processed = cycle exists
            remaining = set(self._adjacency.keys()) - set(result)
            raise CircularDependencyError(
                f"Circular dependency detected involving: {remaining}"
            )

        return result


class ModuleResolver:
    """Resolves statute reference paths to filesystem paths."""

    def __init__(self, statute_root: Path):
        """Initialize resolver with statute root directory.

        Args:
            statute_root: Root directory for statute files (e.g., cosilico-us)
        """
        self.statute_root = statute_root

    def resolve(self, statute_path: str) -> Path:
        """Resolve a statute path to a filesystem path.

        Args:
            statute_path: Path like "statute/26/32/c/2/A/earned_income"

        Returns:
            Absolute path to the .rac file

        Raises:
            ModuleNotFoundError: If file doesn't exist
        """
        # Keep the full path (including statute/ prefix)
        # Files are organized as: cosilico-us/statute/26/32/...
        clean_path = statute_path

        # Try with .rac extension
        file_path = self.statute_root / f"{clean_path}.rac"
        if file_path.exists():
            return file_path

        # Try as directory with same-named file inside
        dir_path = self.statute_root / clean_path
        if dir_path.is_dir():
            # Look for a .rac file with the last component name
            name = Path(clean_path).name
            candidate = dir_path / f"{name}.rac"
            if candidate.exists():
                return candidate

        raise ModuleNotFoundError(f"Cannot resolve '{statute_path}' from {self.statute_root}")


@dataclass
class ResolvedModule:
    """A parsed module with its path and dependencies resolved."""

    path: str
    file_path: Path
    module: Module
    dependencies: list[str]


class DependencyResolver:
    """Full dependency resolver: finds, parses, and orders modules."""

    def __init__(self, statute_root: Path):
        """Initialize with statute root directory.

        Args:
            statute_root: Root directory for statute files
        """
        self.module_resolver = ModuleResolver(statute_root)
        self._cache: dict[str, ResolvedModule] = {}

    def resolve_all(self, entry_point: str) -> list[ResolvedModule]:
        """Resolve entry point and all its dependencies recursively.

        Args:
            entry_point: Starting statute path

        Returns:
            List of ResolvedModule in execution order (dependencies first)
        """
        # Clear cache for fresh resolution
        self._cache.clear()

        # Recursively load all modules
        self._load_recursive(entry_point)

        # Build dependency graph
        graph = DependencyGraph()
        for path, resolved in self._cache.items():
            graph.add_module(path, resolved.dependencies)

        # Get execution order
        order = graph.topological_sort()

        # Return modules in order
        return [self._cache[path] for path in order if path in self._cache]

    def _load_recursive(self, statute_path: str) -> None:
        """Load a module and its dependencies recursively.

        Args:
            statute_path: Statute path to load
        """
        if statute_path in self._cache:
            return

        # Resolve and parse
        try:
            file_path = self.module_resolver.resolve(statute_path)
        except ModuleNotFoundError:
            # Module not found - might be a primitive input
            # Create a placeholder
            self._cache[statute_path] = ResolvedModule(
                path=statute_path,
                file_path=Path(),
                module=None,
                dependencies=[]
            )
            return

        # Try to parse, but handle errors gracefully
        try:
            module = parse_file(str(file_path))
            dependencies = extract_dependencies(module)
        except SyntaxError as e:
            # Parse error - treat as placeholder with warning
            import warnings
            warnings.warn(f"Parse error in {statute_path}: {e}")
            self._cache[statute_path] = ResolvedModule(
                path=statute_path,
                file_path=file_path,
                module=None,
                dependencies=[]
            )
            return

        # Cache this module
        self._cache[statute_path] = ResolvedModule(
            path=statute_path,
            file_path=file_path,
            module=module,
            dependencies=dependencies
        )

        # Recursively load dependencies
        for dep in dependencies:
            self._load_recursive(dep)
