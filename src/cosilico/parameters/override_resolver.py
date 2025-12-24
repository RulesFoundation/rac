"""Override resolution for parameters.

Handles the hierarchy: statute → regulation → IRS guidance.
IRS guidance files declare `overrides:` pointing to statute parameters.
"""

from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Optional

import yaml


@dataclass
class Override:
    """An override from IRS guidance to a statute parameter."""

    source_path: str  # e.g., "irs/rev-proc-2023-34/eitc-2024.yaml"
    target_path: str  # e.g., "statute/26/32/b/2/A/base_amounts#earned_income_amount"
    implements: str   # e.g., "statute/26/32/j/1" (authority)
    tax_year: int     # e.g., 2024
    value: Any        # The override value
    indexed_by: Optional[str] = None


@dataclass
class OverrideIndex:
    """Index of overrides by target path and tax year."""

    # {target_path: {tax_year: Override}}
    overrides: dict[str, dict[int, Override]] = field(default_factory=dict)

    def add(self, override: Override):
        """Add an override to the index."""
        if override.target_path not in self.overrides:
            self.overrides[override.target_path] = {}
        self.overrides[override.target_path][override.tax_year] = override

    def get(self, target_path: str, tax_year: int) -> Optional[Override]:
        """Get override for a path and tax year."""
        if target_path in self.overrides:
            return self.overrides[target_path].get(tax_year)
        return None

    def has_override(self, target_path: str, tax_year: int) -> bool:
        """Check if an override exists."""
        return self.get(target_path, tax_year) is not None


class OverrideResolver:
    """Resolves parameter values with override hierarchy.

    Scans IRS guidance files for `overrides:` attributes and builds
    an index. When a statute parameter is requested, checks for
    overrides before returning the base value.
    """

    def __init__(self, rules_dir: str):
        """Initialize resolver.

        Args:
            rules_dir: Root directory containing statute/ and irs/ folders.
        """
        self.rules_dir = Path(rules_dir)
        self.index = OverrideIndex()
        self._base_values: dict[str, dict] = {}  # {path: yaml_data}

    def load(self):
        """Load all overrides from IRS guidance files."""
        irs_dir = self.rules_dir / "irs"
        if not irs_dir.exists():
            return

        for yaml_file in irs_dir.rglob("*.yaml"):
            self._load_irs_guidance(yaml_file)

    def _load_irs_guidance(self, filepath: Path):
        """Load overrides from an IRS guidance file."""
        try:
            with open(filepath) as f:
                data = yaml.safe_load(f)

            if not data:
                return

            # Extract tax year from source metadata or filename
            source = data.get("source", {})
            effective_date = source.get("effective_date")
            if isinstance(effective_date, date):
                tax_year = effective_date.year
            elif isinstance(effective_date, str):
                tax_year = int(effective_date.split("-")[0])
            else:
                # Try to extract from filename (e.g., eitc-2024.yaml)
                import re
                match = re.search(r"(\d{4})", filepath.name)
                tax_year = int(match.group(1)) if match else None

            if tax_year is None:
                return

            # Process each parameter definition
            for key, definition in data.items():
                if key in ("source", "_metadata") or not isinstance(definition, dict):
                    continue

                overrides = definition.get("overrides")
                implements = definition.get("implements", "")

                if not overrides:
                    continue

                # Get the value
                value = definition.get("values") or definition.get("value")
                indexed_by = definition.get("indexed_by")

                override = Override(
                    source_path=str(filepath.relative_to(self.rules_dir)),
                    target_path=overrides,
                    implements=implements,
                    tax_year=tax_year,
                    value=value,
                    indexed_by=indexed_by,
                )
                self.index.add(override)

        except Exception as e:
            print(f"Warning: Failed to load IRS guidance {filepath}: {e}")

    def load_base_value(self, path: str) -> Optional[dict]:
        """Load a base parameter YAML file.

        Args:
            path: Statute path like "statute/26/32/b/2/A/base_amounts"

        Returns:
            The YAML data, or None if not found.
        """
        if path in self._base_values:
            return self._base_values[path]

        # Try .yaml extension
        yaml_path = self.rules_dir / f"{path}.yaml"
        if yaml_path.exists():
            with open(yaml_path) as f:
                data = yaml.safe_load(f)
            self._base_values[path] = data
            return data

        return None

    def resolve(
        self,
        path: str,
        fragment: Optional[str] = None,
        tax_year: Optional[int] = None,
        **indices,
    ) -> Any:
        """Resolve a parameter value with override hierarchy.

        Args:
            path: Statute path like "statute/26/32/b/2/A/base_amounts"
            fragment: Fragment like "earned_income_amount"
            tax_year: Tax year for override lookup
            **indices: Index values (e.g., num_qualifying_children=2)

        Returns:
            The resolved value (override if available, else base).
        """
        if tax_year is None:
            tax_year = date.today().year

        # Build full path with fragment
        full_path = f"{path}#{fragment}" if fragment else path

        # Check for override
        override = self.index.get(full_path, tax_year)
        if override:
            value = override.value
            # Handle indexed values
            if indices and isinstance(value, dict):
                for idx_name, idx_val in indices.items():
                    if idx_val in value:
                        return value[idx_val]
                    elif str(idx_val) in value:
                        return value[str(idx_val)]
                    elif isinstance(idx_val, int) and idx_val in value:
                        return value[idx_val]
            return value

        # Fall back to base value
        base_data = self.load_base_value(path)
        if base_data and fragment and fragment in base_data:
            value = base_data[fragment]

            # Handle indexed values
            if indices and isinstance(value, dict):
                for idx_name, idx_val in indices.items():
                    if idx_val in value:
                        return value[idx_val]
                    elif str(idx_val) in value:
                        return value[str(idx_val)]

            return value

        raise KeyError(f"Parameter not found: {full_path}")


def create_resolver(rules_dir: str) -> OverrideResolver:
    """Create and initialize an override resolver."""
    resolver = OverrideResolver(rules_dir)
    resolver.load()
    return resolver
