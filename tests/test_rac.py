"""Core RAC tests."""

from datetime import date

import pytest

from rac import compile, execute, generate_rust, parse


class TestParser:
    def test_parse_scalar_variable(self):
        module = parse("""
            variable gov/tax/rate:
                from 2024-01-01: 0.25
        """)
        assert len(module.variables) == 1
        assert module.variables[0].path == "gov/tax/rate"

    def test_parse_entity_variable(self):
        module = parse("""
            variable person/tax:
                entity: person
                from 2024-01-01: income * 0.2
        """)
        assert module.variables[0].entity == "person"

    def test_parse_temporal_ranges(self):
        module = parse("""
            variable gov/threshold:
                from 2023-01-01 to 2023-12-31: 10000
                from 2024-01-01: 12000
        """)
        assert len(module.variables[0].values) == 2

    def test_parse_expressions(self):
        module = parse("""
            variable test/expr:
                from 2024-01-01: max(0, income - 10000) * 0.22
        """)
        assert module.variables[0].values[0].expr is not None

    def test_parse_conditional(self):
        module = parse("""
            variable test/cond:
                from 2024-01-01:
                    if income > 50000: income * 0.3
                    else: income * 0.1
        """)
        assert module.variables[0].values[0].expr.type == "cond"

    def test_parse_entity_declaration(self):
        module = parse("""
            entity person:
                age: int
                income: float
        """)
        assert len(module.entities) == 1
        assert module.entities[0].name == "person"


class TestCompiler:
    def test_compile_scalar(self):
        module = parse("""
            variable gov/rate:
                from 2024-01-01: 0.25
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        assert "gov/rate" in ir.variables

    def test_temporal_resolution(self):
        module = parse("""
            variable gov/val:
                from 2023-01-01 to 2023-12-31: 100
                from 2024-01-01: 200
        """)
        ir_2023 = compile([module], as_of=date(2023, 6, 1))
        ir_2024 = compile([module], as_of=date(2024, 6, 1))

        # Check the literal values
        assert ir_2023.variables["gov/val"].expr.value == 100
        assert ir_2024.variables["gov/val"].expr.value == 200

    def test_amendment_override(self):
        base = parse("""
            variable gov/rate:
                from 2024-01-01: 0.10
        """)
        amendment = parse("""
            amend gov/rate:
                from 2024-06-01: 0.15
        """)
        ir = compile([base, amendment], as_of=date(2024, 7, 1))
        assert ir.variables["gov/rate"].expr.value == 0.15

    def test_dependency_ordering(self):
        module = parse("""
            variable gov/base:
                from 2024-01-01: 1000

            variable gov/rate:
                from 2024-01-01: 0.1

            variable gov/tax:
                from 2024-01-01: gov/base * gov/rate
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        # tax should come after base and rate
        assert ir.order.index("gov/tax") > ir.order.index("gov/base")
        assert ir.order.index("gov/tax") > ir.order.index("gov/rate")


class TestExecutor:
    def test_execute_scalar(self):
        module = parse("""
            variable gov/rate:
                from 2024-01-01: 0.25
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        result = execute(ir, {})
        assert result.scalars["gov/rate"] == 0.25

    def test_execute_arithmetic(self):
        module = parse("""
            variable test/a:
                from 2024-01-01: 10

            variable test/b:
                from 2024-01-01: test/a * 2 + 5
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        result = execute(ir, {})
        assert result.scalars["test/b"] == 25

    def test_execute_builtin_functions(self):
        module = parse("""
            variable test/clipped:
                from 2024-01-01: clip(150, 0, 100)
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        result = execute(ir, {})
        assert result.scalars["test/clipped"] == 100

    def test_execute_conditional(self):
        module = parse("""
            variable test/threshold:
                from 2024-01-01: 50

            variable test/result:
                from 2024-01-01:
                    if test/threshold > 40: 100
                    else: 0
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        result = execute(ir, {})
        assert result.scalars["test/result"] == 100

    def test_execute_entity_variable(self):
        module = parse("""
            variable person/doubled:
                entity: person
                from 2024-01-01: income * 2
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        data = {"person": [{"id": 1, "income": 1000}, {"id": 2, "income": 2000}]}
        result = execute(ir, data)
        assert result.entities["person"]["person/doubled"] == [2000, 4000]


class TestRustCodegen:
    def test_generate_rust_basic(self):
        module = parse("""
            variable gov/rate:
                from 2024-01-01: 0.25

            variable gov/base:
                from 2024-01-01: 1000

            variable gov/tax:
                from 2024-01-01: gov/base * gov/rate
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        rust_code = generate_rust(ir)

        assert "pub struct Scalars" in rust_code
        assert "fn compute" in rust_code
        assert "gov_rate" in rust_code
        assert "gov_base" in rust_code

    def test_generate_rust_with_entity(self):
        module = parse("""
            entity person:
                age: int
                income: float
        """)
        ir = compile([module], as_of=date(2024, 6, 1))
        rust_code = generate_rust(ir)

        assert "pub struct PersonInput" in rust_code
        assert "pub age: i64" in rust_code
        assert "pub income: f64" in rust_code
