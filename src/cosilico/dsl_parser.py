"""Cosilico DSL Parser.

Parses .cosilico files according to the DSL specification in docs/DSL.md.
This is a recursive descent parser that produces an AST.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Union


class TokenType(Enum):
    # Keywords
    MODULE = "module"
    VERSION = "version"
    JURISDICTION = "jurisdiction"
    IMPORT = "import"
    IMPORTS = "imports"  # Import block for variables and parameters
    REFERENCES = "references"  # Deprecated alias for imports (backwards compat)
    PARAMETERS = "parameters"  # Parameter block for policy values
    VARIABLE = "variable"
    ENUM = "enum"
    ENTITY = "entity"
    PERIOD = "period"
    DTYPE = "dtype"
    LABEL = "label"
    DESCRIPTION = "description"
    UNIT = "unit"
    FORMULA = "formula"
    DEFINED_FOR = "defined_for"
    DEFAULT = "default"
    PRIVATE = "private"
    INTERNAL = "internal"
    LET = "let"
    RETURN = "return"
    IF = "if"
    THEN = "then"
    ELSE = "else"
    MATCH = "match"
    CASE = "case"
    AND = "and"
    OR = "or"
    NOT = "not"
    TRUE = "true"
    FALSE = "false"

    # Symbols
    LBRACE = "{"
    RBRACE = "}"
    LPAREN = "("
    RPAREN = ")"
    LBRACKET = "["
    RBRACKET = "]"
    COMMA = ","
    COLON = ":"
    DOT = "."
    EQUALS = "="
    ARROW = "=>"
    PLUS = "+"
    MINUS = "-"
    STAR = "*"
    SLASH = "/"
    PERCENT = "%"
    LT = "<"
    GT = ">"
    LE = "<="
    GE = ">="
    EQ = "=="
    NE = "!="
    QUESTION = "?"
    AMPERSAND = "&"  # Alternative for logical and
    PIPE = "|"  # Alternative for logical or
    HASH = "#"  # Fragment identifier in paths

    # Literals
    NUMBER = "NUMBER"
    STRING = "STRING"
    IDENTIFIER = "IDENTIFIER"

    # Special
    EOF = "EOF"
    NEWLINE = "NEWLINE"
    COMMENT = "COMMENT"


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    column: int


@dataclass
class ModuleDecl:
    path: str


@dataclass
class VersionDecl:
    version: str


@dataclass
class JurisdictionDecl:
    jurisdiction: str


@dataclass
class ImportDecl:
    module_path: str
    names: list[str]  # ["*"] for wildcard
    alias: Optional[str] = None


@dataclass
class StatuteReference:
    """A reference mapping a local alias to a statute path.

    Example:
        references {
          earned_income: us/irc/subtitle_a/.../§32/c/2/A/earned_income
          filing_status: us/irc/.../§1/filing_status
        }
    """
    alias: str  # Local name used in formulas
    statute_path: str  # Full statute path (us/irc/.../variable_name)


@dataclass
class ReferencesBlock:
    """Block of statute-path references that alias variables for use in formulas."""
    references: list[StatuteReference] = field(default_factory=list)

    def get_path(self, alias: str) -> Optional[str]:
        """Get the statute path for a given alias."""
        for ref in self.references:
            if ref.alias == alias:
                return ref.statute_path
        return None

    def as_dict(self) -> dict[str, str]:
        """Return references as alias -> path dict."""
        return {ref.alias: ref.statute_path for ref in self.references}


@dataclass
class LetBinding:
    name: str
    value: "Expression"


@dataclass
class VariableRef:
    name: str
    period_offset: Optional[int] = None


@dataclass
class ParameterRef:
    path: str
    index: Optional[str] = None  # For indexed params like rate[n_children]


@dataclass
class BinaryOp:
    op: str
    left: "Expression"
    right: "Expression"


@dataclass
class UnaryOp:
    op: str
    operand: "Expression"


@dataclass
class FunctionCall:
    name: str
    args: list["Expression"]


@dataclass
class IfExpr:
    condition: "Expression"
    then_branch: "Expression"
    else_branch: "Expression"


@dataclass
class MatchCase:
    condition: Optional["Expression"]  # None for else
    value: "Expression"


@dataclass
class MatchExpr:
    match_value: Optional["Expression"]  # Value to match against (None for condition-only)
    cases: list[MatchCase]


@dataclass
class Literal:
    value: Any
    dtype: str  # "number", "string", "bool"


@dataclass
class Identifier:
    name: str


@dataclass
class IndexExpr:
    """Subscript/index expression: base[index]

    Used for parameter lookups like credit_percentage[num_qualifying_children]
    """
    base: "Expression"
    index: "Expression"


Expression = Union[
    'LetBinding', 'VariableRef', 'ParameterRef', 'BinaryOp', 'UnaryOp',
    'FunctionCall', 'IfExpr', 'MatchExpr', 'Literal', 'Identifier', 'IndexExpr'
]


@dataclass
class FormulaBlock:
    bindings: list[LetBinding]
    return_expr: Expression


@dataclass
class VariableDef:
    name: str
    entity: str
    period: str
    dtype: str
    label: Optional[str] = None
    description: Optional[str] = None
    unit: Optional[str] = None
    formula: Optional[FormulaBlock] = None
    defined_for: Optional[Expression] = None
    default: Optional[Any] = None
    visibility: str = "public"  # "public", "private", "internal"


@dataclass
class EnumDef:
    name: str
    values: list[str]


@dataclass
class Module:
    module_decl: Optional[ModuleDecl] = None
    version_decl: Optional[VersionDecl] = None
    jurisdiction_decl: Optional[JurisdictionDecl] = None
    legacy_imports: list[ImportDecl] = field(default_factory=list)  # Old import syntax
    imports: Optional[ReferencesBlock] = None  # imports { } block for vars/params
    variables: list[VariableDef] = field(default_factory=list)
    enums: list[EnumDef] = field(default_factory=list)


class Lexer:
    """Tokenizer for Cosilico DSL."""

    KEYWORDS = {
        "module", "version", "jurisdiction", "import", "imports", "references", "parameters", "variable", "enum",
        "entity", "period", "dtype", "label", "description",
        "unit", "formula", "defined_for", "default", "private", "internal",
        "let", "return", "if", "then", "else", "match", "case",
        "and", "or", "not", "true", "false",
    }

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.line = 1
        self.column = 1
        self.tokens: list[Token] = []

    def tokenize(self) -> list[Token]:
        while self.pos < len(self.source):
            self._skip_whitespace_and_comments()
            if self.pos >= len(self.source):
                break

            ch = self.source[self.pos]

            # String literals
            if ch == '"':
                self._read_string()
            # Numbers
            elif ch.isdigit() or (ch == '-' and self._peek(1).isdigit()):
                self._read_number()
            # Identifiers and keywords (including § for statute section references)
            elif ch.isalpha() or ch == '_' or ch == '§':
                self._read_identifier()
            # Operators and symbols
            else:
                self._read_symbol()

        self.tokens.append(Token(TokenType.EOF, None, self.line, self.column))
        return self.tokens

    def _peek(self, offset: int = 0) -> str:
        pos = self.pos + offset
        if pos < len(self.source):
            return self.source[pos]
        return ""

    def _advance(self) -> str:
        ch = self.source[self.pos]
        self.pos += 1
        if ch == '\n':
            self.line += 1
            self.column = 1
        else:
            self.column += 1
        return ch

    def _skip_whitespace_and_comments(self):
        consumed_whitespace = False
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch in ' \t\r\n':
                consumed_whitespace = True
                self._advance()
            elif ch == '#':
                # # is a comment only if:
                # 1. At start of line (column == 1), OR
                # 2. After whitespace (we just consumed spaces/tabs/newlines)
                # Otherwise it's a token (fragment identifier in paths)
                if self.column == 1 or consumed_whitespace:
                    while self.pos < len(self.source) and self.source[self.pos] != '\n':
                        self._advance()
                    consumed_whitespace = False  # Reset after consuming comment
                else:
                    break  # Not a comment, let tokenizer handle it
            elif ch == '/' and self.pos + 1 < len(self.source) and self.source[self.pos + 1] == '/':
                # Skip to end of line (// style comment)
                while self.pos < len(self.source) and self.source[self.pos] != '\n':
                    self._advance()
            else:
                break

    def _read_string(self):
        start_line, start_col = self.line, self.column
        self._advance()  # Skip opening quote

        value = ""
        while self.pos < len(self.source) and self.source[self.pos] != '"':
            if self.source[self.pos] == '\\':
                self._advance()
                if self.pos < len(self.source):
                    escape_ch = self._advance()
                    if escape_ch == 'n':
                        value += '\n'
                    elif escape_ch == 't':
                        value += '\t'
                    else:
                        value += escape_ch
            else:
                value += self._advance()

        if self.pos < len(self.source):
            self._advance()  # Skip closing quote

        self.tokens.append(Token(TokenType.STRING, value, start_line, start_col))

    def _read_number(self):
        start_line, start_col = self.line, self.column
        value = ""

        if self.source[self.pos] == '-':
            value += self._advance()

        # Read digits
        while self.pos < len(self.source) and self.source[self.pos].isdigit():
            value += self._advance()

        # Read decimal part only if dot is followed by a digit
        # This allows "26.32.a.1" to be lexed as "26" "." "32" "." "a" "." "1"
        # while still allowing "3.14" to be lexed as a single float
        if self.pos < len(self.source) and self.source[self.pos] == '.':
            # Peek ahead to see if there's a digit after the dot
            if self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit():
                value += self._advance()  # Consume the dot
                # Read fractional digits
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    value += self._advance()

        # Check for percentage
        if self.pos < len(self.source) and self.source[self.pos] == '%':
            self._advance()
            num_value = float(value) / 100
        else:
            num_value = float(value) if '.' in value else int(value)

        self.tokens.append(Token(TokenType.NUMBER, num_value, start_line, start_col))

    def _read_identifier(self):
        start_line, start_col = self.line, self.column
        value = ""

        # Allow alphanumeric, underscore, and § (section symbol) in identifiers
        while self.pos < len(self.source) and (
            self.source[self.pos].isalnum() or
            self.source[self.pos] == '_' or
            self.source[self.pos] == '§'
        ):
            value += self._advance()

        # Check if keyword
        if value in self.KEYWORDS:
            token_type = TokenType[value.upper()]
        else:
            token_type = TokenType.IDENTIFIER

        self.tokens.append(Token(token_type, value, start_line, start_col))

    def _read_symbol(self):
        start_line, start_col = self.line, self.column
        ch = self._advance()

        # Two-character operators
        if ch == '=' and self._peek() == '>':
            self._advance()
            self.tokens.append(Token(TokenType.ARROW, "=>", start_line, start_col))
        elif ch == '=' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.EQ, "==", start_line, start_col))
        elif ch == '!' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.NE, "!=", start_line, start_col))
        elif ch == '<' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.LE, "<=", start_line, start_col))
        elif ch == '>' and self._peek() == '=':
            self._advance()
            self.tokens.append(Token(TokenType.GE, ">=", start_line, start_col))
        # Single-character operators
        elif ch == '{':
            self.tokens.append(Token(TokenType.LBRACE, ch, start_line, start_col))
        elif ch == '}':
            self.tokens.append(Token(TokenType.RBRACE, ch, start_line, start_col))
        elif ch == '(':
            self.tokens.append(Token(TokenType.LPAREN, ch, start_line, start_col))
        elif ch == ')':
            self.tokens.append(Token(TokenType.RPAREN, ch, start_line, start_col))
        elif ch == '[':
            self.tokens.append(Token(TokenType.LBRACKET, ch, start_line, start_col))
        elif ch == ']':
            self.tokens.append(Token(TokenType.RBRACKET, ch, start_line, start_col))
        elif ch == ',':
            self.tokens.append(Token(TokenType.COMMA, ch, start_line, start_col))
        elif ch == ':':
            self.tokens.append(Token(TokenType.COLON, ch, start_line, start_col))
        elif ch == '.':
            self.tokens.append(Token(TokenType.DOT, ch, start_line, start_col))
        elif ch == '=':
            self.tokens.append(Token(TokenType.EQUALS, ch, start_line, start_col))
        elif ch == '+':
            self.tokens.append(Token(TokenType.PLUS, ch, start_line, start_col))
        elif ch == '-':
            self.tokens.append(Token(TokenType.MINUS, ch, start_line, start_col))
        elif ch == '*':
            self.tokens.append(Token(TokenType.STAR, ch, start_line, start_col))
        elif ch == '/':
            self.tokens.append(Token(TokenType.SLASH, ch, start_line, start_col))
        elif ch == '%':
            self.tokens.append(Token(TokenType.PERCENT, ch, start_line, start_col))
        elif ch == '<':
            self.tokens.append(Token(TokenType.LT, ch, start_line, start_col))
        elif ch == '>':
            self.tokens.append(Token(TokenType.GT, ch, start_line, start_col))
        elif ch == '?':
            self.tokens.append(Token(TokenType.QUESTION, ch, start_line, start_col))
        elif ch == '&':
            self.tokens.append(Token(TokenType.AMPERSAND, ch, start_line, start_col))
        elif ch == '|':
            self.tokens.append(Token(TokenType.PIPE, ch, start_line, start_col))
        elif ch == '#':
            self.tokens.append(Token(TokenType.HASH, ch, start_line, start_col))
        else:
            raise SyntaxError(f"Unexpected character '{ch}' at line {start_line}, column {start_col}")


class Parser:
    """Recursive descent parser for Cosilico DSL."""

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.pos = 0

    def parse(self) -> Module:
        module = Module()

        while not self._is_at_end():
            if self._check(TokenType.MODULE):
                module.module_decl = self._parse_module_decl()
            elif self._check(TokenType.VERSION):
                module.version_decl = self._parse_version_decl()
            elif self._check(TokenType.JURISDICTION):
                module.jurisdiction_decl = self._parse_jurisdiction_decl()
            elif self._check(TokenType.IMPORT):
                module.legacy_imports.append(self._parse_import())
            elif self._check(TokenType.IMPORTS) or self._check(TokenType.REFERENCES):
                module.imports = self._parse_imports_block()
            elif self._check(TokenType.PARAMETERS):
                # Skip parameters block for now - parameters are loaded separately
                self._skip_parameters_block()
            elif self._check(TokenType.PRIVATE) or self._check(TokenType.INTERNAL):
                visibility = self._advance().value
                if self._check(TokenType.VARIABLE):
                    var = self._parse_variable()
                    var.visibility = visibility
                    module.variables.append(var)
            elif self._check(TokenType.VARIABLE):
                module.variables.append(self._parse_variable())
            elif self._check(TokenType.ENUM):
                module.enums.append(self._parse_enum())
            elif self._check(TokenType.ENTITY):
                # Inline variable format (file-is-the-variable)
                # entity TaxUnit
                # period Year
                # dtype Money
                # formula:
                #   ...
                inline_var = self._parse_inline_variable()
                if inline_var:
                    module.variables.append(inline_var)
            else:
                # Skip unexpected tokens
                self._advance()

        return module

    def _is_at_end(self) -> bool:
        return self._peek().type == TokenType.EOF

    def _peek(self) -> Token:
        return self.tokens[self.pos]

    def _previous(self) -> Token:
        return self.tokens[self.pos - 1]

    def _check(self, token_type: TokenType) -> bool:
        if self._is_at_end():
            return False
        return self._peek().type == token_type

    def _peek_next_is(self, token_type: TokenType) -> bool:
        """Check if the next token (after current) is of given type."""
        if self.pos + 1 >= len(self.tokens):
            return False
        return self.tokens[self.pos + 1].type == token_type

    def _advance(self) -> Token:
        if not self._is_at_end():
            self.pos += 1
        return self._previous()

    def _consume(self, token_type: TokenType, message: str) -> Token:
        if self._check(token_type):
            return self._advance()
        raise SyntaxError(f"{message} at line {self._peek().line}")

    def _parse_module_decl(self) -> ModuleDecl:
        self._consume(TokenType.MODULE, "Expected 'module'")
        path = self._parse_dotted_name()
        return ModuleDecl(path=path)

    def _parse_version_decl(self) -> VersionDecl:
        self._consume(TokenType.VERSION, "Expected 'version'")
        version = self._consume(TokenType.STRING, "Expected version string").value
        return VersionDecl(version=version)

    def _parse_jurisdiction_decl(self) -> JurisdictionDecl:
        self._consume(TokenType.JURISDICTION, "Expected 'jurisdiction'")
        jurisdiction = self._consume(TokenType.IDENTIFIER, "Expected jurisdiction name").value
        return JurisdictionDecl(jurisdiction=jurisdiction)

    def _parse_import(self) -> ImportDecl:
        self._consume(TokenType.IMPORT, "Expected 'import'")
        module_path = self._parse_dotted_name()

        self._consume(TokenType.LPAREN, "Expected '('")

        names = []
        if self._check(TokenType.STAR):
            self._advance()
            names = ["*"]
        else:
            names.append(self._consume(TokenType.IDENTIFIER, "Expected identifier").value)
            while self._check(TokenType.COMMA):
                self._advance()
                names.append(self._consume(TokenType.IDENTIFIER, "Expected identifier").value)

        self._consume(TokenType.RPAREN, "Expected ')'")

        alias = None
        # Check for 'as alias' pattern

        return ImportDecl(module_path=module_path, names=names, alias=alias)

    def _skip_parameters_block(self):
        """Skip over parameters block - these are loaded separately from YAML files."""
        self._consume(TokenType.PARAMETERS, "Expected 'parameters'")
        self._consume(TokenType.COLON, "Expected ':' after parameters")

        # Skip until we hit a top-level keyword at column 1
        # Keywords in paths (like 1001/parameters#key) should be skipped
        top_level_keywords = {
            TokenType.ENTITY, TokenType.PERIOD, TokenType.DTYPE,
            TokenType.LABEL, TokenType.DESCRIPTION, TokenType.UNIT,
            TokenType.FORMULA, TokenType.DEFINED_FOR, TokenType.DEFAULT,
            TokenType.VARIABLE, TokenType.ENUM, TokenType.PRIVATE,
            TokenType.INTERNAL, TokenType.MODULE, TokenType.VERSION,
            TokenType.IMPORTS, TokenType.REFERENCES, TokenType.PARAMETERS,
        }

        while not self._is_at_end():
            # Only consider keywords at column 1 as block terminators
            if self._peek().column == 1 and any(self._check(kw) for kw in top_level_keywords):
                break
            self._advance()

    def _parse_imports_block(self) -> ReferencesBlock:
        """Parse an imports block mapping aliases to file paths.

        Syntax (YAML-like with colon):
            imports:
              earned_income: statute/26/32/c/2/A/earned_income
              filing_status: statute/26/1/filing_status

        Also accepts 'references' for backwards compatibility.
        Block ends when we hit a top-level keyword (entity, period, formula, etc.)
        """
        # Accept either 'imports' or 'references' keyword
        if self._check(TokenType.IMPORTS):
            self._advance()
        else:
            self._consume(TokenType.REFERENCES, "Expected 'imports' or 'references'")

        # Expect colon after imports keyword
        self._consume(TokenType.COLON, "Expected ':' after imports")

        references = []

        # Parse references until we hit a top-level keyword
        top_level_keywords = {
            TokenType.ENTITY, TokenType.PERIOD, TokenType.DTYPE,
            TokenType.LABEL, TokenType.DESCRIPTION, TokenType.UNIT,
            TokenType.FORMULA, TokenType.DEFINED_FOR, TokenType.DEFAULT,
            TokenType.VARIABLE, TokenType.ENUM, TokenType.PRIVATE,
            TokenType.INTERNAL, TokenType.MODULE, TokenType.VERSION,
            TokenType.IMPORTS, TokenType.REFERENCES, TokenType.PARAMETERS,
        }

        while not self._is_at_end():
            # Check if we've hit a top-level keyword (end of imports block)
            if any(self._check(kw) for kw in top_level_keywords):
                break

            # Skip comments (already handled by lexer, but check for empty lines)
            if self._check(TokenType.EOF):
                break

            # Parse alias name
            alias = self._consume(TokenType.IDENTIFIER, "Expected alias name").value

            self._consume(TokenType.COLON, "Expected ':'")

            # Parse statute path (can contain /, §, and other characters)
            statute_path = self._parse_statute_path()

            references.append(StatuteReference(alias=alias, statute_path=statute_path))

        return ReferencesBlock(references=references)

    def _parse_statute_path(self) -> str:
        """Parse a statute path like 'us/irc/subtitle_a/.../§32/c/2/A/variable_name'."""
        # Consume tokens until we hit something that's not part of a path
        # Path components: identifiers, numbers, §, /, .
        # Keywords like 'parameters', 'imports' can also appear in paths
        parts = []

        # Keywords that can appear in paths as identifiers
        path_keywords = {
            TokenType.PARAMETERS, TokenType.IMPORTS, TokenType.REFERENCES,
            TokenType.ENTITY, TokenType.PERIOD, TokenType.DTYPE,
            TokenType.VARIABLE, TokenType.FORMULA,
        }

        # First component must be identifier or path-allowed keyword
        if self._check(TokenType.IDENTIFIER):
            parts.append(self._advance().value)
        elif any(self._check(kw) for kw in path_keywords):
            parts.append(self._advance().value)
        else:
            parts.append(self._consume(TokenType.IDENTIFIER, "Expected path component").value)

        while True:
            if self._check(TokenType.SLASH):
                self._advance()
                parts.append("/")
                # Next can be identifier, number, keyword, or special chars
                if self._check(TokenType.IDENTIFIER):
                    parts.append(self._advance().value)
                elif self._check(TokenType.NUMBER):
                    parts.append(str(self._advance().value))
                elif any(self._check(kw) for kw in path_keywords):
                    parts.append(self._advance().value)
                else:
                    break
            elif self._check(TokenType.DOT):
                # Could be .. in path
                self._advance()
                parts.append(".")
                if self._check(TokenType.DOT):
                    self._advance()
                    parts.append(".")
            elif self._check(TokenType.HASH):
                # Fragment identifier like path#fragment
                self._advance()
                parts.append("#")
                # Fragment name must be identifier or keyword
                if self._check(TokenType.IDENTIFIER):
                    parts.append(self._advance().value)
                elif any(self._check(kw) for kw in path_keywords):
                    parts.append(self._advance().value)
                # Fragment is the end of the path
                break
            else:
                break

        return "".join(parts)

    def _parse_dotted_name(self) -> str:
        """Parse a dotted name that can contain identifiers or numbers.

        Examples:
            - gov.irs.eitc (identifiers only)
            - statute.26.32.a.1 (mixed identifiers and numbers)

        Note: The lexer may read "26.32" as a float. In dotted name context,
        we keep it as "26.32" which is the correct representation for statute paths.
        """
        # First component must be an identifier
        name = self._consume(TokenType.IDENTIFIER, "Expected identifier").value

        while self._check(TokenType.DOT):
            self._advance()
            # After a dot, we can have either an identifier or a number
            if self._check(TokenType.IDENTIFIER):
                name += "." + self._advance().value
            elif self._check(TokenType.NUMBER):
                # Convert number to string for the path
                num_value = self._advance().value
                # Handle both int and float
                if isinstance(num_value, int):
                    name += "." + str(num_value)
                elif isinstance(num_value, float):
                    # Check if it's actually an integer value stored as float
                    if num_value == int(num_value):
                        name += "." + str(int(num_value))
                    else:
                        # It's a true float (e.g., 26.32 in statute.26.32.a)
                        # Keep the float representation
                        name += "." + str(num_value)
                else:
                    name += "." + str(num_value)
            else:
                raise SyntaxError(f"Expected identifier or number after '.' at line {self._peek().line}")

        return name

    def _parse_inline_variable(self) -> Optional[VariableDef]:
        """Parse inline variable format (file-is-the-variable).

        Syntax (YAML-like):
            entity TaxUnit
            period Year
            dtype Money
            unit "USD"
            label "..."
            description "..."

            formula:
              let x = ...
              return ...

        Returns a VariableDef with name="inline" (to be set by caller based on filename).
        """
        var = VariableDef(
            name="inline",  # Caller should set based on filename
            entity="",
            period="",
            dtype="",
        )

        # Top-level keywords that end metadata and start formula
        formula_start = {TokenType.FORMULA}

        while not self._is_at_end():
            if self._check(TokenType.ENTITY):
                self._advance()
                var.entity = self._consume(TokenType.IDENTIFIER, "Expected entity type").value
            elif self._check(TokenType.PERIOD):
                self._advance()
                var.period = self._consume(TokenType.IDENTIFIER, "Expected period type").value
            elif self._check(TokenType.DTYPE):
                self._advance()
                var.dtype = self._parse_dtype()
            elif self._check(TokenType.LABEL):
                self._advance()
                var.label = self._consume(TokenType.STRING, "Expected label string").value
            elif self._check(TokenType.DESCRIPTION):
                self._advance()
                var.description = self._consume(TokenType.STRING, "Expected description string").value
            elif self._check(TokenType.UNIT):
                self._advance()
                var.unit = self._consume(TokenType.STRING, "Expected unit string").value
            elif self._check(TokenType.DEFAULT):
                self._advance()
                var.default = self._parse_literal_value()
            elif self._check(TokenType.FORMULA):
                self._advance()
                # Expect colon after formula keyword
                self._consume(TokenType.COLON, "Expected ':' after formula")
                # Parse formula block (no braces in YAML-like syntax)
                var.formula = self._parse_inline_formula_block()
                break  # Formula is last, stop parsing
            elif self._check(TokenType.DEFINED_FOR):
                self._advance()
                # defined_for can use colon syntax too
                if self._check(TokenType.COLON):
                    self._advance()
                elif self._check(TokenType.LBRACE):
                    self._advance()
                var.defined_for = self._parse_expression()
                if self._check(TokenType.RBRACE):
                    self._advance()
            else:
                # Unknown token - stop parsing this variable
                break

        return var

    def _parse_inline_formula_block(self) -> FormulaBlock:
        """Parse formula block in YAML-like syntax (no braces).

        Syntax:
            formula:
              let earned = wages + salaries
              return earned * rate

        Ends at EOF or next top-level keyword.
        """
        bindings = []
        return_expr = None

        # Keywords that end the formula block
        end_keywords = {
            TokenType.ENTITY, TokenType.PERIOD, TokenType.DTYPE,
            TokenType.LABEL, TokenType.DESCRIPTION, TokenType.UNIT,
            TokenType.FORMULA, TokenType.VARIABLE, TokenType.ENUM,
            TokenType.MODULE, TokenType.VERSION, TokenType.IMPORTS,
            TokenType.REFERENCES, TokenType.PARAMETERS,
        }

        while not self._is_at_end():
            # Check if we've hit a top-level keyword (end of formula)
            if any(self._check(kw) for kw in end_keywords):
                break

            if self._check(TokenType.LET):
                bindings.append(self._parse_let_binding())
            elif self._check(TokenType.RETURN):
                self._advance()
                return_expr = self._parse_expression()
                break
            elif self._check(TokenType.IF):
                # Statement-level if: "if condition then \n return value"
                # This is an early-exit pattern, not an if-expression
                if_expr = self._parse_statement_if()
                if if_expr is not None:
                    # Early return with condition - wrap as conditional return
                    # Create a conditional binding that evaluates to the return value
                    # and continue parsing
                    # For now, we handle this as a conditional with early-exit semantics
                    # by wrapping remaining formula in the else branch
                    early_return_condition = if_expr.condition
                    early_return_value = if_expr.then_branch
                    # Parse rest of formula as the "else" case
                    rest_bindings, rest_expr = self._parse_rest_of_formula(end_keywords)
                    # Wrap in conditional: if condition then early_return else rest
                    return_expr = IfExpr(
                        condition=early_return_condition,
                        then_branch=early_return_value,
                        else_branch=self._wrap_bindings_as_expr(rest_bindings, rest_expr)
                    )
                    break
            elif self._check(TokenType.IDENTIFIER):
                # Could be assignment: name = expr
                if self._peek_next_is(TokenType.EQUALS):
                    # Parse as let binding without 'let' keyword
                    name = self._advance().value
                    self._consume(TokenType.EQUALS, "Expected '='")
                    value = self._parse_expression()
                    bindings.append(LetBinding(name=name, value=value))
                else:
                    # Expression - treat as return
                    return_expr = self._parse_expression()
                    break
            else:
                # Unknown - treat as return expression
                return_expr = self._parse_expression()
                break

        return FormulaBlock(bindings=bindings, return_expr=return_expr)

    def _parse_statement_if(self) -> Optional[IfExpr]:
        """Parse statement-level if: 'if condition then' followed by 'return value'.

        Returns an IfExpr with condition and then_branch (the return value),
        or None if this isn't a statement-level if.
        """
        self._consume(TokenType.IF, "Expected 'if'")
        condition = self._parse_expression()
        self._consume(TokenType.THEN, "Expected 'then'")

        # Check if next token is RETURN (statement-level if)
        if self._check(TokenType.RETURN):
            self._advance()  # consume 'return'
            then_value = self._parse_expression()
            return IfExpr(condition=condition, then_branch=then_value, else_branch=Literal(value=0, dtype="number"))
        else:
            # This is an expression-level if, parse as normal
            then_branch = self._parse_expression()
            self._consume(TokenType.ELSE, "Expected 'else'")
            else_branch = self._parse_expression()
            return IfExpr(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def _parse_rest_of_formula(self, end_keywords: set) -> tuple[list[LetBinding], Optional[Expression]]:
        """Parse remaining bindings and return expression after an early-exit if."""
        bindings = []
        return_expr = None

        while not self._is_at_end():
            if any(self._check(kw) for kw in end_keywords):
                break

            if self._check(TokenType.LET):
                bindings.append(self._parse_let_binding())
            elif self._check(TokenType.RETURN):
                self._advance()
                return_expr = self._parse_expression()
                break
            elif self._check(TokenType.IDENTIFIER):
                if self._peek_next_is(TokenType.EQUALS):
                    name = self._advance().value
                    self._consume(TokenType.EQUALS, "Expected '='")
                    value = self._parse_expression()
                    bindings.append(LetBinding(name=name, value=value))
                else:
                    return_expr = self._parse_expression()
                    break
            else:
                return_expr = self._parse_expression()
                break

        return bindings, return_expr

    def _wrap_bindings_as_expr(self, bindings: list[LetBinding], return_expr: Optional[Expression]) -> Expression:
        """Wrap a list of bindings and a return expression as a single expression.

        For execution, we need to represent let bindings + return as a single expression.
        We use a nested structure of function applications to simulate let bindings.
        """
        if not bindings:
            return return_expr if return_expr else Literal(value=0, dtype="number")

        # For now, if there are bindings, create a LetExpr-like structure
        # by using the last binding's value modified to include the return
        # This is a simplification - ideally we'd have proper let-in expressions
        # For the standard_deduction case: basic = ..., return basic + additional
        # We wrap as: (let basic = ... in basic + additional)

        # Create a simplified representation using FormulaBlock
        # The executor will need to handle this appropriately
        return FormulaBlock(bindings=bindings, return_expr=return_expr)

    def _parse_variable(self) -> VariableDef:
        self._consume(TokenType.VARIABLE, "Expected 'variable'")
        name = self._consume(TokenType.IDENTIFIER, "Expected variable name").value
        self._consume(TokenType.LBRACE, "Expected '{'")

        var = VariableDef(
            name=name,
            entity="",
            period="",
            dtype="",
        )

        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.ENTITY):
                self._advance()
                var.entity = self._consume(TokenType.IDENTIFIER, "Expected entity type").value
            elif self._check(TokenType.PERIOD):
                self._advance()
                var.period = self._consume(TokenType.IDENTIFIER, "Expected period type").value
            elif self._check(TokenType.DTYPE):
                self._advance()
                var.dtype = self._parse_dtype()
            elif self._check(TokenType.LABEL):
                self._advance()
                var.label = self._consume(TokenType.STRING, "Expected label string").value
            elif self._check(TokenType.DESCRIPTION):
                self._advance()
                var.description = self._consume(TokenType.STRING, "Expected description string").value
            elif self._check(TokenType.UNIT):
                self._advance()
                var.unit = self._consume(TokenType.STRING, "Expected unit string").value
            elif self._check(TokenType.FORMULA):
                self._advance()
                self._consume(TokenType.LBRACE, "Expected '{'")
                var.formula = self._parse_formula_block()
                self._consume(TokenType.RBRACE, "Expected '}'")
            elif self._check(TokenType.DEFINED_FOR):
                self._advance()
                self._consume(TokenType.LBRACE, "Expected '{'")
                var.defined_for = self._parse_expression()
                self._consume(TokenType.RBRACE, "Expected '}'")
            elif self._check(TokenType.DEFAULT):
                self._advance()
                var.default = self._parse_literal_value()
            else:
                # Strict mode: error on unknown fields
                token = self._peek()
                raise SyntaxError(
                    f"Unknown field '{token.value}' in variable definition at line {token.line}. "
                    f"Valid fields: entity, period, dtype, label, description, unit, formula, defined_for, default"
                )

        self._consume(TokenType.RBRACE, "Expected '}'")
        return var

    def _parse_dtype(self) -> str:
        dtype = self._consume(TokenType.IDENTIFIER, "Expected data type").value
        # Handle parameterized types like Enum(T)
        if self._check(TokenType.LPAREN):
            self._advance()
            inner = self._consume(TokenType.IDENTIFIER, "Expected type parameter").value
            self._consume(TokenType.RPAREN, "Expected ')'")
            dtype = f"{dtype}({inner})"
        return dtype

    def _parse_enum(self) -> EnumDef:
        self._consume(TokenType.ENUM, "Expected 'enum'")
        name = self._consume(TokenType.IDENTIFIER, "Expected enum name").value
        self._consume(TokenType.LBRACE, "Expected '{'")

        values = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            values.append(self._consume(TokenType.IDENTIFIER, "Expected enum value").value)

        self._consume(TokenType.RBRACE, "Expected '}'")
        return EnumDef(name=name, values=values)

    def _parse_formula_block(self) -> FormulaBlock:
        bindings = []
        guards = []  # List of (condition, return_value) tuples for if-guards
        return_expr = None

        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.LET):
                bindings.append(self._parse_let_binding())
            elif self._check(TokenType.IF):
                # Check if this is an if-guard statement (if ... then return ...)
                # or an if-expression (if ... then expr else expr)
                guard = self._try_parse_if_guard()
                if guard:
                    guards.append(guard)
                else:
                    # It's an if-expression, parse as return expression
                    return_expr = self._parse_expression()
                    break
            elif self._check(TokenType.RETURN):
                self._advance()
                return_expr = self._parse_expression()
                break
            else:
                # Implicit return - expression without 'return' keyword
                return_expr = self._parse_expression()
                break

        # Build nested if-else from guards and final return
        if guards and return_expr:
            # Transform guards into nested if-else expression
            # guards = [(cond1, val1), (cond2, val2)]
            # return_expr = final_val
            # Result: if cond1 then val1 else if cond2 then val2 else final_val
            result = return_expr
            for condition, guard_value in reversed(guards):
                result = IfExpr(condition=condition, then_branch=guard_value, else_branch=result)
            return_expr = result

        return FormulaBlock(bindings=bindings, return_expr=return_expr)

    def _try_parse_if_guard(self) -> tuple | None:
        """Try to parse an if-guard statement: if <cond> then return <expr>

        Returns (condition, return_value) tuple if successful, None if this is
        a regular if-expression that should be parsed differently.
        """
        # Save position to backtrack if this isn't a guard
        saved_pos = self.pos

        self._advance()  # consume 'if'
        condition = self._parse_expression()

        if not self._check(TokenType.THEN):
            # Not a valid if, backtrack
            self.pos = saved_pos
            return None

        self._advance()  # consume 'then'

        # Check if next token is 'return' - that makes this a guard
        if self._check(TokenType.RETURN):
            self._advance()  # consume 'return'
            return_value = self._parse_expression()
            return (condition, return_value)

        # Not a guard, backtrack and let caller parse as expression
        self.pos = saved_pos
        return None

    def _parse_let_binding(self) -> LetBinding:
        self._consume(TokenType.LET, "Expected 'let'")
        name = self._consume(TokenType.IDENTIFIER, "Expected variable name").value
        self._consume(TokenType.EQUALS, "Expected '='")
        value = self._parse_expression()
        return LetBinding(name=name, value=value)

    def _parse_expression(self) -> Expression:
        return self._parse_ternary()

    def _parse_ternary(self) -> Expression:
        """Parse ternary operator: condition ? then_value : else_value"""
        condition = self._parse_or_expr()

        if self._check(TokenType.QUESTION):
            self._advance()  # consume '?'
            then_branch = self._parse_expression()
            self._consume(TokenType.COLON, "Expected ':' in ternary expression")
            else_branch = self._parse_expression()
            return IfExpr(condition=condition, then_branch=then_branch, else_branch=else_branch)

        return condition

    def _parse_or_expr(self) -> Expression:
        left = self._parse_and_expr()

        while self._check(TokenType.OR) or self._check(TokenType.PIPE):
            self._advance()
            right = self._parse_and_expr()
            left = BinaryOp(op="or", left=left, right=right)

        return left

    def _parse_and_expr(self) -> Expression:
        left = self._parse_comparison()

        while self._check(TokenType.AND) or self._check(TokenType.AMPERSAND):
            self._advance()
            right = self._parse_comparison()
            left = BinaryOp(op="and", left=left, right=right)

        return left

    def _parse_comparison(self) -> Expression:
        left = self._parse_additive()

        while self._check(TokenType.EQ) or self._check(TokenType.NE) or \
              self._check(TokenType.LT) or self._check(TokenType.GT) or \
              self._check(TokenType.LE) or self._check(TokenType.GE):
            op = self._advance().value
            right = self._parse_additive()
            left = BinaryOp(op=op, left=left, right=right)

        return left

    def _parse_additive(self) -> Expression:
        left = self._parse_multiplicative()

        while self._check(TokenType.PLUS) or self._check(TokenType.MINUS):
            op = self._advance().value
            right = self._parse_multiplicative()
            left = BinaryOp(op=op, left=left, right=right)

        return left

    def _parse_multiplicative(self) -> Expression:
        left = self._parse_unary()

        while self._check(TokenType.STAR) or self._check(TokenType.SLASH) or self._check(TokenType.PERCENT):
            op = self._advance().value
            right = self._parse_unary()
            left = BinaryOp(op=op, left=left, right=right)

        return left

    def _parse_unary(self) -> Expression:
        if self._check(TokenType.MINUS) or self._check(TokenType.NOT):
            op = self._advance().value
            operand = self._parse_unary()
            return UnaryOp(op=op, operand=operand)

        return self._parse_primary()

    def _parse_primary(self) -> Expression:
        # If expression
        if self._check(TokenType.IF):
            return self._parse_if_expr()

        # Match expression
        if self._check(TokenType.MATCH):
            return self._parse_match_expr()

        # Parenthesized expression
        if self._check(TokenType.LPAREN):
            self._advance()
            expr = self._parse_expression()
            self._consume(TokenType.RPAREN, "Expected ')'")
            return expr

        # Literals
        if self._check(TokenType.NUMBER):
            value = self._advance().value
            return Literal(value=value, dtype="number")

        if self._check(TokenType.STRING):
            value = self._advance().value
            return Literal(value=value, dtype="string")

        if self._check(TokenType.TRUE):
            self._advance()
            return Literal(value=True, dtype="bool")

        if self._check(TokenType.FALSE):
            self._advance()
            return Literal(value=False, dtype="bool")

        # Special handling for variable() and parameter() as function calls
        # These are keywords but can also be used as function names
        if self._check(TokenType.VARIABLE) and self._peek_next_is(TokenType.LPAREN):
            name = self._advance().value  # "variable"
            return self._parse_function_call(name)

        # Function calls or identifiers
        if self._check(TokenType.IDENTIFIER):
            name = self._advance().value

            # Check for function call
            if self._check(TokenType.LPAREN):
                return self._parse_function_call(name)

            # Check for dotted access (e.g., parameter path)
            while self._check(TokenType.DOT):
                self._advance()
                name += "." + self._consume(TokenType.IDENTIFIER, "Expected identifier").value

                # Check for method call
                if self._check(TokenType.LPAREN):
                    return self._parse_function_call(name)

            # Check for indexing: base[index]
            if self._check(TokenType.LBRACKET):
                self._advance()
                index = self._parse_expression()
                self._consume(TokenType.RBRACKET, "Expected ']'")
                # Return IndexExpr so base is evaluated as variable/expression first
                return IndexExpr(base=Identifier(name=name), index=index)

            return Identifier(name=name)

        raise SyntaxError(f"Unexpected token {self._peek().type} at line {self._peek().line}")

    def _parse_function_call(self, name: str) -> Expression:
        self._consume(TokenType.LPAREN, "Expected '('")

        args = []
        if not self._check(TokenType.RPAREN):
            args.append(self._parse_expression())
            while self._check(TokenType.COMMA):
                self._advance()
                args.append(self._parse_expression())

        self._consume(TokenType.RPAREN, "Expected ')'")

        # Special handling for variable() and parameter()
        if name == "variable":
            if args and isinstance(args[0], Identifier):
                return VariableRef(name=args[0].name)
            elif args and isinstance(args[0], Literal):
                return VariableRef(name=str(args[0].value))

        if name == "parameter":
            if args and isinstance(args[0], Identifier):
                return ParameterRef(path=args[0].name)
            elif args and isinstance(args[0], Literal):
                return ParameterRef(path=str(args[0].value))

        return FunctionCall(name=name, args=args)

    def _parse_if_expr(self) -> IfExpr:
        self._consume(TokenType.IF, "Expected 'if'")
        condition = self._parse_expression()
        self._consume(TokenType.THEN, "Expected 'then'")
        then_branch = self._parse_expression()
        self._consume(TokenType.ELSE, "Expected 'else'")
        else_branch = self._parse_expression()
        return IfExpr(condition=condition, then_branch=then_branch, else_branch=else_branch)

    def _parse_match_expr(self) -> MatchExpr:
        self._consume(TokenType.MATCH, "Expected 'match'")

        # Check if matching on a value (match x { ... }) or conditions (match { ... })
        match_value = None
        if not self._check(TokenType.LBRACE):
            match_value = self._parse_expression()

        self._consume(TokenType.LBRACE, "Expected '{'")

        cases = []
        while not self._check(TokenType.RBRACE) and not self._is_at_end():
            if self._check(TokenType.CASE):
                self._advance()
                condition = self._parse_expression()
                self._consume(TokenType.ARROW, "Expected '=>'")
                value = self._parse_expression()
                cases.append(MatchCase(condition=condition, value=value))
            elif self._check(TokenType.ELSE):
                self._advance()
                self._consume(TokenType.ARROW, "Expected '=>'")
                value = self._parse_expression()
                cases.append(MatchCase(condition=None, value=value))
            else:
                break

        self._consume(TokenType.RBRACE, "Expected '}'")
        return MatchExpr(match_value=match_value, cases=cases)

    def _parse_literal_value(self) -> Any:
        if self._check(TokenType.NUMBER):
            return self._advance().value
        if self._check(TokenType.STRING):
            return self._advance().value
        if self._check(TokenType.TRUE):
            self._advance()
            return True
        if self._check(TokenType.FALSE):
            self._advance()
            return False
        if self._check(TokenType.IDENTIFIER):
            return self._advance().value
        return None


def parse_dsl(source: str) -> Module:
    """Parse Cosilico DSL source code into an AST."""
    lexer = Lexer(source)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    return parser.parse()


def parse_file(filepath: str) -> Module:
    """Parse a .cosilico file."""
    with open(filepath, 'r') as f:
        source = f.read()
    return parse_dsl(source)
