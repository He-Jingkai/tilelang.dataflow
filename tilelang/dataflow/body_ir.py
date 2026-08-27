"""Structured body IR for Dataflow operator-derived PrimFunc lowering."""

from __future__ import annotations

import ast
import collections.abc
from dataclasses import dataclass, field, replace
import inspect
import re
from types import MappingProxyType
import textwrap
from typing import Any, get_args, get_origin
from collections.abc import Mapping

from .ir import (
    IntermediateType,
    DataflowReducerContract,
    OperatorCall,
    get_intermediate_type,
)
from .range_coarsening import DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION


_RANGE_TILES_PER_HANDLER_CALLS = frozenset(
    {
        "T.dataflow_range_tiles_per_handler",
        "dataflow_range_tiles_per_handler",
    }
)


class DataflowBodyIRLoweringError(NotImplementedError):
    """Raised when a Dataflow operator body cannot be represented as body IR."""


@dataclass(frozen=True)
class ExprIR:
    """Base class for Dataflow body expressions."""


@dataclass(frozen=True)
class StmtIR:
    """Base class for Dataflow body statements."""


@dataclass(frozen=True)
class ScalarVarIR(ExprIR):
    name: str


@dataclass(frozen=True)
class LiteralIR(ExprIR):
    value: Any
    dtype: str | None = None


@dataclass(frozen=True)
class CastIR(ExprIR):
    dtype: str
    value: ExprIR


@dataclass(frozen=True)
class TensorLoadIR(ExprIR):
    tensor_name: str
    indices: tuple[ExprIR, ...]


@dataclass(frozen=True)
class SliceIR(ExprIR):
    start: ExprIR | None
    stop: ExprIR | None


@dataclass(frozen=True)
class FieldAccessIR(ExprIR):
    base: ExprIR
    field_name: str


@dataclass(frozen=True)
class FieldElementAccessIR(ExprIR):
    base: ExprIR
    field_name: str
    indices: tuple[int, ...]
    flat_index: int


@dataclass(frozen=True)
class TupleExprIR(ExprIR):
    values: tuple[ExprIR, ...]


@dataclass(frozen=True)
class BinaryOpIR(ExprIR):
    op: str
    left: ExprIR
    right: ExprIR


@dataclass(frozen=True)
class CallIR(ExprIR):
    name: str
    args: tuple[ExprIR, ...]


@dataclass(frozen=True)
class AssignIR(StmtIR):
    target: ExprIR
    value: ExprIR


@dataclass(frozen=True)
class AugAssignIR(StmtIR):
    target: ExprIR
    op: str
    value: ExprIR


@dataclass(frozen=True)
class TensorStoreIR(StmtIR):
    tensor_name: str
    index: ExprIR
    value: ExprIR


@dataclass(frozen=True)
class DataflowAccumulatorIR:
    name: str
    dtype: str
    init: ExprIR


@dataclass(frozen=True)
class DataflowLoopIR(StmtIR):
    var: ScalarVarIR
    iterable_kind: str
    body: tuple[StmtIR, ...]


@dataclass(frozen=True)
class DataflowBodyIR:
    operator_kind: str
    operator_name: str
    accumulators: tuple[DataflowAccumulatorIR, ...] = ()
    loop: DataflowLoopIR | None = None
    returns: Mapping[str, ExprIR] = field(default_factory=dict)
    stores: tuple[TensorStoreIR, ...] = ()
    tilelang_body: tuple[str, ...] = ()
    reducer_contract: DataflowReducerContract | None = None
    reducer_input_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "returns", MappingProxyType(dict(self.returns)))


_BIN_OPS: dict[type[ast.operator], str] = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.Div: "/",
    ast.FloorDiv: "//",
}


def lower_operator_call_to_body_ir(
    call: OperatorCall,
    operator_kind: str,
    *,
    specialization_constants: Mapping[str, Any] | None = None,
) -> DataflowBodyIR:
    """Parse a supported Dataflow operator call body into structured body IR."""

    function, source = function_ast(call, operator_kind)
    context = f"primfunc handler lowering {operator_kind} operator {call.name!r}"
    if operator_kind == "iter":
        return lower_iter(
            function,
            source,
            call,
            context,
            specialization_constants=specialization_constants,
        )
    if operator_kind == "map":
        return replace(
            lower_iter(
                function,
                source,
                call,
                context,
                specialization_constants=specialization_constants,
            ),
            operator_kind="map",
        )
    if operator_kind == "reduce":
        return lower_reduce(function, source, call, context)
    if operator_kind == "finalize":
        return lower_finalize(function, source, call, context)
    raise error(context, f"unsupported operator kind {operator_kind!r}")


def function_ast(call: OperatorCall, operator_kind: str) -> tuple[ast.FunctionDef, str]:
    context = f"primfunc handler lowering {operator_kind} operator {call.name!r}"
    try:
        source = inspect.getsource(call.operator.func)
    except OSError as err:
        raise error(context, "requires Python source") from err
    source = textwrap.dedent(source)
    module = ast.parse(source)
    for node in module.body:
        if isinstance(node, ast.FunctionDef):
            return node, source
    raise error(context, "could not find Python function body")


def lower_iter(
    function: ast.FunctionDef,
    source: str,
    call: OperatorCall,
    context: str,
    *,
    specialization_constants: Mapping[str, Any] | None = None,
) -> DataflowBodyIR:
    constants = resolved_specialization_constants(
        call,
        context,
        overrides=specialization_constants,
    )
    statements = specialize_top_level_body(
        body_without_docstring(function),
        context,
        specialization_constants=constants,
    )
    if looks_like_raw_tilelang_iter(
        statements,
        allow_no_return=call.output_type is None,
    ):
        return lower_raw_tilelang_iter(
            function,
            source,
            call,
            context,
            statements=statements,
            specialization_constants=constants,
        )
    accumulators, loop_statement, return_statement = split_accumulator_loop_return(statements, context)
    accumulator_names = {item.name for item in accumulators}
    scalar_names = accumulator_names | scalar_parameter_names(call)
    intermediate_collections = intermediate_collection_parameter_types(call)
    loop = parse_iter_loop(
        loop_statement,
        accumulator_names,
        scalar_names,
        tensor_parameter_names(call),
        intermediate_collections,
        context,
    )
    returns = parse_intermediate_return(return_statement, call, context)
    validate_returned_accumulators(returns, accumulators, context)
    return DataflowBodyIR(
        operator_kind="iter",
        operator_name=call.name,
        accumulators=accumulators,
        loop=loop,
        returns=returns,
    )


def lower_raw_tilelang_iter(
    function: ast.FunctionDef,
    source: str,
    call: OperatorCall,
    context: str,
    *,
    statements: list[ast.stmt] | None = None,
    specialization_constants: Mapping[str, Any] | None = None,
) -> DataflowBodyIR:
    if statements is None:
        statements = body_without_docstring(function)
    outputless = call.output_type is None
    return_statement = statements[-1] if statements and isinstance(statements[-1], ast.Return) else None
    if outputless:
        if return_statement is not None and not (
            return_statement.value is None or (isinstance(return_statement.value, ast.Constant) and return_statement.value.value is None)
        ):
            raise error(
                context,
                "outputless raw TileLang map must return None or fall through",
            )
        body_statements = statements[:-1] if return_statement is not None else statements
    else:
        if len(statements) < 2 or return_statement is None:
            raise error(
                context,
                "raw TileLang iter body must end with an intermediate return",
            )
        body_statements = statements[:-1]
    if not body_statements:
        raise error(context, "raw TileLang iter body must contain TileLang statements before return")
    for statement in body_statements:
        if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            raise error(context, "raw TileLang iter body cannot contain nested returns")

    tilelang_body = []
    for statement in body_statements:
        tilelang_body.append(
            normalize_raw_tilelang_source(
                raw_tilelang_statement_source(source, statement, context),
                specialization_constants=specialization_constants,
                context=context,
            )
        )

    raw_source = "\n".join(tilelang_body)
    if not contains_tilelang_primitive(raw_source):
        raise error(context, "raw TileLang iter body must contain TileLang primitive statements")
    returns = {} if outputless else parse_intermediate_return(return_statement, call, context)
    return DataflowBodyIR(
        operator_kind="iter",
        operator_name=call.name,
        returns=returns,
        tilelang_body=tuple(tilelang_body),
    )


def specialize_top_level_body(
    statements: list[ast.stmt],
    context: str,
    *,
    specialization_constants: Mapping[str, Any],
) -> list[ast.stmt]:
    """Select a single static body variant before raw TileLang detection."""

    selected_statements = list(statements)
    while selected_statements and isinstance(selected_statements[0], ast.If):
        selected = static_condition_value(
            selected_statements[0].test,
            specialization_constants,
            context,
        )
        if selected is None:
            break
        branch = selected_statements[0].body if selected else selected_statements[0].orelse
        suffix = selected_statements[1:]
        if not branch and not suffix:
            raise error(context, "selected static body variant must not be empty")
        if branch and isinstance(branch[-1], ast.Return):
            selected_statements = list(branch)
        else:
            selected_statements = [*branch, *suffix]
    return selected_statements


def static_condition_value(
    expression: ast.expr,
    constants: Mapping[str, Any],
    context: str,
) -> bool | None:
    def scalar(node: ast.expr) -> Any:
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Call) and resolve_call_name(node) in _RANGE_TILES_PER_HANDLER_CALLS:
            if node.args or node.keywords:
                raise error(
                    context,
                    "T.dataflow_range_tiles_per_handler() does not accept arguments",
                )
            return selected_range_tiles_per_handler(constants, context)
        if isinstance(node, ast.Name):
            if node.id == DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION:
                raise error(
                    context,
                    "bare dataflow_range_tiles_per_handler is unsupported; use T.dataflow_range_tiles_per_handler()",
                )
            value = constants.get(node.id, _MISSING_STATIC_VALUE)
            if value is None or isinstance(value, (bool, int, float, str)):
                return value
        return _MISSING_STATIC_VALUE

    direct = scalar(expression)
    if direct is not _MISSING_STATIC_VALUE:
        return bool(direct)
    if isinstance(expression, ast.UnaryOp) and isinstance(expression.op, ast.Not):
        value = static_condition_value(expression.operand, constants, context)
        return None if value is None else not value
    if isinstance(expression, ast.Compare) and len(expression.ops) == 1 and len(expression.comparators) == 1:
        left = scalar(expression.left)
        right = scalar(expression.comparators[0])
        if left is _MISSING_STATIC_VALUE or right is _MISSING_STATIC_VALUE:
            return None
        if isinstance(expression.ops[0], ast.Eq):
            return left == right
        if isinstance(expression.ops[0], ast.NotEq):
            return left != right
        if isinstance(expression.ops[0], ast.Lt):
            return left < right
        if isinstance(expression.ops[0], ast.LtE):
            return left <= right
        if isinstance(expression.ops[0], ast.Gt):
            return left > right
        if isinstance(expression.ops[0], ast.GtE):
            return left >= right
    return None


_MISSING_STATIC_VALUE = object()


def resolved_specialization_constants(
    call: OperatorCall,
    context: str,
    *,
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    raw_constants = call.operator.attrs.get("specialization_constants", {})
    if not isinstance(raw_constants, Mapping):
        raise error(context, "specialization_constants must be a mapping")
    if overrides is not None and not isinstance(overrides, Mapping):
        raise error(context, "specialization overrides must be a mapping")
    constants = call.operator.specialization_values
    constants.update(raw_constants)
    constants.update(overrides or {})
    return constants


def selected_range_tiles_per_handler(
    constants: Mapping[str, Any],
    context: str,
) -> int:
    value = constants.get(
        DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION,
        _MISSING_STATIC_VALUE,
    )
    if value is _MISSING_STATIC_VALUE:
        raise error(
            context,
            "T.dataflow_range_tiles_per_handler() requires a selected range coarsening plan",
        )
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise error(
            context,
            f"selected range tiles_per_handler must be a positive integer, got {value!r}",
        )
    return value


def lower_reduce(function: ast.FunctionDef, source: str, call: OperatorCall, context: str) -> DataflowBodyIR:
    statements = body_without_docstring(function)
    if call.operator.reducer_contract is DataflowReducerContract.ASSOCIATIVE_BINARY:
        return lower_associative_reduce(statements, source, call, context)
    if looks_like_raw_tilelang_reduce(statements):
        return lower_raw_tilelang_reduce(source, statements, call, context)
    accumulators, loop_statement, return_statement = split_accumulator_loop_return(statements, context)
    collection_name = first_parameter_name(call, context)
    field_names = input_intermediate_field_names(call, context)
    field_shapes = input_intermediate_field_shapes(call, context)
    loop = parse_reduce_loop(
        loop_statement,
        collection_name,
        {item.name for item in accumulators},
        field_names,
        field_shapes,
        context,
    )
    returns = parse_intermediate_return(return_statement, call, context)
    validate_returned_accumulators(returns, accumulators, context)
    return DataflowBodyIR(
        operator_kind="reduce",
        operator_name=call.name,
        accumulators=accumulators,
        loop=loop,
        returns=returns,
        reducer_contract=DataflowReducerContract.LEGACY_NARY,
        reducer_input_names=(first_parameter_name(call, context),),
    )


def lower_associative_reduce(
    statements: list[ast.stmt],
    source: str,
    call: OperatorCall,
    context: str,
) -> DataflowBodyIR:
    input_names = tuple(call.operator.signature.parameters)
    if len(input_names) != 2:
        raise error(
            context,
            "associative reducer must declare exactly two intermediate parameters",
        )
    if looks_like_raw_tilelang_reduce(statements):
        lowered = lower_raw_tilelang_reduce(source, statements, call, context)
        return replace(
            lowered,
            reducer_contract=DataflowReducerContract.ASSOCIATIVE_BINARY,
            reducer_input_names=input_names,
        )
    if len(statements) != 1 or not isinstance(statements[0], ast.Return):
        raise error(
            context,
            "associative expression reducer must directly return an intermediate, or use raw TileLang statements",
        )
    returns = parse_intermediate_return(statements[0], call, context)
    validate_associative_returns(returns, call, input_names, context)
    return DataflowBodyIR(
        operator_kind="reduce",
        operator_name=call.name,
        returns=returns,
        reducer_contract=DataflowReducerContract.ASSOCIATIVE_BINARY,
        reducer_input_names=input_names,
    )


def lower_raw_tilelang_reduce(
    source: str,
    statements: list[ast.stmt],
    call: OperatorCall,
    context: str,
) -> DataflowBodyIR:
    if len(statements) < 2 or not isinstance(statements[-1], ast.Return):
        raise error(context, "raw TileLang reduce body must end with an intermediate return")
    body_statements = statements[:-1]
    if not body_statements:
        raise error(context, "raw TileLang reduce body must contain TileLang statements before return")
    tilelang_body = []
    for statement in body_statements:
        if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            raise error(context, "raw TileLang reduce body cannot contain nested returns")
        tilelang_body.append(normalize_raw_tilelang_source(raw_tilelang_statement_source(source, statement, context)))

    raw_source = "\n".join(tilelang_body)
    if not contains_tilelang_primitive(raw_source):
        raise error(context, "raw TileLang reduce body must contain TileLang primitive statements")
    returns = parse_intermediate_return(statements[-1], call, context)
    return DataflowBodyIR(
        operator_kind="reduce",
        operator_name=call.name,
        returns=returns,
        tilelang_body=tuple(tilelang_body),
        reducer_contract=(call.operator.reducer_contract or DataflowReducerContract.LEGACY_NARY),
        reducer_input_names=tuple(call.operator.signature.parameters),
    )


def validate_associative_returns(
    returns: Mapping[str, ExprIR],
    call: OperatorCall,
    input_names: tuple[str, str],
    context: str,
) -> None:
    field_names = input_intermediate_field_names(call, context)

    def visit(expression: ExprIR) -> None:
        if isinstance(expression, (LiteralIR,)):
            return
        if isinstance(expression, ScalarVarIR):
            raise error(
                context,
                f"associative return references unsupported local {expression.name!r}",
            )
        if isinstance(expression, CastIR):
            visit(expression.value)
            return
        if isinstance(expression, BinaryOpIR):
            visit(expression.left)
            visit(expression.right)
            return
        if isinstance(expression, CallIR):
            for argument in expression.args:
                visit(argument)
            return
        if isinstance(expression, FieldAccessIR):
            if not isinstance(expression.base, ScalarVarIR) or expression.base.name not in input_names:
                raise error(context, "associative return field must be read from left or right")
            require_intermediate_field(expression.field_name, field_names, context)
            return
        if isinstance(expression, FieldElementAccessIR):
            if not isinstance(expression.base, ScalarVarIR) or expression.base.name not in input_names:
                raise error(context, "associative return field element must be read from left or right")
            require_intermediate_field(expression.field_name, field_names, context)
            return
        if isinstance(expression, TupleExprIR):
            for value in expression.values:
                visit(value)
            return
        raise error(
            context,
            f"unsupported associative return expression {expression!r}",
        )

    for returned in returns.values():
        visit(returned)


def lower_finalize(
    function: ast.FunctionDef,
    source: str,
    call: OperatorCall,
    context: str,
) -> DataflowBodyIR:
    statements = body_without_docstring(function)
    if statements and is_empty_return(statements[-1]):
        statements = statements[:-1]
    if not statements:
        raise error(context, "finalize body must contain output tensor assignments")
    if looks_like_raw_tilelang_finalize(statements):
        return lower_raw_tilelang_finalize(source, statements, call.name, context)
    input_name = first_parameter_name(call, context)
    output_names = finalize_output_parameter_names(call, input_name)
    scalar_names = scalar_parameter_names(call)
    field_names = input_intermediate_field_names(call, context)
    field_shapes = input_intermediate_field_shapes(call, context)
    stores = tuple(
        parse_finalize_store(
            statement,
            input_name,
            field_names,
            field_shapes,
            output_names,
            scalar_names,
            context,
        )
        for statement in statements
    )
    return DataflowBodyIR(operator_kind="finalize", operator_name=call.name, stores=stores)


def lower_raw_tilelang_finalize(
    source: str,
    statements: list[ast.stmt],
    operator_name: str,
    context: str,
) -> DataflowBodyIR:
    tilelang_body = []
    for statement in statements:
        if any(isinstance(node, ast.Return) for node in ast.walk(statement)):
            raise error(context, "raw TileLang finalize body cannot contain returns")
        tilelang_body.append(normalize_raw_tilelang_source(raw_tilelang_statement_source(source, statement, context)))

    raw_source = "\n".join(tilelang_body)
    if not contains_tilelang_primitive(raw_source):
        raise error(context, "raw TileLang finalize body must contain TileLang primitive statements")
    return DataflowBodyIR(
        operator_kind="finalize",
        operator_name=operator_name,
        tilelang_body=tuple(tilelang_body),
    )


def body_without_docstring(function: ast.FunctionDef) -> list[ast.stmt]:
    body = list(function.body)
    if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) and isinstance(body[0].value.value, str):
        body.pop(0)
    return body


def looks_like_raw_tilelang_iter(
    statements: list[ast.stmt],
    *,
    allow_no_return: bool = False,
) -> bool:
    if not statements:
        return False
    has_return = isinstance(statements[-1], ast.Return)
    if not allow_no_return and (len(statements) < 2 or not has_return):
        return False
    body = statements[:-1] if has_return else statements
    if not body:
        return False
    try:
        raw_source = "\n".join(node_source(statement) for statement in body)
    except Exception:
        return False
    return contains_tilelang_primitive(raw_source)


def looks_like_raw_tilelang_reduce(statements: list[ast.stmt]) -> bool:
    if len(statements) < 2 or not isinstance(statements[-1], ast.Return):
        return False
    try:
        raw_source = "\n".join(node_source(statement) for statement in statements[:-1])
    except Exception:
        return False
    return contains_tilelang_primitive(raw_source)


def looks_like_raw_tilelang_finalize(statements: list[ast.stmt]) -> bool:
    try:
        raw_source = "\n".join(node_source(statement) for statement in statements)
    except Exception:
        return False
    return contains_tilelang_primitive(raw_source)


def contains_tilelang_primitive(source: str) -> bool:
    return any(
        marker in source
        for marker in (
            "T.alloc_shared",
            "T.alloc_fragment",
            "T.Pipelined",
            "T.Parallel",
            "T.copy",
            "T.gemm",
            "T.padded_wgmma_gemm",
            "T.fill",
        )
    )


def raw_tilelang_statement_source(source: str, statement: ast.stmt, context: str) -> str:
    if isinstance(statement, (ast.If, ast.For, ast.While, ast.With)):
        return ast.unparse(statement).rstrip()
    segment = ast.get_source_segment(source, statement)
    if segment is None:
        raise error(context, f"could not recover source for raw TileLang statement {node_source(statement)}")
    return textwrap.dedent(segment).rstrip()


class RangeTilesPerHandlerRewriter(ast.NodeTransformer):
    def __init__(
        self,
        constants: Mapping[str, Any],
        context: str,
    ) -> None:
        self.constants = constants
        self.context = context
        self.replaced = False

    def visit_Call(self, node: ast.Call) -> ast.expr:
        if resolve_call_name(node) not in _RANGE_TILES_PER_HANDLER_CALLS:
            return self.generic_visit(node)
        if node.args or node.keywords:
            raise error(
                self.context,
                "T.dataflow_range_tiles_per_handler() does not accept arguments",
            )
        self.replaced = True
        return ast.copy_location(
            ast.Constant(
                value=selected_range_tiles_per_handler(
                    self.constants,
                    self.context,
                )
            ),
            node,
        )

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id == DATAFLOW_RANGE_TILES_PER_HANDLER_SPECIALIZATION:
            raise error(
                self.context,
                "bare dataflow_range_tiles_per_handler is unsupported; use T.dataflow_range_tiles_per_handler()",
            )
        return node


def normalize_raw_tilelang_source(
    source: str,
    *,
    specialization_constants: Mapping[str, Any] | None = None,
    context: str = "Dataflow raw TileLang body",
) -> str:
    try:
        module = ast.parse(source)
    except SyntaxError as err:
        raise error(context, "could not parse raw TileLang statement") from err
    rewriter = RangeTilesPerHandlerRewriter(
        specialization_constants or {},
        context,
    )
    module = rewriter.visit(module)
    if rewriter.replaced:
        ast.fix_missing_locations(module)
        source = "\n".join(ast.unparse(statement).rstrip() for statement in module.body)
    source = re.sub(
        r"(?:T\.)?dataflow_next_task_coord\(\s*(\d+)\s*\)",
        r"dataflow_next_task_coord_\1",
        source,
    )
    replacements = {
        "T.dataflow_range_begin()": "range_begin",
        "dataflow_range_begin()": "range_begin",
        "T.dataflow_range_end()": "range_end",
        "dataflow_range_end()": "range_end",
        "T.dataflow_task_id()": "task_id",
        "dataflow_task_id()": "task_id",
        "T.dataflow_handoff_stage_count()": "dataflow_handoff_stage_count",
        "dataflow_handoff_stage_count()": "dataflow_handoff_stage_count",
        "T.dataflow_input_count()": "input_count",
        "dataflow_input_count()": "input_count",
    }
    for old, new in replacements.items():
        source = source.replace(old, new)
    return source


def split_accumulator_loop_return(
    statements: list[ast.stmt],
    context: str,
) -> tuple[tuple[DataflowAccumulatorIR, ...], ast.For, ast.Return]:
    if len(statements) < 3:
        raise error(context, "body must initialize an accumulator, contain one loop, and return an intermediate")

    accumulators: list[DataflowAccumulatorIR] = []
    index = 0
    while index < len(statements) and isinstance(statements[index], ast.Assign):
        accumulators.append(parse_accumulator(statements[index], context))
        index += 1

    if not accumulators:
        raise error(context, "body must initialize at least one scalar accumulator")
    if index >= len(statements) or not isinstance(statements[index], ast.For):
        raise error(context, "body must contain one loop after accumulator initialization")
    loop_statement = statements[index]
    index += 1
    if index >= len(statements) or not isinstance(statements[index], ast.Return):
        raise error(context, "body must return an intermediate after the loop")
    return_statement = statements[index]
    index += 1
    if index != len(statements):
        raise error(context, "body has unsupported statements after the return")
    return tuple(accumulators), loop_statement, return_statement


def parse_accumulator(statement: ast.Assign, context: str) -> DataflowAccumulatorIR:
    if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
        raise error(context, "accumulator initialization must assign to a local variable")
    init = parse_accumulator_init(statement.value, context)
    dtype = init.dtype
    return DataflowAccumulatorIR(name=statement.targets[0].id, dtype=dtype, init=init)


def parse_accumulator_init(expression: ast.expr, context: str) -> CastIR:
    if not isinstance(expression, ast.Call):
        raise error(context, "accumulator must be initialized with a typed scalar literal")
    call_name = resolve_call_name(expression)
    if not is_typed_scalar_call_name(call_name) or len(expression.args) != 1 or expression.keywords:
        raise error(context, "accumulator must be initialized with a typed scalar literal")
    value = expression.args[0]
    literal = numeric_literal_value(value)
    if literal is None:
        raise error(context, "accumulator must be initialized with a typed scalar literal")
    assert call_name is not None
    return CastIR(dtype=call_name.rsplit(".", 1)[-1], value=LiteralIR(literal))


def parse_iter_loop(
    statement: ast.For,
    accumulators: set[str],
    scalar_names: set[str],
    tensor_names: set[str],
    intermediate_collections: Mapping[str, Any],
    context: str,
) -> DataflowLoopIR:
    if not isinstance(statement.target, ast.Name):
        raise error(context, "iter loop target must be a local variable")
    if statement.orelse:
        raise error(context, "iter loop must not use for-else")
    if not is_scheduled_range(statement.iter):
        raise error(context, "iter loop must be `for i in T.serial(T.dataflow_range_begin(), T.dataflow_range_end())`")
    loop_var = statement.target.id
    body = tuple(
        parse_iter_stmt(
            item,
            accumulators,
            scalar_names | {loop_var},
            tensor_names,
            intermediate_collections,
            context,
        )
        for item in statement.body
    )
    if not body:
        raise error(context, "iter loop must update at least one accumulator")
    return DataflowLoopIR(var=ScalarVarIR(statement.target.id), iterable_kind="dataflow_range", body=body)


def parse_reduce_loop(
    statement: ast.For,
    collection_name: str,
    accumulators: set[str],
    field_names: set[str],
    field_shapes: dict[str, tuple[int, ...] | None],
    context: str,
) -> DataflowLoopIR:
    if not isinstance(statement.target, ast.Name):
        raise error(context, "reduce loop target must be a local variable")
    if statement.orelse:
        raise error(context, "reduce loop must not use for-else")
    if not is_name(statement.iter, collection_name):
        raise error(context, f"reduce loop must iterate over intermediate input parameter {collection_name!r}")
    item_var = statement.target.id
    body = tuple(parse_reduce_stmt(item, accumulators, item_var, field_names, field_shapes, context) for item in statement.body)
    if not body:
        raise error(context, "reduce loop must update at least one accumulator")
    return DataflowLoopIR(var=ScalarVarIR(statement.target.id), iterable_kind=collection_name, body=body)


def parse_iter_stmt(
    statement: ast.stmt,
    accumulators: set[str],
    scalar_names: set[str],
    tensor_names: set[str],
    intermediate_collections: Mapping[str, Any],
    context: str,
) -> StmtIR:
    if isinstance(statement, ast.AugAssign):
        if not isinstance(statement.target, ast.Name) or statement.target.id not in accumulators:
            raise error(context, "loop update target must be one of the initialized accumulators")
        return AugAssignIR(
            target=ScalarVarIR(statement.target.id),
            op=operator_symbol(statement.op, context),
            value=parse_iter_expr(statement.value, scalar_names, tensor_names, intermediate_collections, context),
        )
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise error(context, "loop assignment target must be one of the initialized accumulators")
        if statement.targets[0].id not in accumulators:
            raise error(context, "loop assignment target must be one of the initialized accumulators")
        return AssignIR(
            target=ScalarVarIR(statement.targets[0].id),
            value=parse_iter_expr(statement.value, scalar_names, tensor_names, intermediate_collections, context),
        )
    raise error(context, "loop body currently supports accumulator assignments only")


def parse_reduce_stmt(
    statement: ast.stmt,
    accumulators: set[str],
    item_var: str,
    field_names: set[str],
    field_shapes: dict[str, tuple[int, ...] | None],
    context: str,
) -> StmtIR:
    if isinstance(statement, ast.AugAssign):
        if not isinstance(statement.target, ast.Name) or statement.target.id not in accumulators:
            raise error(context, "loop update target must be one of the initialized accumulators")
        return AugAssignIR(
            target=ScalarVarIR(statement.target.id),
            op=operator_symbol(statement.op, context),
            value=parse_reduce_expr(statement.value, accumulators, item_var, field_names, field_shapes, context),
        )
    if isinstance(statement, ast.Assign):
        if len(statement.targets) != 1 or not isinstance(statement.targets[0], ast.Name):
            raise error(context, "loop assignment target must be one of the initialized accumulators")
        if statement.targets[0].id not in accumulators:
            raise error(context, "loop assignment target must be one of the initialized accumulators")
        return AssignIR(
            target=ScalarVarIR(statement.targets[0].id),
            value=parse_reduce_expr(statement.value, accumulators, item_var, field_names, field_shapes, context),
        )
    raise error(context, "loop body currently supports accumulator assignments only")


def parse_intermediate_return(statement: ast.Return, call: OperatorCall, context: str) -> dict[str, ExprIR]:
    if not isinstance(statement.value, ast.Call):
        raise error(context, "return must construct the Dataflow intermediate")
    output_type = call.output_type
    if output_type is None:
        raise error(context, "operator has no declared intermediate output type")
    constructor_name = resolve_call_name(statement.value.func)
    if constructor_name != output_type.name:
        raise error(context, f"return must construct declared intermediate type {output_type.name!r}")
    if statement.value.args:
        raise error(context, "intermediate return must use keyword field construction only")
    returns: dict[str, ExprIR] = {}
    for keyword in statement.value.keywords:
        if keyword.arg is None:
            raise error(context, "intermediate return cannot use **kwargs")
        if keyword.arg in returns:
            raise error(context, f"duplicate return value for field {keyword.arg!r}")
        returns[keyword.arg] = parse_expr(keyword.value, context)
    expected = {field.name for field in output_type.fields}
    actual = set(returns)
    if actual != expected:
        raise error(context, f"return fields must exactly match {sorted(expected)}, got {sorted(actual)}")
    return returns


def validate_returned_accumulators(
    returns: dict[str, ExprIR],
    accumulators: tuple[DataflowAccumulatorIR, ...],
    context: str,
) -> None:
    accumulator_names = {item.name for item in accumulators}
    flattened = sorted({name for value in returns.values() for name in referenced_scalar_names(value) - accumulator_names})
    if flattened:
        raise error(context, f"returns uninitialized accumulator(s): {', '.join(flattened)}")


def referenced_scalar_names(expression: ExprIR) -> set[str]:
    if isinstance(expression, ScalarVarIR):
        return {expression.name}
    if isinstance(expression, CastIR):
        return referenced_scalar_names(expression.value)
    if isinstance(expression, BinaryOpIR):
        return referenced_scalar_names(expression.left) | referenced_scalar_names(expression.right)
    if isinstance(expression, CallIR):
        return {name for arg in expression.args for name in referenced_scalar_names(arg)}
    if isinstance(expression, TensorLoadIR):
        return {name for index in expression.indices for name in referenced_scalar_names(index)}
    if isinstance(expression, SliceIR):
        names: set[str] = set()
        if expression.start is not None:
            names |= referenced_scalar_names(expression.start)
        if expression.stop is not None:
            names |= referenced_scalar_names(expression.stop)
        return names
    if isinstance(expression, FieldAccessIR):
        return referenced_scalar_names(expression.base)
    if isinstance(expression, FieldElementAccessIR):
        return referenced_scalar_names(expression.base)
    if isinstance(expression, TupleExprIR):
        return {name for value in expression.values for name in referenced_scalar_names(value)}
    return set()


def parse_finalize_store(
    statement: ast.stmt,
    input_name: str,
    field_names: set[str],
    field_shapes: dict[str, tuple[int, ...] | None],
    output_names: set[str],
    scalar_names: set[str],
    context: str,
) -> TensorStoreIR:
    if not isinstance(statement, ast.Assign) or len(statement.targets) != 1:
        raise error(context, "finalize body must contain output assignments")
    target = statement.targets[0]
    if not isinstance(target, ast.Subscript) or not isinstance(target.value, ast.Name):
        raise error(context, "finalize output assignment must target a tensor subscript")
    output_name = target.value.id
    if output_name not in output_names:
        raise error(context, f"finalize output tensor {output_name!r} is not a bound output parameter")
    index = parse_finalize_index(target.slice, scalar_names, context)
    if not index_references_task(index, scalar_names):
        raise error(
            context,
            "finalize output tensor index must reference `T.dataflow_task_id()` or a scalar task parameter",
        )
    return TensorStoreIR(
        tensor_name=output_name,
        index=index,
        value=parse_finalize_value(statement.value, input_name, field_names, field_shapes, scalar_names, context),
    )


def parse_finalize_index(
    expression: ast.expr,
    scalar_names: set[str],
    context: str,
) -> ExprIR:
    if isinstance(expression, ast.Tuple):
        return TupleExprIR(values=tuple(parse_finalize_index_item(item, scalar_names, context) for item in expression.elts))
    return parse_finalize_index_item(expression, scalar_names, context)


def parse_finalize_index_item(
    expression: ast.expr,
    scalar_names: set[str],
    context: str,
) -> ExprIR:
    return parse_finalize_index_expr(expression, scalar_names, context)


def parse_finalize_index_expr(
    expression: ast.expr,
    scalar_names: set[str],
    context: str,
) -> ExprIR:
    literal = numeric_literal_value(expression)
    if isinstance(literal, int):
        return LiteralIR(literal)
    if is_zero_arg_call(expression, {"T.dataflow_task_id", "dataflow_task_id"}):
        return ScalarVarIR("T.dataflow_task_id")
    if is_zero_arg_call(expression, {"T.dataflow_range_begin", "dataflow_range_begin"}):
        return ScalarVarIR("T.dataflow_range_begin")
    if is_zero_arg_call(expression, {"T.dataflow_range_end", "dataflow_range_end"}):
        return ScalarVarIR("T.dataflow_range_end")
    if isinstance(expression, ast.Name) and expression.id in scalar_names:
        return ScalarVarIR(expression.id)
    if isinstance(expression, ast.Call):
        call_name = resolve_call_name(expression)
        if is_typed_scalar_call_name(call_name):
            if len(expression.args) != 1 or expression.keywords:
                raise error(context, f"typed scalar call {call_name!r} must have exactly one positional argument")
            return CastIR(
                dtype=call_name.rsplit(".", 1)[-1],
                value=parse_finalize_index_expr(expression.args[0], scalar_names, context),
            )
    if isinstance(expression, ast.BinOp):
        return BinaryOpIR(
            op=operator_symbol(expression.op, context),
            left=parse_finalize_index_expr(expression.left, scalar_names, context),
            right=parse_finalize_index_expr(expression.right, scalar_names, context),
        )
    raise error(
        context,
        "finalize output tensor indices must be integer constants, scalar task parameters, "
        "`T.dataflow_task_id()`, or Dataflow range bounds",
    )


def index_references_task(index: ExprIR, scalar_names: set[str]) -> bool:
    if isinstance(index, ScalarVarIR):
        return index.name in {"T.dataflow_task_id", "T.dataflow_range_begin", "T.dataflow_range_end"} or index.name in scalar_names
    if isinstance(index, CastIR):
        return index_references_task(index.value, scalar_names)
    if isinstance(index, BinaryOpIR):
        return index_references_task(index.left, scalar_names) or index_references_task(index.right, scalar_names)
    if isinstance(index, TupleExprIR):
        return any(index_references_task(value, scalar_names) for value in index.values)
    return False


def parse_iter_expr(
    expression: ast.expr,
    scalar_names: set[str],
    tensor_names: set[str],
    intermediate_collections: Mapping[str, Any],
    context: str,
) -> ExprIR:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (int, float)):
        return LiteralIR(expression.value)
    literal = numeric_literal_value(expression)
    if literal is not None:
        return LiteralIR(literal)
    if isinstance(expression, ast.Name):
        if expression.id not in scalar_names:
            raise error(context, f"iter expression references unsupported scalar {expression.id!r}")
        return ScalarVarIR(expression.id)
    if isinstance(expression, ast.Call):
        call_name = resolve_call_name(expression)
        if is_zero_arg_call(expression, {"T.dataflow_range_begin", "dataflow_range_begin"}):
            return ScalarVarIR("T.dataflow_range_begin")
        if is_zero_arg_call(expression, {"T.dataflow_range_end", "dataflow_range_end"}):
            return ScalarVarIR("T.dataflow_range_end")
        if is_typed_scalar_call_name(call_name):
            if len(expression.args) != 1 or expression.keywords:
                raise error(context, f"typed scalar call {call_name!r} must have exactly one positional argument")
            return CastIR(
                dtype=call_name.rsplit(".", 1)[-1],
                value=parse_iter_expr(
                    expression.args[0],
                    scalar_names,
                    tensor_names,
                    intermediate_collections,
                    context,
                ),
            )
        math_call_name = resolve_math_call_name(call_name)
        if math_call_name is not None:
            validate_math_call_arity(math_call_name, expression, context)
            return CallIR(
                name=math_call_name,
                args=tuple(parse_iter_expr(arg, scalar_names, tensor_names, intermediate_collections, context) for arg in expression.args),
            )
    if isinstance(expression, ast.BinOp):
        return BinaryOpIR(
            op=operator_symbol(expression.op, context),
            left=parse_iter_expr(expression.left, scalar_names, tensor_names, intermediate_collections, context),
            right=parse_iter_expr(expression.right, scalar_names, tensor_names, intermediate_collections, context),
        )
    intermediate_access = parse_iter_intermediate_list_access(expression, intermediate_collections, context)
    if intermediate_access is not None:
        return intermediate_access
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
        tensor_name = expression.value.id
        if tensor_name not in tensor_names:
            raise error(context, f"iter tensor load uses unsupported tensor parameter {tensor_name!r}")
        return TensorLoadIR(
            tensor_name=tensor_name,
            indices=parse_iter_indices(expression.slice, scalar_names, tensor_names, intermediate_collections, context),
        )
    raise error(context, f"unsupported iter expression: {node_source(expression)}")


def parse_iter_indices(
    expression: ast.expr,
    scalar_names: set[str],
    tensor_names: set[str],
    intermediate_collections: Mapping[str, Any],
    context: str,
) -> tuple[ExprIR, ...]:
    if isinstance(expression, ast.Tuple):
        return tuple(parse_iter_expr(item, scalar_names, tensor_names, intermediate_collections, context) for item in expression.elts)
    return (parse_iter_expr(expression, scalar_names, tensor_names, intermediate_collections, context),)


def parse_iter_intermediate_list_access(
    expression: ast.expr,
    intermediate_collections: Mapping[str, IntermediateType],
    context: str,
) -> ExprIR | None:
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Attribute):
        base = intermediate_list_field_base(expression.value, intermediate_collections, context)
        if base is None:
            return None
        slot_name, field_name, intermediate = base
        field = intermediate.field(field_name)
        shape = fixed_shape(field.shape, context, field_name)
        if shape is None:
            raise error(context, f"scalar intermediate field {field_name!r} cannot be indexed")
        indices = constant_indices(expression.slice, context)
        return FieldElementAccessIR(
            base=ScalarVarIR(slot_name),
            field_name=field_name,
            indices=indices,
            flat_index=flat_field_index(field_name, shape, indices, context),
        )
    if isinstance(expression, ast.Attribute):
        base = intermediate_list_field_base(expression, intermediate_collections, context)
        if base is None:
            return None
        slot_name, field_name, _ = base
        return FieldAccessIR(ScalarVarIR(slot_name), field_name)
    return None


def intermediate_list_field_base(
    expression: ast.Attribute,
    intermediate_collections: Mapping[str, IntermediateType],
    context: str,
) -> tuple[str, str, IntermediateType] | None:
    owner = expression.value
    if not isinstance(owner, ast.Subscript) or not isinstance(owner.value, ast.Name):
        return None
    collection_name = owner.value.id
    intermediate = intermediate_collections.get(collection_name)
    if intermediate is None:
        return None
    slot_index = constant_nonnegative_int(owner.slice, context)
    field_name = expression.attr
    try:
        intermediate.field(field_name)
    except KeyError as err:
        raise error(
            context,
            f"unknown intermediate field {field_name!r}; expected {[field.name for field in intermediate.fields]}",
        ) from err
    return f"Item{slot_index}", field_name, intermediate


def constant_nonnegative_int(expression: ast.expr, context: str) -> int:
    literal = numeric_literal_value(expression)
    if not isinstance(literal, int) or literal < 0:
        raise error(context, "intermediate list slot index must be a non-negative integer constant")
    return literal


def parse_reduce_expr(
    expression: ast.expr,
    accumulators: set[str],
    item_var: str,
    field_names: set[str],
    field_shapes: dict[str, tuple[int, ...] | None],
    context: str,
) -> ExprIR:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (int, float)):
        return LiteralIR(expression.value)
    literal = numeric_literal_value(expression)
    if literal is not None:
        return LiteralIR(literal)
    if isinstance(expression, ast.Name):
        if expression.id not in accumulators:
            raise error(context, f"reduce expression references unsupported scalar {expression.id!r}")
        return ScalarVarIR(expression.id)
    if isinstance(expression, ast.Call):
        call_name = resolve_call_name(expression)
        if is_typed_scalar_call_name(call_name):
            if len(expression.args) != 1 or expression.keywords:
                raise error(context, f"typed scalar call {call_name!r} must have exactly one positional argument")
            return CastIR(
                dtype=call_name.rsplit(".", 1)[-1],
                value=parse_reduce_expr(expression.args[0], accumulators, item_var, field_names, field_shapes, context),
            )
        math_call_name = resolve_math_call_name(call_name)
        if math_call_name is not None:
            validate_math_call_arity(math_call_name, expression, context)
            return CallIR(
                name=math_call_name,
                args=tuple(parse_reduce_expr(arg, accumulators, item_var, field_names, field_shapes, context) for arg in expression.args),
            )
    if isinstance(expression, ast.BinOp):
        return BinaryOpIR(
            op=operator_symbol(expression.op, context),
            left=parse_reduce_expr(expression.left, accumulators, item_var, field_names, field_shapes, context),
            right=parse_reduce_expr(expression.right, accumulators, item_var, field_names, field_shapes, context),
        )
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Attribute):
        owner = expression.value.value
        if isinstance(owner, ast.Name) and owner.id == item_var:
            field_name = expression.value.attr
            require_intermediate_field(field_name, field_names, context)
            shape = field_shapes[field_name]
            if shape is None:
                raise error(context, f"scalar intermediate field {field_name!r} cannot be indexed")
            indices = constant_indices(expression.slice, context)
            return FieldElementAccessIR(
                base=ScalarVarIR(item_var),
                field_name=field_name,
                indices=indices,
                flat_index=flat_field_index(field_name, shape, indices, context),
            )
    if isinstance(expression, ast.Attribute) and is_name(expression.value, item_var):
        require_intermediate_field(expression.attr, field_names, context)
        return FieldAccessIR(ScalarVarIR(item_var), expression.attr)
    raise error(context, f"unsupported reduce expression: {node_source(expression)}")


def parse_finalize_value(
    expression: ast.expr,
    input_name: str,
    field_names: set[str],
    field_shapes: dict[str, tuple[int, ...] | None],
    scalar_names: set[str],
    context: str,
) -> ExprIR:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (int, float)):
        return LiteralIR(expression.value)
    literal = numeric_literal_value(expression)
    if literal is not None:
        return LiteralIR(literal)
    if isinstance(expression, ast.Name) and expression.id in scalar_names:
        return ScalarVarIR(expression.id)
    if isinstance(expression, ast.Call):
        call_name = resolve_call_name(expression)
        if is_typed_scalar_call_name(call_name):
            if len(expression.args) != 1 or expression.keywords:
                raise error(context, f"typed scalar call {call_name!r} must have exactly one positional argument")
            return CastIR(
                dtype=call_name.rsplit(".", 1)[-1],
                value=parse_finalize_value(
                    expression.args[0],
                    input_name,
                    field_names,
                    field_shapes,
                    scalar_names,
                    context,
                ),
            )
        math_call_name = resolve_math_call_name(call_name)
        if math_call_name is not None:
            validate_math_call_arity(math_call_name, expression, context)
            return CallIR(
                name=math_call_name,
                args=tuple(
                    parse_finalize_value(arg, input_name, field_names, field_shapes, scalar_names, context) for arg in expression.args
                ),
            )
    if isinstance(expression, ast.BinOp):
        return BinaryOpIR(
            op=operator_symbol(expression.op, context),
            left=parse_finalize_value(expression.left, input_name, field_names, field_shapes, scalar_names, context),
            right=parse_finalize_value(
                expression.right,
                input_name,
                field_names,
                field_shapes,
                scalar_names,
                context,
            ),
        )
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Attribute):
        owner = expression.value.value
        if isinstance(owner, ast.Name) and owner.id == input_name:
            field_name = expression.value.attr
            require_intermediate_field(field_name, field_names, context)
            shape = field_shapes[field_name]
            if shape is None:
                raise error(context, f"scalar intermediate field {field_name!r} cannot be indexed")
            indices = constant_indices(expression.slice, context)
            return FieldElementAccessIR(
                base=ScalarVarIR(input_name),
                field_name=field_name,
                indices=indices,
                flat_index=flat_field_index(field_name, shape, indices, context),
            )
    if isinstance(expression, ast.Attribute) and is_name(expression.value, input_name):
        require_intermediate_field(expression.attr, field_names, context)
        return FieldAccessIR(ScalarVarIR(input_name), expression.attr)
    raise error(context, f"unsupported finalize expression: {node_source(expression)}")


def parse_expr(expression: ast.expr, context: str) -> ExprIR:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (int, float)):
        return LiteralIR(expression.value)
    literal = numeric_literal_value(expression)
    if literal is not None:
        return LiteralIR(literal)
    if isinstance(expression, ast.Name):
        return ScalarVarIR(expression.id)
    if isinstance(expression, ast.Call):
        call_name = resolve_call_name(expression)
        if call_name in {"T.dataflow_task_id", "dataflow_task_id"} and not expression.args and not expression.keywords:
            return ScalarVarIR("T.dataflow_task_id")
        if is_typed_scalar_call_name(call_name):
            if len(expression.args) != 1 or expression.keywords:
                raise error(context, f"typed scalar call {call_name!r} must have exactly one positional argument")
            return CastIR(dtype=call_name.rsplit(".", 1)[-1], value=parse_expr(expression.args[0], context))
        math_call_name = resolve_math_call_name(call_name)
        if math_call_name is not None:
            validate_math_call_arity(math_call_name, expression, context)
            return CallIR(name=math_call_name, args=tuple(parse_expr(arg, context) for arg in expression.args))
    if isinstance(expression, ast.BinOp):
        return BinaryOpIR(
            op=operator_symbol(expression.op, context),
            left=parse_expr(expression.left, context),
            right=parse_expr(expression.right, context),
        )
    if isinstance(expression, ast.Tuple):
        return TupleExprIR(values=tuple(parse_expr(item, context) for item in expression.elts))
    if isinstance(expression, ast.Slice):
        if expression.step is not None:
            raise error(context, "tensor slices in intermediate returns do not support steps")
        return SliceIR(
            start=None if expression.lower is None else parse_expr(expression.lower, context),
            stop=None if expression.upper is None else parse_expr(expression.upper, context),
        )
    if isinstance(expression, ast.Subscript) and isinstance(expression.value, ast.Name):
        return TensorLoadIR(tensor_name=expression.value.id, indices=parse_indices(expression.slice, context))
    if isinstance(expression, ast.Attribute):
        return FieldAccessIR(base=parse_expr(expression.value, context), field_name=expression.attr)
    raise error(context, f"unsupported expression: {node_source(expression)}")


def parse_indices(expression: ast.expr, context: str) -> tuple[ExprIR, ...]:
    if isinstance(expression, ast.Tuple):
        return tuple(parse_expr(item, context) for item in expression.elts)
    return (parse_expr(expression, context),)


def is_scheduled_range(expression: ast.expr) -> bool:
    if not isinstance(expression, ast.Call) or resolve_call_name(expression) not in {"T.serial", "serial"}:
        return False
    if expression.keywords or len(expression.args) != 2:
        return False
    return is_zero_arg_call(expression.args[0], {"T.dataflow_range_begin", "dataflow_range_begin"}) and is_zero_arg_call(
        expression.args[1], {"T.dataflow_range_end", "dataflow_range_end"}
    )


def is_empty_return(statement: ast.stmt) -> bool:
    return isinstance(statement, ast.Return) and statement.value is None


def first_parameter_name(call: OperatorCall, context: str) -> str:
    for name in call.operator.signature.parameters:
        return name
    raise error(context, "operator must have an input parameter")


def input_intermediate_field_names(call: OperatorCall, context: str) -> set[str]:
    if not call.operator.input_types:
        raise error(context, "operator must consume an intermediate")
    return {field.name for field in call.operator.input_types[0].fields}


def input_intermediate_field_shapes(call: OperatorCall, context: str) -> dict[str, tuple[int, ...] | None]:
    if not call.operator.input_types:
        raise error(context, "operator must consume an intermediate")
    return {field.name: fixed_shape(field.shape, context, field.name) for field in call.operator.input_types[0].fields}


def require_intermediate_field(field_name: str, field_names: set[str], context: str) -> None:
    if field_name not in field_names:
        raise error(context, f"unknown intermediate field {field_name!r}; expected {sorted(field_names)}")


def fixed_shape(shape: tuple[Any, ...] | None, context: str, field_name: str) -> tuple[int, ...] | None:
    if shape is None:
        return None
    extents = []
    for extent in shape:
        try:
            extent_value = int(extent)
        except (TypeError, ValueError) as err:
            raise error(context, f"tensor field {field_name!r} must have fixed integer extents") from err
        if extent_value <= 0:
            raise error(context, f"tensor field {field_name!r} must have positive extents")
        extents.append(extent_value)
    return tuple(extents)


def constant_indices(expression: ast.expr, context: str) -> tuple[int, ...]:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, int):
        return (expression.value,)
    if isinstance(expression, ast.Tuple):
        indices = []
        for item in expression.elts:
            if not isinstance(item, ast.Constant) or not isinstance(item.value, int):
                raise error(context, "tensor field indices must be integer constants")
            indices.append(item.value)
        return tuple(indices)
    raise error(context, "tensor field indices must be integer constants")


def flat_field_index(field_name: str, shape: tuple[int, ...], indices: tuple[int, ...], context: str) -> int:
    numel = 1
    for extent in shape:
        numel *= extent
    if not indices:
        raise error(context, f"tensor field {field_name!r} requires at least one index")
    if len(indices) == 1:
        index = indices[0]
        if index < 0 or index >= numel:
            raise error(
                context,
                f"tensor field {field_name!r} flat index {index} is out of bounds for {numel} element(s)",
            )
        return index
    if len(indices) != len(shape):
        raise error(
            context,
            f"tensor field {field_name!r} expects either one flat index or {len(shape)} rank index(es), got {len(indices)}",
        )
    flat_index = 0
    for index, extent in zip(indices, shape):
        if index < 0 or index >= extent:
            raise error(context, f"tensor field {field_name!r} index {index} is out of bounds for extent {extent}")
        flat_index = flat_index * extent + index
    return flat_index


def is_name(expression: ast.expr, name: str) -> bool:
    return isinstance(expression, ast.Name) and expression.id == name


def is_zero_arg_call(expression: ast.expr, names: set[str]) -> bool:
    return isinstance(expression, ast.Call) and resolve_call_name(expression) in names and not expression.args and not expression.keywords


def tensor_parameter_names(call: OperatorCall) -> set[str]:
    return {name for name in call.operator.signature.parameters if call.operator.is_external_tensor_parameter(name)}


def scalar_parameter_names(call: OperatorCall) -> set[str]:
    names: set[str] = set()
    for name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(name, parameter.annotation)
        if is_scalar_annotation(annotation):
            names.add(name)
    return names


def intermediate_collection_parameter_types(call: OperatorCall) -> dict[str, IntermediateType]:
    result: dict[str, IntermediateType] = {}
    for name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(name, parameter.annotation)
        origin = get_origin(annotation)
        if origin not in (list, tuple, collections.abc.Sequence, collections.abc.Iterable):
            continue
        args = get_args(annotation)
        if len(args) != 1:
            continue
        intermediate = get_intermediate_type(args[0])
        if intermediate is not None:
            result[name] = intermediate
    return result


def finalize_output_parameter_names(call: OperatorCall, input_name: str) -> set[str]:
    names: set[str] = set()
    for name, parameter in call.operator.signature.parameters.items():
        annotation = call.operator.annotations.get(name, parameter.annotation)
        if name == input_name or get_intermediate_type(annotation) is not None:
            continue
        if name in call.bound_arguments:
            names.add(name)
    return names


def is_scalar_annotation(annotation: Any) -> bool:
    if annotation is inspect.Signature.empty:
        return False
    return str(annotation) in {"int32", "uint32", "float16", "float32"}


def operator_symbol(operator: ast.operator, context: str) -> str:
    for operator_type, symbol in _BIN_OPS.items():
        if isinstance(operator, operator_type):
            return symbol
    raise error(context, f"unsupported operator {node_source(operator)}")


def numeric_literal_value(expression: ast.expr) -> int | float | None:
    if isinstance(expression, ast.Constant) and isinstance(expression.value, (int, float)):
        return expression.value
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.USub)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, (int, float))
    ):
        return -expression.operand.value
    if (
        isinstance(expression, ast.UnaryOp)
        and isinstance(expression.op, ast.UAdd)
        and isinstance(expression.operand, ast.Constant)
        and isinstance(expression.operand.value, (int, float))
    ):
        return expression.operand.value
    return None


def resolve_math_call_name(name: str | None) -> str | None:
    if name in {"T.max", "max"}:
        return "T.max"
    if name in {"T.exp2", "exp2"}:
        return "T.exp2"
    return None


def validate_math_call_arity(name: str, expression: ast.Call, context: str) -> None:
    expected = {"T.max": 2, "T.exp2": 1}[name]
    if len(expression.args) != expected or expression.keywords:
        raise error(context, f"math call {name!r} must have exactly {expected} positional argument(s)")


def resolve_call_name(expression: ast.expr) -> str | None:
    if isinstance(expression, ast.Call):
        return resolve_call_name(expression.func)
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute):
        base = resolve_call_name(expression.value)
        if base is None:
            return expression.attr
        return f"{base}.{expression.attr}"
    return None


def is_typed_scalar_call_name(name: str | None) -> bool:
    return name in {
        "T.int32",
        "T.uint32",
        "T.float16",
        "T.float32",
        "int32",
        "uint32",
        "float16",
        "float32",
    }


def node_source(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return node.__class__.__name__


def error(context: str, message: str) -> DataflowBodyIRLoweringError:
    return DataflowBodyIRLoweringError(f"{context} {message}")
