import pickletools

from itertools import count
from pickle import dumps
from pickletools import markobject, pybool, pyint, pynone, pyunicode

import attr
import pytest
import six

import pikara.analysis as a

from pikara.analysis import _parse, _ParseEntry
from pikara.analysis import _ParseResult as _PR

from .compat import intish_type, parametrize_proto


def test_rfind():
    my_sentinel = object()
    assert a._rfind([], 1, my_sentinel) is my_sentinel
    assert a._rfind([1], 1, my_sentinel) == 0
    assert a._rfind([1, 1, 1, 1], 1, my_sentinel) == 3


ops_by_name = {opcode.name: opcode for opcode in pickletools.opcodes}
ops = type("Opcodes", (), ops_by_name)

_MISSING = object()

if getattr(pickletools, "_RawOpcodeInfo", _MISSING) is _MISSING:
    pickletools._RawOpcodeInfo = pickletools.OpcodeInfo
    pickletools.OpcodeInfo = attr.s(
        these={
            "name": attr.ib(),
            "code": attr.ib(),
            "arg": attr.ib(),
            "stack_before": attr.ib(),
            "stack_after": attr.ib(),
            "proto": attr.ib(),
        },
        init=False,
    )(
        pickletools.OpcodeInfo
    )

if getattr(pickletools, "_RawArgumentDescriptor", _MISSING) is _MISSING:
    pickletools._RawArgumentDescriptor = pickletools.ArgumentDescriptor
    pickletools.ArgumentDescriptor = attr.s(
        these={"name": attr.ib(), "n": attr.ib()}, init=False
    )(
        pickletools.ArgumentDescriptor
    )


@attr.s(cmp=False)
class _Wildcard(object):
    """
    A wildcard object, equal to any other object.

    This helps us preserve some position data in our tests without actually
    checking that value.
    """
    expected_value = attr.ib(default=None)

    def __eq__(self, other):
        return True

    def __ne__(self, other):
        return False


def _PE(**kw):
    kw["pos"] = _Wildcard(kw.get("pos"))
    return _ParseEntry(**kw)


def _constant_fn(v):
    return lambda *a, **kw: v


def _identity(v):
    return v


def _memoize_ops(proto):
    if proto >= 4:
        op = ops.MEMOIZE
        argfn = _constant_fn(None)
    elif proto > 0:
        op = ops.BINPUT
        argfn = _identity
    else:
        op = ops.PUT
        argfn = _identity

    for i in count():
        yield _PE(op=op, arg=argfn(i), pos=_Wildcard(), stackslice=None)


@parametrize_proto()
def test_unicode_string(proto, maxproto):
    parse_entries = []
    if proto >= 2:
        parse_entries.append(
            _PE(op=ops.PROTO, arg=proto, pos=0, stackslice=None)
        )
    parse_entries += [
        _PE(
            op=ops.BINUNICODE if proto > 0 else ops.UNICODE,
            arg=u"a",
            pos=2,
            stackslice=None,
        ),
        _PE(
            op=ops.BINPUT if proto > 0 else ops.PUT,
            arg=0,
            pos=8,
            stackslice=None,
        ),
        _PE(op=ops.STOP, arg=None, pos=10, stackslice=[pyunicode]),
    ]
    expected = _PR(
        parse_entries=parse_entries,
        maxproto=maxproto,
        stack=[],
        memo={0: pyunicode},
    )
    actual = a._parse(dumps(u"a", protocol=proto))
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


@parametrize_proto(protos=[1, 2, 3, 4])
def test_list_of_three_ints(proto, maxproto):
    """
    This test isn't run for p0 because p0 doesn't have APPENDS so the internal
    opcode structure is quite different.
    """
    parse_entries = []
    if proto >= 2:
        parse_entries.append(
            _PE(op=ops.PROTO, arg=proto, pos=0, stackslice=None)
        )
    if proto >= 4:
        parse_entries.append(_PE(op=ops.FRAME, arg=11, pos=2, stackslice=None))
        PUT = _PE(op=ops.MEMOIZE, arg=None, pos=3, stackslice=None)
    else:
        PUT = _PE(op=ops.BINPUT, arg=0, pos=3, stackslice=None)
    parse_entries += [
        _PE(op=ops.EMPTY_LIST, arg=None, pos=2, stackslice=None),
        PUT,
        _PE(op=ops.MARK, arg=None, pos=5, stackslice=None),
        _PE(op=ops.BININT1, arg=1, pos=6, stackslice=None),
        _PE(op=ops.BININT1, arg=2, pos=8, stackslice=None),
        _PE(op=ops.BININT1, arg=3, pos=10, stackslice=None),
        _PE(
            op=ops.APPENDS,
            arg=None,
            pos=12,
            stackslice=[[], markobject, [pyint, pyint, pyint]],
        ),
        _PE(op=ops.STOP, arg=None, pos=13, stackslice=[[pyint, pyint, pyint]]),
    ]
    expected = _PR(
        parse_entries=parse_entries, maxproto=maxproto, stack=[], memo={0: []}
    )
    actual = a._parse(dumps([1, 2, 3], protocol=proto))
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


def test_list_of_three_ints_p0():
    """
    A list of three ints, in protocol version zero.

    This structurally changes the list because p0 didn't have APPENDS, so each
    element is added manually.
    """
    intish = intish_type(0)
    intop = ops.LONG if six.PY3 else ops.INT
    expected = _PR(
        parse_entries=[
            _PE(op=ops.MARK, arg=None, pos=0, stackslice=None),
            _PE(op=ops.LIST, arg=None, pos=1, stackslice=[markobject, []]),
            _PE(op=ops.PUT, arg=0, pos=2, stackslice=None),
            _PE(op=intop, arg=1, pos=5, stackslice=None),
            _PE(
                op=ops.APPEND, arg=None, pos=9, stackslice=[[], intish]
            ),  # after stack to [pyint]
            _PE(op=intop, arg=2, pos=10, stackslice=None),
            _PE(
                op=ops.APPEND, arg=None, pos=14, stackslice=[[intish], intish]
            ),
            _PE(op=intop, arg=3, pos=15, stackslice=None),
            _PE(
                op=ops.APPEND,
                arg=None,
                pos=19,
                stackslice=[[intish, intish], intish],
            ),
            _PE(
                op=ops.STOP,
                arg=None,
                pos=20,
                stackslice=[[intish, intish, intish]],
            ),
        ],
        maxproto=0,
        stack=[],
        memo={0: []},
    )
    actual = _parse(dumps([1, 2, 3], protocol=0))
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


@parametrize_proto(protos=[3])
def test_nested_list(proto, maxproto):
    """
    A test for parsing nested lists.

    This test is not run on protocol version zero because the lack of
    MARK/APPENDS fundamentally changes the structure of the pickle. We skip
    version 4 because framing changes the structure of the pickle.
    """
    inner = [1]
    middle = [2, inner]
    outer = [3, middle]

    # innerslice: no markobject because plain append, not appends
    innerslice = [[], pyint]
    middleslice = [[], markobject, [pyint, [pyint]]]
    outerslice = [[], markobject, [pyint, [pyint, [pyint]]]]

    expected = _PR(
        parse_entries=[
            _PE(op=ops.PROTO, arg=3, pos=0, stackslice=None),
            # Outer list
            _PE(op=ops.EMPTY_LIST, arg=None, pos=2, stackslice=None),
            _PE(op=ops.BINPUT, arg=0, pos=3, stackslice=None),
            _PE(op=ops.MARK, arg=None, pos=5, stackslice=None),
            _PE(op=ops.BININT1, arg=3, pos=6, stackslice=None),
            # Middle list
            _PE(op=ops.EMPTY_LIST, arg=None, pos=8, stackslice=None),
            _PE(op=ops.BINPUT, arg=1, pos=9, stackslice=None),
            _PE(op=ops.MARK, arg=None, pos=11, stackslice=None),
            _PE(op=ops.BININT1, arg=2, pos=12, stackslice=None),
            # Inner list
            _PE(op=ops.EMPTY_LIST, arg=None, pos=14, stackslice=None),
            _PE(op=ops.BINPUT, arg=2, pos=15, stackslice=None),
            _PE(op=ops.BININT1, arg=1, pos=17, stackslice=None),
            # Build inner, middle, outer lists
            _PE(op=ops.APPEND, arg=None, pos=19, stackslice=innerslice),
            _PE(op=ops.APPENDS, arg=None, pos=20, stackslice=middleslice),
            _PE(op=ops.APPENDS, arg=None, pos=21, stackslice=outerslice),
            _PE(
                op=ops.STOP,
                arg=None,
                pos=22,
                stackslice=[[pyint, [pyint, [pyint]]]],
            ),
        ],
        maxproto=maxproto,
        stack=[],
        # TODO: these should actually be mutated!
        memo={0: [], 1: [], 2: []},
    )
    actual = a._parse(dumps(outer, protocol=proto))
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


# TODO: nested_lists_p4


class NullReduce(object):

    def __reduce__(self):
        return NullReduce, ()


@pytest.mark.skipif(six.PY2, reason="protocol v3 is not available in Python 2")
def test_reduce_v3():
    actual = a._parse(dumps(NullReduce(), protocol=3))
    g_nr = actual.global_objects[("tests.test_parse", "NullReduce")]
    expected = _PR(
        parse_entries=[
            _PE(op=ops.PROTO, arg=3, pos=0, stackslice=None),
            _PE(
                op=ops.GLOBAL,
                arg="tests.test_parse NullReduce",
                pos=2,
                stackslice=None,
            ),
            _PE(op=ops.BINPUT, arg=0, pos=31, stackslice=None),
            _PE(op=ops.EMPTY_TUPLE, arg=None, pos=33, stackslice=None),
            _PE(op=ops.REDUCE, arg=None, pos=34, stackslice=[g_nr, ()]),
            _PE(op=ops.BINPUT, arg=1, pos=35, stackslice=None),
            _PE(op=ops.STOP, arg=None, pos=37, stackslice=[[g_nr, ()]]),
        ],
        maxproto=2,
        stack=[],
        memo={0: g_nr, 1: [g_nr, ()]},
    )
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


class ReduceSentinel(object):

    def __init__(self, s):
        self.s = s

    def __reduce__(self):
        return ReduceSentinel, (self.s,)


@pytest.mark.skipif(six.PY2, reason="protocol v3 is not available in Python 2")
def test_reduce_sentinel_v3():
    # int isn't special except that it's a globally available builtin that maps
    # to the name int on py2 and py3.
    actual = a._parse(dumps(ReduceSentinel(int), protocol=3))
    g = actual.global_objects
    g_int = g[("builtins", "int")]
    g_rs = g[("tests.test_parse", "ReduceSentinel")]
    expected = _PR(
        parse_entries=[
            _PE(op=ops.PROTO, arg=3, pos=0, stackslice=None),
            _PE(
                op=ops.GLOBAL,
                arg="tests.test_parse ReduceSentinel",
                pos=2,
                stackslice=None,
            ),
            _PE(op=ops.BINPUT, arg=0, pos=35, stackslice=None),
            _PE(op=ops.GLOBAL, arg="builtins int", pos=37, stackslice=None),
            _PE(op=ops.BINPUT, arg=1, pos=56, stackslice=None),
            _PE(op=ops.TUPLE1, arg=None, pos=58, stackslice=[g_int]),
            _PE(op=ops.BINPUT, arg=2, pos=59, stackslice=None),
            _PE(op=ops.REDUCE, arg=None, pos=61, stackslice=[g_rs, (g_int,)]),
            _PE(op=ops.BINPUT, arg=3, pos=62, stackslice=None),
            _PE(op=ops.STOP, arg=None, pos=64, stackslice=[[g_rs, (g_int,)]]),
        ],
        maxproto=2,
        stack=[],
        memo={0: g_rs, 1: g_int, 2: (g_int,), 3: [g_rs, (g_int,)]},
    )
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


@pytest.mark.skipif(six.PY2, reason="protocol v3 is not available in Python 2")
def test_reduce_sentinel_list_v3():
    # int isn't special except that it's a globally available builtin that maps
    # to the name int on py2 and py3.
    actual = a._parse(
        dumps(
            [ReduceSentinel(int), ReduceSentinel(True), ReduceSentinel(None)],
            protocol=3,
        )
    )
    g_int = actual.global_objects[("builtins", "int")]
    g_sentinel = actual.global_objects[("tests.test_parse", "ReduceSentinel")]
    expected = _PR(
        parse_entries=[
            _PE(op=ops.PROTO, arg=3, pos=0, stackslice=None),
            _PE(op=ops.EMPTY_LIST, arg=None, pos=2, stackslice=None),
            _PE(op=ops.BINPUT, arg=0, pos=3, stackslice=None),
            _PE(op=ops.MARK, arg=None, pos=5, stackslice=None),
            _PE(
                op=ops.GLOBAL,
                arg="tests.test_parse ReduceSentinel",
                pos=6,
                stackslice=None,
            ),
            _PE(op=ops.BINPUT, arg=1, pos=39, stackslice=None),
            _PE(op=ops.GLOBAL, arg="builtins int", pos=41, stackslice=None),
            _PE(op=ops.BINPUT, arg=2, pos=60, stackslice=None),
            _PE(op=ops.TUPLE1, arg=None, pos=62, stackslice=[g_int]),
            _PE(op=ops.BINPUT, arg=3, pos=63, stackslice=None),
            _PE(
                op=ops.REDUCE,
                arg=None,
                pos=65,
                stackslice=[g_sentinel, (g_int,)],
            ),
            _PE(op=ops.BINPUT, arg=4, pos=66, stackslice=None),
            _PE(op=ops.BINGET, arg=1, pos=68, stackslice=None),
            _PE(op=ops.NEWTRUE, arg=None, pos=70, stackslice=None),
            _PE(op=ops.TUPLE1, arg=None, pos=71, stackslice=[pybool]),
            _PE(op=ops.BINPUT, arg=5, pos=72, stackslice=None),
            _PE(
                op=ops.REDUCE,
                arg=None,
                pos=74,
                stackslice=[g_sentinel, (pybool,)],
            ),
            _PE(op=ops.BINPUT, arg=6, pos=75, stackslice=None),
            _PE(op=ops.BINGET, arg=1, pos=77, stackslice=None),
            _PE(op=ops.NONE, arg=None, pos=79, stackslice=None),
            _PE(op=ops.TUPLE1, arg=None, pos=80, stackslice=[pynone]),
            _PE(op=ops.BINPUT, arg=7, pos=81, stackslice=None),
            _PE(
                op=ops.REDUCE,
                arg=None,
                pos=83,
                stackslice=[g_sentinel, (pynone,)],
            ),
            _PE(op=ops.BINPUT, arg=8, pos=84, stackslice=None),
            _PE(
                op=ops.APPENDS,
                arg=None,
                pos=86,
                stackslice=[
                    [],
                    markobject,
                    [
                        [g_sentinel, (g_int,)],
                        [g_sentinel, (pybool,)],
                        [g_sentinel, (pynone,)],
                    ],
                ],
            ),
            _PE(
                op=ops.STOP,
                arg=None,
                pos=87,
                stackslice=[
                    [
                        [g_sentinel, (g_int,)],
                        [g_sentinel, (pybool,)],
                        [g_sentinel, (pynone,)],
                    ]
                ],
            ),
        ],
        maxproto=2,
        stack=[],
        memo={
            0: [],
            1: g_sentinel,
            2: g_int,
            3: (g_int,),
            4: [g_sentinel, (g_int,)],
            5: (pybool,),
            6: [g_sentinel, (pybool,)],
            7: (pynone,),
            8: [g_sentinel, (pynone,)],
        },
    )
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


class NullReduceEx(object):

    def __reduce_ex__(self, protocol):
        return NullReduceEx, ()


@parametrize_proto(protos=[1, 2, 3])
def test_reduce_ex(proto, maxproto):
    actual = a._parse(dumps(NullReduceEx(), protocol=proto))
    g_nre = actual.global_objects[("tests.test_parse", "NullReduceEx")]
    memoize_ops = _memoize_ops(proto)
    parse_entries = []
    if proto >= 2:
        parse_entries.append(
            _PE(op=ops.PROTO, arg=proto, pos=0, stackslice=None)
        )
    parse_entries.extend(
        [
            _PE(
                op=ops.GLOBAL,
                arg="tests.test_parse NullReduceEx",
                pos=2,
                stackslice=None,
            ),
            next(memoize_ops),
            _PE(op=ops.EMPTY_TUPLE, arg=None, pos=35, stackslice=None),
            _PE(op=ops.REDUCE, arg=None, pos=36, stackslice=[g_nre, ()]),
            next(memoize_ops),
            _PE(op=ops.STOP, arg=None, pos=39, stackslice=[[g_nre, ()]]),
        ]
    )

    expected = _PR(
        parse_entries=parse_entries,
        maxproto=maxproto,
        stack=[],
        memo={0: g_nre, 1: [g_nre, ()]},
    )
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo


def test_reduce_ex_p0():
    actual = a._parse(dumps(NullReduceEx(), protocol=0))
    g_nre = actual.global_objects[("tests.test_parse", "NullReduceEx")]
    memoize_ops = _memoize_ops(0)
    expected = _PR(
        parse_entries=[
            _PE(
                op=ops.GLOBAL,
                arg="tests.test_parse NullReduceEx",
                pos=2,
                stackslice=None,
            ),
            next(memoize_ops),
            _PE(op=ops.MARK, arg=None, pos=26, stackslice=None),
            _PE(op=ops.TUPLE, arg=None, pos=27, stackslice=[markobject, []]),
            _PE(op=ops.REDUCE, arg=None, pos=28, stackslice=[g_nre, ()]),
            next(memoize_ops),
            _PE(op=ops.STOP, arg=None, pos=32, stackslice=[[g_nre, ()]]),
        ],
        maxproto=0,
        stack=[],
        memo={0: g_nre, 1: [g_nre, ()]},
    )
    assert expected.parse_entries == actual.parse_entries
    assert expected.maxproto == actual.maxproto
    assert expected.stack == actual.stack
    assert expected.memo == actual.memo
