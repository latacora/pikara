import io
import pickle

from pickle import DICT, INT, LIST, MARK, STOP, TUPLE

from pytest import raises

from pikara.analysis import (
    CritiqueException, MissingDictValueException, _Brine, _extract_brine,
    pickled_int, pickled_int_or_bool, pickled_none, pickled_string
)

from .compat import boolish_type, intish_type, parametrize_proto
from .test_critique import proto_op


@parametrize_proto()
def test_unicode_string(proto, maxproto):
    expected = _Brine(shape=pickled_string, maxproto=maxproto)
    actual = _extract_brine(pickle.dumps(u"a", protocol=proto))
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_list_of_three_ints(proto, maxproto):
    intish = intish_type(proto)
    expected = _Brine(shape=[intish, intish, intish], maxproto=maxproto)
    actual = _extract_brine(pickle.dumps([1, 2, 3], protocol=proto))
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_explicit_list_instruction(proto, maxproto):
    instructions = [proto_op(proto), MARK, INT, b"1\n", LIST, STOP]
    pickle = b"".join(instructions)
    # v2 is the first protocol to introduce the PROTO instruction, the other
    # instructions are in every version. proto_op returns b"" for <2
    maxproto = 0 if proto < 2 else 2
    # this is unconditionally a list of a pickled_int_or_bool because it uses
    # the INT instruction.
    expected = _Brine(shape=[pickled_int_or_bool], maxproto=maxproto)
    actual = _extract_brine(pickle)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_explicit_tuple_instruction(proto, maxproto):
    instructions = [proto_op(proto), MARK, INT, b"1\n", TUPLE, STOP]
    pickle = b"".join(instructions)
    # v2 is the first protocol to introduce the PROTO instruction, the other
    # instructions are in every version
    maxproto = 0 if proto < 2 else 2
    expected = _Brine(shape=(pickled_int_or_bool,), maxproto=maxproto)
    actual = _extract_brine(pickle)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_nested_list(proto, maxproto):
    intish = intish_type(proto)
    expected = _Brine(shape=[intish, [intish, [intish]]], maxproto=maxproto)
    actual = _extract_brine(pickle.dumps([3, [2, [1]]], protocol=proto))
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


class NullReduce(object):
    """
    A simple object that uses __reduce__ to pickle itself.
    """

    def __reduce__(self):
        return NullReduce, ()


# TODO: we return the wrong structure for proto 0 (see #13)
# TODO: we don't know how to deal with STACK_GLOBAL in proto 4 (see #12)
@parametrize_proto(protos=[1, 2, 3])
def test_reduce(proto, maxproto):
    actual = _extract_brine(pickle.dumps(NullReduce(), protocol=proto))
    expected = _Brine(
        shape=[actual.global_objects[("tests.test_brine", "NullReduce")], ()],
        maxproto=maxproto,
    )
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


class ReduceSentinel(object):

    def __init__(self, s):
        self.s = s

    def __reduce__(self):
        return ReduceSentinel, (self.s,)


@parametrize_proto()
def test_reduce_sentinel(proto, maxproto):
    # io.BytesIO isn't special except that it isn't rewritten via the
    # _pickle_compat module to enhance compatibility with Python 3. When
    # producing/consuming pickles <= v2, Python 3 knows to translate between
    # the two... sort of.
    actual = _extract_brine(
        pickle.dumps(ReduceSentinel(io.BytesIO), protocol=proto)
    )
    expected = _Brine(
        shape=[
            actual.global_objects[("tests.test_brine", "ReduceSentinel")],
            (actual.global_objects[("_io", "BytesIO")],),
        ],
        maxproto=maxproto,
    )
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_reduce_sentinel_list(proto, maxproto):
    # io.BytesIO isn't special except that it isn't rewritten via the
    # _pickle_compat module to enhance compatibility with Python 3. When
    # producing/consuming pickles <= v2, Python 3 knows to translate between
    # the two... sort of.
    actual = _extract_brine(
        pickle.dumps(
            [
                ReduceSentinel(io.BytesIO),
                ReduceSentinel(True),
                ReduceSentinel(None),
            ],
            protocol=proto,
        )
    )
    expected = _Brine(
        shape=[
            [
                actual.global_objects[("tests.test_brine", "ReduceSentinel")],
                (actual.global_objects[("_io", "BytesIO")],),
            ],
            [
                actual.global_objects[("tests.test_brine", "ReduceSentinel")],
                (boolish_type(proto),),
            ],
            [
                actual.global_objects[("tests.test_brine", "ReduceSentinel")],
                (pickled_none,),
            ],
        ],
        maxproto=maxproto,
    )
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


class NullReduceEx(object):

    def __reduce_ex__(self, protocol):
        return NullReduceEx, ()


# TODO: we return the wrong structure for proto 0 (see #13)
# TODO: we don't know how to deal with STACK_GLOBAL in proto 4 (see #12)
@parametrize_proto(protos=[1, 2, 3])
def test_reduce_ex(proto, maxproto):
    actual = _extract_brine(pickle.dumps(NullReduceEx(), protocol=proto))
    expected = _Brine(
        shape=[
            actual.global_objects[("tests.test_brine", "NullReduceEx")], ()
        ],
        maxproto=maxproto,
    )
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_empty_dict(proto, maxproto):
    """An empty dict is special because protocols >= 1 introduce a special
    EMPTY_DICT opcode. Protocol version zero uses DICT with an empty
    stackslice.
    """
    actual = _extract_brine(pickle.dumps({}, protocol=proto))
    expected = _Brine(shape={}, maxproto=maxproto)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_single_item_dict(proto, maxproto):
    """
    A single item dict is different from an empty dict in structure. Protocol 0
    technically has a DICT instruction and a SETITEM instruction. DICT takes a
    stack slice of even length (MARK K1 V1 K2 V2... DICT). SETITEM takes a
    2-element stack slice. I can't get the legitimate pickle VM to produce
    anything other than MARK DICT (effectively a an empty dictionary) and then
    SETITEM'ing individual items on it, though.
    """
    actual = _extract_brine(pickle.dumps({1: 2}, protocol=proto))
    expected = _Brine(shape={1: pickled_int}, maxproto=maxproto)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_explicit_stackslice_single_item_dict(proto, maxproto):
    instructions = [
        proto_op(proto), MARK, INT, b"1\n", INT, b"2\n", DICT, STOP
    ]
    pickle = b"".join(instructions)
    # v2 is the first protocol to introduce the PROTO instruction, the other
    # instructions are in every version. proto_op returns b"" for <2
    maxproto = 0 if proto < 2 else 2
    # this is unconditionally a list of a pickled_int_or_bool because it uses
    # the INT instruction.
    expected = _Brine(shape={1: pickled_int_or_bool}, maxproto=maxproto)
    actual = _extract_brine(pickle)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_explicit_stackslice_multi_item_dict(proto, maxproto):
    instructions = [
        proto_op(proto),
        MARK,
        INT,
        b"1\n",
        INT,
        b"2\n",
        INT,
        b"3\n",
        INT,
        b"4\n",
        DICT,
        STOP,
    ]
    pickle = b"".join(instructions)
    # v2 is the first protocol to introduce the PROTO instruction, the other
    # instructions are in every version. proto_op returns b"" for <2
    maxproto = 0 if proto < 2 else 2
    # this is unconditionally a list of a pickled_int_or_bool because it uses
    # the INT instruction.
    expected = _Brine(
        shape={1: pickled_int_or_bool, 3: pickled_int_or_bool},
        maxproto=maxproto,
    )
    actual = _extract_brine(pickle)
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_multi_item_dict(proto, maxproto):
    actual = _extract_brine(pickle.dumps({1: 2, 3: 4}, protocol=proto))
    expected = _Brine(
        shape={1: pickled_int, 3: pickled_int}, maxproto=maxproto
    )
    assert expected.shape == actual.shape
    assert expected.maxproto == actual.maxproto


@parametrize_proto()
def test_explicit_stackslice_missing_dict_value(proto, maxproto):
    pickle = b"".join([proto_op(proto), MARK, INT, b"1\n", DICT, STOP])
    with raises(CritiqueException) as excinfo:
        _extract_brine(pickle)
    ce = excinfo.value
    assert len(ce.issues) == 1
    de = excinfo.value.issues[0]
    assert isinstance(de, MissingDictValueException)
    assert de.kvlist == [1]
