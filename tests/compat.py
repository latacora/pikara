import pickle

import pytest
import six

from pikara.analysis import pickled_bool, pickled_int, pickled_int_or_bool


def py2_compat_maxproto(proto):
    """The default protocol in most versions of Python 3 is v3. However, even when
    you ask for v3, as long as you don't pickle any bytes objects, you'll
    actually get a v2 pickle out. This is done to maximize backwards
    compatibility with Python 2.

    This changes with protocol v4. It introduces framing, which is
    non-optional. So if you ask for proto 4, your max proto level in the pickle
    will also be 4.

    """
    if proto == 3:
        return 2
    else:
        return proto


def maxproto_same_as_proto(proto):
    """
    See py2_compat_maxproto.
    """
    return proto


all_protos = range(pickle.HIGHEST_PROTOCOL)


def parametrize_proto(protos=all_protos, maxproto_fn=py2_compat_maxproto):
    # sometimes a test will run on both py2 and py3, so skip the test if it's
    # parametrized for a version python2 doesn't support (3+)
    versions = [
        (proto, maxproto_fn(proto))
        for proto in protos
        if proto <= pickle.HIGHEST_PROTOCOL
    ]
    return pytest.mark.parametrize("proto,maxproto", versions)


def intish_type(proto):
    if six.PY3 or proto > 0:
        return pickled_int
    else:
        return pickled_int_or_bool


def boolish_type(proto):
    if proto > 1:
        return pickled_bool
    else:
        return pickled_int_or_bool
