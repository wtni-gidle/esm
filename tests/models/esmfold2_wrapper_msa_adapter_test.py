"""Tests for Boltz-component A3M adaptation to ESMFold2 native MSAs."""

import numpy as np
import pytest

from esm.esmfold2_wrapper.msa_adapter import (
    PreparedMSAError,
    native_a3m_to_esmfold2_msa,
    split_a3m_to_esmfold2_msa,
    split_a3m_to_keyed_text,
)


def test_split_adapter_uses_original_paired_row_numbers_and_deduplicates_query():
    paired = ">query old key=99\nACDE\n>hit1\nAC-E\n>padding\n----\n>hit3\nA-DE\n"
    unpaired = ">query\nACDE\n>single1\nACdDE\n>single2\nA--E\n"

    keyed = split_a3m_to_keyed_text(
        paired_a3m=paired, unpaired_a3m=unpaired, query_sequence="ACDE"
    )

    assert keyed == (
        ">paired_row_0 key=0\nACDE\n"
        ">paired_row_1 key=1\nAC-E\n"
        ">paired_row_3 key=3\nA-DE\n"
        ">unpaired_row_1 key=-1\nACdDE\n"
        ">unpaired_row_2 key=-1\nA--E\n"
    )


def test_split_adapter_preserves_insertion_deletion_counts():
    msa = split_a3m_to_esmfold2_msa(
        paired_a3m=">query\nACDE\n>paired\nAC-E\n",
        unpaired_a3m=">query\nACDE\n>inserted\nACdDE\n",
        query_sequence="ACDE",
    )

    assert msa.sequences == ["ACDE", "AC-E", "ACDE"]
    assert msa.headers == [
        "paired_row_0 key=0",
        "paired_row_1 key=1",
        "unpaired_row_1 key=-1",
    ]
    np.testing.assert_array_equal(
        msa.deletions,
        np.asarray([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float32),
    )


def test_empty_paired_keeps_the_unpaired_query_first():
    msa = split_a3m_to_esmfold2_msa(
        paired_a3m="",
        unpaired_a3m=">query\nACDE\n>hit\nA-DE\n",
        query_sequence="ACDE",
    )

    assert msa.headers == ["unpaired_row_0 key=-1", "unpaired_row_1 key=-1"]
    assert msa.query == "ACDE"


def test_native_adapter_preserves_existing_keys_and_deletions():
    msa = native_a3m_to_esmfold2_msa(
        a3m=">query\nACDE\n>paired key=7\nACdDE\n",
        query_sequence="ACDE",
    )

    assert msa.headers[1] == "paired key=7"
    assert msa.sequences[1] == "ACDE"
    assert msa.deletions[1, 2] == 1


@pytest.mark.parametrize("header", ["bad key=-2", "bad key=1 key=2"])
def test_native_adapter_rejects_ambiguous_or_invalid_keys(header):
    with pytest.raises(PreparedMSAError, match="invalid key|more than one"):
        native_a3m_to_esmfold2_msa(
            a3m=f">query\nACDE\n>{header}\nACDE\n",
            query_sequence="ACDE",
        )


@pytest.mark.parametrize(
    "paired,unpaired,message",
    [
        ("", "", "unpairedMsa"),
        (">q\nAAAA\n", ">q\nACDE\n", "pairedMsa query"),
        (">q\nACDE\n>bad\nACD\n", ">q\nACDE\n", "aligned width 3"),
        ("", ">q\nACDE\n>bad\nAC*E\n", "invalid A3M characters"),
    ],
)
def test_split_adapter_rejects_invalid_components(paired, unpaired, message):
    with pytest.raises(PreparedMSAError, match=message):
        split_a3m_to_esmfold2_msa(
            paired_a3m=paired,
            unpaired_a3m=unpaired,
            query_sequence="ACDE",
        )
