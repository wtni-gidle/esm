"""End-to-end keyed-row semantics against ESMFold2's native MSA constructor."""

import numpy as np

from esm.esmfold2_wrapper.msa_adapter import split_a3m_to_esmfold2_msa
from esm.models.esmfold2.paired_msa import construct_paired_msa


def test_boltz_row_keys_form_native_paired_rows_across_chains():
    chain_a = split_a3m_to_esmfold2_msa(
        paired_a3m=(
            ">query\nAC\n>pair1\nA-\n>padding\n--\n>pair3\n-C\n"
        ),
        unpaired_a3m=">query\nAC\n>single_a\n-C\n",
        query_sequence="AC",
    )
    chain_b = split_a3m_to_esmfold2_msa(
        paired_a3m=(
            ">query\nDE\n>pair1\nD-\n>only_b\n-E\n>pair3\nDE\n"
        ),
        unpaired_a3m=">query\nDE\n>single_b\n-E\n",
        query_sequence="DE",
    )
    mapping = {"A": 0, "C": 1, "D": 2, "E": 3, "-": 4, "X": 5}

    residues, _, paired = construct_paired_msa(
        chain_msas={0: chain_a, 1: chain_b},
        chain_query_res_types={0: np.asarray([0, 1]), 1: np.asarray([2, 3])},
        token_asym_ids=np.asarray([0, 0, 1, 1]),
        token_res_ids=np.asarray([0, 1, 0, 1]),
        letter_to_res_type=mapping,
    )

    # Query, raw paired row 1 and raw paired row 3 pair across both chains.
    np.testing.assert_array_equal(paired[:3], np.ones((3, 4), dtype=np.float32))
    # Gap encoding is ESMFold2's fixed MSA_GAP_TOKEN_ID (=1), independent of
    # the ordinary residue lookup supplied for this compact fixture.
    np.testing.assert_array_equal(residues[1], np.asarray([0, 1, 2, 1]))
    np.testing.assert_array_equal(residues[2], np.asarray([1, 1, 2, 3]))
    assert np.all(paired[3:] == 0)
