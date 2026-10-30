"""Regression tests: padded document tokens must be excluded from MaxSim.

Before the fix, ``documents_mask`` was applied by multiplication, which
floored every query-token max at 0: whenever a query token's real
similarities to a document were all negative, the padded (zero) slot won
the max instead of the true maximum. Padding then changed the score and
could even reorder a ranking.
"""

from __future__ import annotations

import torch

from pylate.rank import rerank
from pylate.scores import colbert_kd_scores, colbert_scores


class TestMaskedPaddingExclusion:
    def test_padded_score_equals_dense_score(self) -> None:
        """Adding masked padding tokens must not change the score."""
        q = torch.tensor([[[-1.0], [-1.0]]])
        doc_real = torch.tensor([[[1.0], [1.0]]])
        doc_padded = torch.tensor([[[1.0], [1.0], [0.0]]])
        doc_mask = torch.tensor([[1.0, 1.0, 0.0]])

        dense = colbert_scores(q, doc_real)
        padded = colbert_scores(q, doc_padded, documents_mask=doc_mask)

        torch.testing.assert_close(padded, dense)

    def test_no_mask_call_unchanged(self) -> None:
        q = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]])
        d = torch.tensor([[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.5]]])
        torch.testing.assert_close(colbert_scores(q, d), torch.tensor([[3.0, 3.0]]))

    def test_kd_padded_score_equals_dense(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        docs_real = torch.tensor([[[[1.0], [1.0]], [[1.0], [1.0]]]])
        docs_padded = torch.tensor([[[[1.0], [1.0], [0.0]], [[1.0], [1.0], [0.0]]]])
        doc_mask = torch.tensor([[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]])

        dense = colbert_kd_scores(q, docs_real)
        padded = colbert_kd_scores(q, docs_padded, documents_mask=doc_mask)

        torch.testing.assert_close(padded, dense)

    def test_rerank_unequal_lengths_are_not_reordered_by_padding(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        # docA has 2 real tokens, docB has 1 real token (gets one pad slot).
        results = rerank(
            documents_ids=[["A", "B"]],
            queries_embeddings=q,
            documents_embeddings=[
                [torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0]])],
            ],
        )
        by_id = {r["id"]: r["score"] for r in results[0]}
        assert by_id["A"] == by_id["B"]

    def test_padded_query_token_contributes_zero_no_nan(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        q_mask = torch.tensor([[1.0, 0.0]])
        doc = torch.tensor([[[1.0], [0.0]]])
        doc_mask = torch.tensor([[1.0, 0.0]])

        out = colbert_scores(q, doc, queries_mask=q_mask, documents_mask=doc_mask)

        assert not torch.isnan(out).any()
        torch.testing.assert_close(out, torch.tensor([[-1.0]]))

    def test_fully_invalid_doc_row_is_inf_not_nan(self) -> None:
        q = torch.tensor([[[-1.0], [-1.0]]])
        doc = torch.tensor([[[1.0], [1.0]]])
        doc_mask = torch.tensor([[0.0, 0.0]])

        out = colbert_scores(q, doc, documents_mask=doc_mask)

        assert not torch.isnan(out).any()
        assert torch.isinf(out).all()
