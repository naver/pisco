import torch


def colbert_score_single(q, d):
    """
    'Colbert' scores between a two lists of embeddings
    This score is maxed over doc dim, averaged over q dim.
    q of shape (Nq, dim)
    d of shape (Nd, dim)
    """
    # Pairwise similarities: (N_qk, N_dki)
    sim = q @ d.transpose(-1, -2)

    # Late interaction:
    # For each query token, take max over document tokens → (N_q,)
    max_per_q = sim.max(dim=-1).values

    # Aggregate over query tokens → scalar
    return max_per_q.max()


def colbert_scores_training(queries, docs):
    """
    Compute ColBERT late-interaction scores between queries and their associated documents.
    THIS ASSSUMES THAT THERE ARE NB_DOC_PER_QUERY DOCS PER QUERY => THE TRAINING SETUP.
    ----------
    docs : list of torch.Tensor
        List of length (batch_size * n_doc_per_query).
        docs[k * n_doc_per_query + i] has shape (N_dki, dim),
        where:
            - k indexes the query in the batch
            - i indexes the document associated with that query
            - N_dki is the number of tokens in that document
            - dim is the embedding dimension

    queries : list of torch.Tensor
        List of length batch_size.
        queries[k] has shape (N_qk, dim), where:
            - N_qk is the number of tokens in query k

    Returns
    -------
    scores : torch.Tensor
        Tensor of shape (batch_size, n_doc_per_query)
        containing the ColBERT score between each query and its documents.

        score(q, d) = mean_t( max_j( cosine(q_t, d_j) ) )

    """
    batch_size = len(queries)
    n_doc_per_query = len(docs) // batch_size
    assert len(docs) == batch_size * n_doc_per_query, f"{len(docs)}, {len(queries)}"

    device = queries[0].device
    dtype = queries[0].dtype
    scores = torch.empty((batch_size, n_doc_per_query), device=device, dtype=dtype)

    for k in range(batch_size):
        q = queries[k]  # (N_qk, dim)

        for i in range(n_doc_per_query):
            d = docs[k * n_doc_per_query + i]  # (N_dki, dim)

            scores[k, i] = colbert_score_single(q, d)

    return scores


def colbert_scores_pretraining(docs):
    """
    Compute ColBERT late-interaction scores between queries and their associated documents.
    compute all pairwise similarities, colbert-style.
    """
    device = docs[0].device
    dtype = docs[0].dtype

    n_docs = len(docs)
    scores = torch.empty((n_docs, n_docs), device=device, dtype=dtype)

    for k in range(n_docs):
        scores[k, k] = 1.0
        for i in range(k):
            scores[k, i] = colbert_score_single(docs[k], docs[i])
            scores[i, k] = scores[k, i]

    return scores
