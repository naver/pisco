from sentence_transformers.evaluation import InformationRetrievalEvaluator
from collections import defaultdict


def build_ir_eval_from_data(
    document_collection: list[dict],
    query_collection: list[dict],
    relevant_docs: dict,
    batch_size: int,
    **kwargs,
):
    """
    Builds the InformationRetrievalEvaluator, ensuring all documents and queries
    are correctly formatted with a guaranteed eos_token at the end.
    document_collection contains lists of dicts containing 'id', 'content'
    query_collection contains lists of dicts containing 'id', 'content'
    q_rels is a dictionary mapping q_ids to its positive list of doc ids.
    """
    print("Preparing corpus and queries for ir evaluator")
    corpus = {
        str(row["id"]): (row.get("content") or "").strip()
        for row in document_collection
    }
    queries = {
        str(row["id"]): (row.get("content") or "").strip() for row in query_collection
    }

    evaluator = InformationRetrievalEvaluator(
        queries=queries,
        corpus=corpus,
        relevant_docs=relevant_docs,
        batch_size=batch_size,
        mrr_at_k=[10, 100],
        ndcg_at_k=[10, 100],
        precision_recall_at_k=[10],
        show_progress_bar=True,
        name=kwargs.pop("name", "val_ir"),
        **kwargs,
    )
    return evaluator


def build_ir_eval_from_splare_dataset(
    d_collection: str,
    q_collection: str,
    data_local_cache: str,
    batch_size: int,
    **kwargs,
):
    """
    Builds the InformationRetrievalEvaluator, ensuring all documents and queries
    are correctly formatted with a guaranteed eos_token at the end.
    """
    # TODO: remove this splare dependency here:
    from modules.dataset import DocumentDataset, QueryDataset

    document_collection = DocumentDataset.get_hf_dataset(d_collection, data_local_cache)
    query_collection = QueryDataset.get_hf_dataset(q_collection, data_local_cache)
    qrels = QueryDataset.get_qrels(q_collection)

    relevant_docs = defaultdict(set)
    for r in qrels:
        if getattr(r, "relevance", 0) and int(r.relevance) > 0:
            relevant_docs[str(r.query_id)].add(str(r.doc_id))

    return build_ir_eval_from_data(
        document_collection=document_collection,
        query_collection=query_collection,
        relevant_docs=relevant_docs,
        batch_size=batch_size,
        **kwargs,
    )
