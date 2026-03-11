import torch
import numpy as np
import torch.distributed as dist

from transformers import Trainer

from pisco.eval_utils import build_ir_eval_from_splare_dataset
from pisco.colbert_utils import colbert_scores_training, colbert_scores_pretraining


class ContrastiveLoss:
    """
    InfoNCE loss
    assumes the positive is the first element
    """

    def __init__(self, temperature=1.0, weight=1.0):
        self.init_temperature = temperature
        self.temperature = temperature
        self.weight = weight
        self.loss = torch.nn.CrossEntropyLoss()  # default reduction is `mean`

    def set_temperature(self, step, t_max):
        # fraction of decay completed, capped at 1.0
        frac = min(step / t_max, 1.0)
        # linearly interpolate between self.init_temperature and 1.0
        self.temperature = self.init_temperature * (1.0 - frac) + frac * 1.0

    def __call__(self, scores):
        scores = scores / self.temperature
        labels_index = torch.zeros(scores.size(0)).to(scores.device).long()
        # => distrib is (s+, s-_1, s-_2, ...), so we give the index of positive (first position)
        return self.weight * self.loss(scores, labels_index)


class KLDiv:
    """
    copy-pasted from splare.
    """

    def __init__(self, temperature=1.0, weight=1.0, teacher_temperature=1.0):
        self.temperature = temperature
        self.teacher_temperature = teacher_temperature
        self.weight = weight
        self.loss = torch.nn.KLDivLoss(reduction="batchmean")

    def __call__(self, teacher_scores, scores):
        assert teacher_scores.shape == scores.shape, "shapes must match"
        student_scores = torch.log_softmax(scores / self.temperature, dim=1)
        teacher_scores = teacher_scores.to(scores.device)
        with torch.no_grad():
            teacher_scores = torch.softmax(
                teacher_scores / self.teacher_temperature, dim=1
            )
        loss = (
            self.loss(student_scores, teacher_scores)
            * self.temperature
            * self.teacher_temperature
        )

        return self.weight * loss


class RetrievalPiscoTrainer(Trainer):

    def __init__(
        self,
        contrastive_weight: float = 0.0,
        contrastive_temperature: float = 1.0,
        kl_weight: float = 0.0,
        kl_temperature: float = 1.0,
        generation_weight: float = 1.0,
        eval_d_collection: str = None,
        eval_q_collection: str = None,
        data_local_cache: str = None,
        eval_batch_size: int = None,
        *args,
        **kwargs,
    ):
        self.contrastive_loss = ContrastiveLoss(
            temperature=contrastive_temperature, weight=contrastive_weight
        )
        self.kldiv_loss = KLDiv(temperature=kl_temperature, weight=kl_weight)

        # If one of these weights is non-zero, we also train for retrieval
        self.train_for_retrieval = contrastive_weight > 0 or kl_weight > 0

        self.generation_weight = generation_weight

        if self.train_for_retrieval:
            self.ir_evaluator = build_ir_eval_from_splare_dataset(
                d_collection=eval_d_collection,
                q_collection=eval_q_collection,
                data_local_cache=data_local_cache,
                batch_size=eval_batch_size,
            )
        self.additional_losses = {
            "contrastive_loss": [],
            "kl_loss": [],
            "generation_loss": [],
        }

        super().__init__(*args, **kwargs)

    def pretraining_retrieval_loss(
        self,
        d_embeddings,
        compressed_doc_numbers,
        nb_max_negs=8,
        retrieval_pooling="first",
    ):
        """
        For each document number i:
        - sample two documents having that doc number: they are the positive pair
        - sample up to nb_negs negatives k with m[k] != i
        - scores = [cos(i,j)] + [cos(i,k1), ..., cos(i,kt)]
        - loss = average over anchors that have at least one positive and one negative
        """
        N = len(d_embeddings)

        # Precompute full sim matrix once (simple, usually fine for moderate N)
        if retrieval_pooling == "colbert":
            sims = colbert_scores_pretraining(d_embeddings)
        else:
            sims = d_embeddings @ d_embeddings.t()  # (N, N)

        device = sims.device

        total_loss = torch.tensor(0.0)
        count = 0

        max_number = compressed_doc_numbers.max().item()

        for doc_number in range(max_number + 1):

            # Collecting indices of documents wich number == i
            pos_indices = torch.arange(N, device=device)[
                compressed_doc_numbers == doc_number
            ]

            # If 0 or 1 doc, we cannot use NCE
            if pos_indices.size(0) <= 1:
                continue

            # Index of the positive pair within d_embeddings:
            pos_pair = pos_indices[
                torch.randperm(pos_indices.size(0), device=device)[:2]
            ]

            # ---- sample up to nb_negs negatives (different label) ----
            neg_idx = torch.arange(N, device=device)[
                compressed_doc_numbers != doc_number
            ]

            # No negative (e.g. batch_size = 1)
            if neg_idx.numel() == 0:
                raise ValueError(
                    "WARNING, you cannot use contrastive loss in pretraining with a batch size of 1."
                )

            if nb_max_negs is not None and neg_idx.numel() > nb_max_negs:
                perm = torch.randperm(neg_idx.numel(), device=device)[:nb_max_negs]
                neg_idx = neg_idx[perm]

            # Build scores: positive first
            pos_score = sims[pos_pair[0], pos_pair[1]].unsqueeze(0)  # (1,)
            neg_scores = sims[pos_pair[0], neg_idx]  # (K,)
            scores = torch.cat([pos_score, neg_scores], dim=0)

            total_loss = total_loss + self.contrastive_loss(scores=scores.unsqueeze(0))
            count += 1

        if count == 0:
            return total_loss  # = 0 scalar

        return total_loss / count

    def compute_retrieval_loss(
        self, model_output: dict, model, teacher_scores, compressed_doc_numbers
    ):
        contrastive_loss = torch.tensor(0.0)
        kl_loss = torch.tensor(0.0)

        m = (
            model.module
            if isinstance(model, torch.nn.parallel.DistributedDataParallel)
            else model
        )
        retrieval_pooling = m.config.retrieval_pooling

        # pretraining mode: we don't have 'q_embedding'
        if (
            "d_embedding" in model_output
            and (self.contrastive_loss.weight > 0.0 or self.kldiv_loss.weight > 0.0)
            and "q_embedding" not in model_output
        ):
            d_emb = model_output["d_embedding"]  # (N, dim)

            # assert d_emb.size(0) == compressed_doc_numbers.size(0)
            contrastive_loss = self.pretraining_retrieval_loss(
                d_emb, compressed_doc_numbers, retrieval_pooling=retrieval_pooling
            )

        # classic fine-tuning mode
        elif (
            "d_embedding" in model_output
            and "q_embedding" in model_output
            and (self.contrastive_loss.weight > 0.0 or self.kldiv_loss.weight > 0.0)
        ):
            d_emb = model_output["d_embedding"]  # (bs*nd, dim)
            q_emb = model_output["q_embedding"]  # (bs, dim)

            if retrieval_pooling == "colbert":
                scores = colbert_scores_training(q_emb, d_emb)

            else:
                nd = d_emb.size(0) // q_emb.size(0)

                d_emb = d_emb.view(q_emb.size(0), nd, d_emb.size(-1))  # bs, nd, dim

                # cosine sim: (bs, nd)
                scores = torch.einsum("bd,bnd->bn", q_emb, d_emb)

            if self.contrastive_loss.weight > 0:
                contrastive_loss = self.contrastive_loss(scores=scores)

            if self.kldiv_loss.weight > 0:
                kl_loss = self.kldiv_loss(teacher_scores=teacher_scores, scores=scores)

        return contrastive_loss, kl_loss

    def get_predictions_training(self, model, inputs):
        teacher_scores = inputs.pop("scores", None)

        # if this is provided ,then we are in pretraining mode.
        compressed_doc_numbers = inputs.pop("compressed_doc_numbers", None)

        if self.generation_weight == 0.0:
            # Removing these to save computations
            # We could do it in the collator but it's complicated enough
            inputs.pop("decoder_input_ids")
            inputs.pop("decoder_attention_mask")

        model_output = model(**inputs)

        generation_loss = torch.tensor(0.0)
        if "loss" in model_output:
            generation_loss = self.generation_weight * model_output["loss"]

        output = {"generation_loss": generation_loss}

        contrastive_loss, kl_loss = self.compute_retrieval_loss(
            model_output,
            model,
            teacher_scores=teacher_scores,
            compressed_doc_numbers=compressed_doc_numbers,
        )

        output["contrastive_loss"] = contrastive_loss
        output["kl_loss"] = kl_loss

        output["loss"] = generation_loss + contrastive_loss + kl_loss
        output["logits"] = model_output.get("logits", None)

        return output

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        inputs = self._prepare_inputs(
            inputs
        )  # not 100% sure if needed, but might be for DDP training
        loss_dict = self.get_predictions_training(model, inputs)
        return loss_dict["loss"]  # TODO should we return sthing else ?

    def prediction_step(
        self, model, inputs, prediction_loss_only=None, ignore_keys=None, **kwargs
    ):
        model.eval()
        with torch.no_grad():
            inputs = self._prepare_inputs(inputs)
            loss_dict = self.get_predictions_training(model, inputs)
            # it is not direct to record other losses than the "main" one
            # see for instance ==> https://github.com/zipzou/hf-multitask-trainer

            for k in self.additional_losses.keys():
                if k in loss_dict:
                    self.additional_losses[k].append(loss_dict[k].item())

        if prediction_loss_only:
            return loss_dict["loss"], None, None
        else:
            return loss_dict["loss"], loss_dict["logits"], inputs.get("labels")

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        """
        overrides the default evaluation to include custom test set evaluation at each eval step
        """
        self.model.eval()
        eval_metrics = super().evaluate(
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )
        # adding custom losses also
        for k in self.additional_losses.keys():
            eval_metrics[f"{metric_key_prefix}_{k}"] = np.mean(
                self.additional_losses[k]
            ).item()
            # then re-init the losse:
            self.additional_losses[k] = []

        # Retrieval evaluation:
        if (
            not dist.is_available() or not dist.is_initialized() or dist.get_rank() == 0
        ) and self.train_for_retrieval:
            # Now we evaluate retrieval on NFCorpus or such
            # TODO: we should also do some minimal accuracy evaluation for QA
            with torch.no_grad():
                results = self.ir_evaluator(self.model)
            results = {f"{metric_key_prefix}_{k}": v for k, v in results.items()}
            print("Evaluation results")

            eval_metrics.update(results)

        self.model.train()
        self.log(eval_metrics)
        return eval_metrics
