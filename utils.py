import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    matthews_corrcoef,
)


def callback_get_label(dataset, idx):
    if dataset.data[idx]["n_bubbles"] > 0:
        return 1
    else:
        return 0


def get_best_span(
    span_start_logits: torch.Tensor, span_end_logits: torch.Tensor
) -> torch.Tensor:
    """
    This acts the same as the static method ``BidirectionalAttentionFlow.get_best_span()``
    in ``allennlp/models/reading_comprehension/bidaf.py``. We keep it here so that users can
    directly import this function without the class.
    We call the inputs "logits" - they could either be unnormalized logits or normalized log
    probabilities.  A log_softmax operation is a constant shifting of the entire logit
    vector, so taking an argmax over either one gives the same result.


    From: https://github.com/allenai/allennlp-models/blob/f233052df9feb03f636007dd25c0a3b8d4b546d6/allennlp_models/rc/models/utils.py#L6
    """
    if span_start_logits.dim() != 2 or span_end_logits.dim() != 2:
        raise ValueError("Input shapes must be (batch_size, passage_length)")
    batch_size, passage_length = span_start_logits.size()
    device = span_start_logits.device
    # (batch_size, passage_length, passage_length)
    span_log_probs = span_start_logits.unsqueeze(
        2) + span_end_logits.unsqueeze(1)
    # Only the upper triangle of the span matrix is valid; the lower triangle has entries where
    # the span ends before it starts.
    span_log_mask = torch.triu(
        torch.ones((passage_length, passage_length), device=device)
    ).log()
    valid_span_log_probs = span_log_probs + span_log_mask

    # Here we take the span matrix and flatten it, then find the best span using argmax.  We
    # can recover the start and end indices from this flattened list using simple modular
    # arithmetic.
    # (batch_size, passage_length * passage_length)
    _, best_spans = torch.topk(valid_span_log_probs.view(batch_size, -1), 8)
    span_start_indices = torch.floor_divide(best_spans, passage_length)
    span_end_indices = best_spans % passage_length
    return torch.stack([span_start_indices, span_end_indices], dim=-1)


def do_nms(span_idxs, overlapThresh=2):
    """
    span_idx = (k, 2)
    """
    # span_idxs = span_idxs.numpy()
    pick = []

    x = span_idxs[:, 0]
    y = span_idxs[:, 1]

    k = span_idxs.shape[0]
    idx = np.arange(0, k)
    idx = idx[::-1]

    area = y - x + 1
    while len(idx) > 0:
        last = len(idx) - 1
        i = idx[last]
        pick.append(i)

        xx = np.minimum(x[i], x[idx[:last]])
        yy = np.maximum(y[i], y[idx[:last]])

        span_i = np.maximum(0, yy - xx + 1)
        overlap = span_i / area[idx[:last]]

        idx = np.delete(
            idx, np.concatenate(([last], np.where(overlap > overlapThresh)[0]))
        )

    return span_idxs[pick].astype(int)


def summarize_results(
    true_bubble_list,
    start_idx_pred_list,
    end_idx_pred_list,
    num_bubble_true_list,
    num_bubble_pred_list,
    return_pred_bubble=False,
):

    true_bubble_list = torch.cat(
        true_bubble_list, dim=0).cpu()  # shape = (N, num_days)

    pstart_idx_pred_list = torch.log(
        torch.cat(start_idx_pred_list, dim=0).detach().cpu()
    )
    pend_idx_pred_list = torch.log(
        torch.cat(end_idx_pred_list, dim=0).detach().cpu())

    num_bubble_true_list = torch.cat(
        num_bubble_true_list, dim=0
    ).cpu()  # shape = (bs, max_bubbles)

    num_bubble_pred_list = torch.cat(
        num_bubble_pred_list, dim=0).detach().cpu()
    num_bubble_pred_list = torch.argmax(num_bubble_pred_list, dim=-1)

    best_span = (
        get_best_span(pstart_idx_pred_list,
                      pend_idx_pred_list).int().cpu().numpy()
    )
    bs, n_days = pstart_idx_pred_list.shape
    span_pred_list = []
    for i in range(bs):
        # print(i)
        n_spans = num_bubble_pred_list[i].item()
        predicted = torch.zeros(n_days, dtype=int)
        if n_spans != 0:
            current_idx = do_nms(best_span[i])
            # print(current_idx)
            # print(n_spans)
            if current_idx.shape[0] < n_spans:
                current_idx = best_span[i]
            # print(current_best_span.shape)
            # current_idx = current_best_span[:n_spans]
            mask = []
            # print(n_spans)
            # print(current_idx)
            for j in range(n_spans):
                mask += list(np.arange(current_idx[j, 0], current_idx[j, 1]+1))
            # print(mask)
            predicted[mask] = 1
        span_pred_list.append(predicted)
    # print(span_pred_list)
    span_pred_list = torch.stack(span_pred_list, dim=0)
    em = get_EM(span_pred_list, true_bubble_list)
    em_bubble_only = get_EM_bubble_only(span_pred_list, true_bubble_list, num_bubble_true_list)
    acc_nbubble = get_accuracy(num_bubble_pred_list, num_bubble_true_list)
    precision_nbubble, recall_nbubble, f1_nbubble, _ = get_f1(
        num_bubble_pred_list, num_bubble_true_list
    )
    acc_span = get_accuracy(span_pred_list, true_bubble_list)
    precision_span, recall_span, f1_span, _ = get_f1(
        span_pred_list, true_bubble_list)

    span_pred_list = span_pred_list.view(-1).numpy()
    true_bubble_list = true_bubble_list.view(-1).numpy()
    # print(classification_report(true_bubble_list, span_pred_list))
    mcc = matthews_corrcoef(true_bubble_list, span_pred_list)
    return {
        "EM": em,
        "EM_only_bubble": em_bubble_only,
        "acc_span": acc_span,
        "acc_bubble": acc_nbubble,
        "precision_span": precision_span,
        "recall_span": recall_span,
        "f1_span": f1_span,
        "precision_nbubble": precision_nbubble,
        "recall_nbubble": recall_nbubble,
        "f1_nbubble": f1_nbubble,
        "MCC": mcc,
        "true_bubble_list": true_bubble_list.reshape(bs, n_days),
        "pred_bubble_list": span_pred_list.reshape(bs, n_days),
        "num_bubble_pred_list": num_bubble_pred_list,
        "num_bubble_true_list":  num_bubble_true_list

    }


def get_accuracy(pred, true):
    """
    pred: bs, label
    true: bs, label
    """
    pred = pred.view(-1)
    true = true.view(-1)
    out = torch.mean((pred == true).float())
    return out


def get_f1(pred, true):
    pred = pred.view(-1).numpy()
    true = true.view(-1).numpy()
    return precision_recall_fscore_support(true, pred, average="macro")


def get_EM(pred, true):
    bs, _ = pred.shape
    out = (pred == true).int()
    return torch.true_divide(torch.sum(torch.prod(out, 1)), bs)


def get_EM_bubble_only(pred, true, n_bubbles):
    indices = ((n_bubbles > 0).nonzero(as_tuple=True)[0]).tolist()
    pred = pred[indices]
    true = true[indices]
    out = (pred == true).int()
    if len(indices) !=0:
        return torch.true_divide(torch.sum(torch.prod(out, 1)), len(indices)).item()
    else:
        return 0
