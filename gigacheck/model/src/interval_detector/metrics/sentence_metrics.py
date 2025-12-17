from typing import Dict, List


class SentenceMetrics:
    def __init__(self):
        self.sentence_acc_correct = 0
        self.soft_acc_correct = 0
        self.raw_sentence_acc_correct = 0
        self.raw_soft_acc_correct = 0

        self.mse_sum = 0
        self.raw_mse_sum = 0
        self.num_intervals = 0

    def update(self, preds, gt_spans, text, sep_indices):
        if not sep_indices:
            return
        pred = sorted(preds, key=lambda x: x[2], reverse=True)[0]
        pred_start = int(pred[0].item())

        pred_sentence_idx_start = find_index(sep_indices, pred_start)
        sentence_start_pos = find_start_pos(sep_indices, pred_sentence_idx_start)

        sentence_length = find_length(sep_indices, pred_sentence_idx_start, text)
        pred_sentence_idx_start_processed = (
            min(pred_sentence_idx_start + 1, len(sep_indices))
            if pred_start >= sentence_start_pos + sentence_length / 2
            else pred_sentence_idx_start
        )

        if gt_spans.numel():
            gt = gt_spans
            gt_start = int(gt[0, 0].item())

            gt_sentence_idx_start = find_index(sep_indices, gt_start)

            if pred_sentence_idx_start_processed == gt_sentence_idx_start:
                self.sentence_acc_correct += 1
            if abs(pred_sentence_idx_start_processed - gt_sentence_idx_start) <= 1:
                self.soft_acc_correct += 1
            self.mse_sum += (pred_sentence_idx_start_processed - gt_sentence_idx_start) ** 2

            if pred_sentence_idx_start == gt_sentence_idx_start:
                self.raw_sentence_acc_correct += 1
            if abs(pred_sentence_idx_start - gt_sentence_idx_start) <= 1:
                self.raw_soft_acc_correct += 1
            self.raw_mse_sum += (pred_sentence_idx_start - gt_sentence_idx_start) ** 2

        else:
            if pred[2] < 0.5:
                self.sentence_acc_correct += 1
                self.soft_acc_correct += 1
                self.raw_sentence_acc_correct += 1
                self.raw_soft_acc_correct += 1
            else:
                self.mse_sum += (pred_sentence_idx_start_processed - len(sep_indices) - 1) ** 2
                self.raw_mse_sum += (pred_sentence_idx_start - len(sep_indices) - 1) ** 2
        self.num_intervals += 1

    def compute(self) -> List[float]:
        sentence_acc = self.sentence_acc_correct / self.num_intervals if self.num_intervals != 0 else 0
        soft_acc = self.soft_acc_correct / self.num_intervals if self.num_intervals != 0 else 0
        mse = self.mse_sum / self.num_intervals if self.num_intervals != 0 else 0

        raw_sentence_acc = self.raw_sentence_acc_correct / self.num_intervals if self.num_intervals != 0 else 0
        raw_soft_acc = self.raw_soft_acc_correct / self.num_intervals if self.num_intervals != 0 else 0
        raw_mse = self.raw_mse_sum / self.num_intervals if self.num_intervals != 0 else 0

        return [sentence_acc, soft_acc, mse, raw_sentence_acc, raw_soft_acc, raw_mse]

    def to_dict(self, desc: str) -> Dict[str, float]:
        sentence_acc, soft_acc, mse, raw_sentence_acc, raw_soft_acc, raw_mse = self.compute()
        return {
            f"{desc}_sentence_acc": sentence_acc,
            f"{desc}_soft_acc": soft_acc,
            f"{desc}_mse": mse,
            f"{desc}_raw_sentence_acc": raw_sentence_acc,
            f"{desc}_raw_soft_acc": raw_soft_acc,
            f"{desc}_raw_mse": raw_mse,
        }

    def log(self) -> str:
        sentence_acc, soft_acc, mse, raw_sentence_acc, raw_soft_acc, raw_mse = self.compute()
        log_str = (
            f" sentence_acc: {sentence_acc:.2f}; soft_acc: {soft_acc:.2f}; mse: {mse:.2f}; "
            f"raw_sentence_acc: {raw_sentence_acc}; raw_soft_acc: {raw_soft_acc}; raw_mse: {raw_mse};"
        )
        return log_str


def find_index(sep_indices, char_pos):
    # In the case of only 1 sentence in the text
    if not sep_indices:
        return 0
    for i, sep_index in enumerate(sorted(sep_indices)):
        if char_pos < sep_index:
            return i
    return len(sep_indices)


def find_start_pos(sep_indices, pred_idx_start):
    if pred_idx_start == 0:
        return 0
    return sep_indices[pred_idx_start - 1] + 1


def find_length(sep_indices, pred_idx_start, text):
    if pred_idx_start == 0:
        start_index = 0
    else:
        start_index = sep_indices[pred_idx_start - 1] + 1

    if pred_idx_start < len(sep_indices):
        end_index = sep_indices[pred_idx_start]
    # For the last sentence, end_index is the length of the original text
    else:
        end_index = len(text)

    return end_index - start_index

