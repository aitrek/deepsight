"""Loss function for CRNN"""
import torch
import torch.nn as nn
from .lexicon import Lexicon


class CRNNLoss(nn.Module):

    def __init__(self, lexicon: Lexicon):
        super().__init__()
        self._lexicon = lexicon
        self._C = len(lexicon)      # Number of classes
        self._ctc_loss = nn.CTCLoss(zero_infinity=False, reduction="sum")
        self._use_cuda = False

    def cuda(self, device=None):
        self._use_cuda = True
        return super().cuda(device)

    def forward(self, inputs: torch.Tensor, labels):
        inputs = inputs.transpose(0, 1)
        T, B, C = inputs.shape
        if self._use_cuda:
            log_probs = inputs.log_softmax(2).requires_grad_().cuda()
            targets = torch.tensor(
                [[self._lexicon.index(c) for c in label] for label in labels],
                dtype=torch.long).cuda()
            input_lengths = torch.full((B,), T, dtype=torch.long).cuda()
            target_lengths = torch.tensor([len(label) for label in labels],
                                          dtype=torch.long).cuda()
        else:
            log_probs = inputs.log_softmax(2).requires_grad_()
            targets = torch.tensor(
                [[self._lexicon.index(c) for c in label] for label in labels],
                dtype=torch.long)
            input_lengths = torch.full((B,), T, dtype=torch.long)
            target_lengths = torch.tensor([len(label) for label in labels],
                                          dtype=torch.long)

        loss = self._ctc_loss(log_probs, targets, input_lengths, target_lengths)

        return loss




