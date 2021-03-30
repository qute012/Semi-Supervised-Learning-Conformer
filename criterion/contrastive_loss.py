def buffered_arange(max):
    if not hasattr(buffered_arange, "buf"):
        buffered_arange.buf = torch.LongTensor()
    if max > buffered_arange.buf.numel():
        buffered_arange.buf.resize_(max)
        torch.arange(max, out=buffered_arange.buf)
    return buffered_arange.buf[:max]

def sample_negatives(self, y, num, padding_count=None):
    if self.n_negatives == 0 and self.cross_sample_negatives == 0:
        return y.new(0)

    bsz, tsz, fsz = y.shape
    y = y.view(-1, fsz)  # BTC => (BxT)C

    # FIXME: what happens if padding_count is specified?
    cross_high = tsz * bsz
    high = tsz - (padding_count or 0)
    with torch.no_grad():
        assert high > 1, f"{bsz, tsz, fsz}"

        if self.n_negatives > 0:
            tszs = (
                buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.n_negatives)
                    .flatten()
            )

            neg_idxs = torch.randint(
                low=0, high=high - 1, size=(bsz, self.n_negatives * num)
            )
            neg_idxs[neg_idxs >= tszs] += 1

        if self.cross_sample_negatives > 0:
            tszs = (
                buffered_arange(num)
                    .unsqueeze(-1)
                    .expand(-1, self.cross_sample_negatives)
                    .flatten()
            )

            cross_neg_idxs = torch.randint(
                low=0,
                high=cross_high - 1,
                size=(bsz, self.cross_sample_negatives * num),
            )
            cross_neg_idxs[cross_neg_idxs >= tszs] += 1

    if self.n_negatives > 0:
        for i in range(1, bsz):
            neg_idxs[i] += i * high
    else:
        neg_idxs = cross_neg_idxs

    if self.cross_sample_negatives > 0 and self.n_negatives > 0:
        neg_idxs = torch.cat([neg_idxs, cross_neg_idxs], dim=1)

    negs = y[neg_idxs.view(-1)]
    negs = negs.view(
        bsz, num, self.n_negatives + self.cross_sample_negatives, fsz
    ).permute(
        2, 0, 1, 3
    )  # to NxBxTxC
    return negs, neg_idxs

def compute_preds(self, x, y, negatives):

    neg_is_pos = (y == negatives).all(-1)
    y = y.unsqueeze(0)
    targets = torch.cat([y, negatives], dim=0)

    logits = torch.cosine_similarity(x.float(), targets.float(), dim=-1).type_as(x)

    logits = logits / self.logit_temp

    if is_xla_tensor(logits) or neg_is_pos.any():
        fillval = -float(2 ** 30)
        if not hasattr(self, '_inftensor'):
            self._inftensor = (
                torch.tensor(fillval).to(x.device)
                if is_xla_tensor(logits) else
                float("-inf")
            )
        logits[1:] = index_put(logits[1:], neg_is_pos, self._inftensor)

    return logits

def get_logits(self, net_output):
    logits = net_output["x"]
    logits = logits.transpose(0, 2)
    logits = logits.reshape(-1, logits.size(-1))
    return logits

def get_targets(self, sample, net_output, expand_steps=True):
    x = net_output["x"]
    return x.new_zeros(x.size(1) * x.size(2), dtype=torch.long)


@register_criterion("wav2vec", dataclass=Wav2VecCriterionConfig)
class Wav2vecCriterion(FairseqCriterion):
    def __init__(self, task, infonce=False, loss_weights=None, log_keys=None):
        super().__init__(task)
        self.infonce = infonce
        self.loss_weights = loss_weights
        self.log_keys = [] if log_keys is None else log_keys

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_output = model(**sample["net_input"])
        logits = model.get_logits(net_output).float()
        target = model.get_targets(sample, net_output)
        self.xla = is_xla_tensor(logits)

        # XXX: handle weights on xla.
        weights = None
        if hasattr(model, "get_target_weights") and not self.infonce:
            weights = model.get_target_weights(target, net_output)
            if torch.is_tensor(weights):
                weights = weights.float()

        losses = []

        reduction = "none" if ((not reduce) or self.xla) else "sum"
        if self.infonce:
            loss = F.cross_entropy(logits, target, reduction=reduction)
        else:
            loss = F.binary_cross_entropy_with_logits(
                logits, target.float(), weights, reduction=reduction
            )

        if self.xla:
            # tpu-comment: since dynamic shapes lead to recompilations on xla,
            # we don't shrink tensors using mask_indices.
            # Instead, we use mask indices to adjust loss.
            mi = (
                sample['net_input']['mask_indices']
                .transpose(0, 1)  # logits are transposed in `model.get_logits`
                .reshape(logits.size(0))
            )
            loss = (loss * mi).sum() if reduce else (loss * mi)

        if 'sample_size' in sample and self.infonce:
            sample_size = sample['sample_size']
        elif 'mask_indices' in sample['net_input']:
            sample_size = sample['net_input']['mask_indices'].sum()
        else:
            sample_size = target.numel() if self.infonce else target.long().sum().item()
        losses.append(loss.detach().clone())

        if self.loss_weights is not None:
            assert hasattr(model, "get_extra_losses")
            extra_losses = model.get_extra_losses(net_output)
            if torch.is_tensor(extra_losses):
                extra_losses = [extra_losses]
            if len(self.loss_weights) == 1 and len(extra_losses) != 1:
                self.loss_weights = [self.loss_weights[0]] * len(extra_losses)
            assert len(extra_losses) == len(
                self.loss_weights
            ), f"{len(extra_losses)}, {len(self.loss_weights)}"
            for p, coef in zip(extra_losses, self.loss_weights):
                if coef != 0 and p is not None:
                    p = coef * p.float() * sample_size
                    loss += p
                    losses.append(p)

        logging_output = {
            "loss": loss.item() if (reduce and not self.xla) else loss.detach(),
            "ntokens": sample_size,
            "nsentences": sample["id"].numel(),
            "sample_size": sample_size,
        }

        for lk in self.log_keys:
            # Only store "logits" and "target" for computing MAP and MAUC
            # during validation
            if lk == "logits":
                if not self.training:
                    logging_output["logits"] = logits.cpu().numpy()
            elif lk == "target":
                if not self.training:
                    logging_output["target"] = target.cpu().numpy()
            elif lk in net_output:
                value = net_output[lk]
                if not is_xla_tensor(value):
                    value = float(value)
                logging_output[lk] = value

        if len(losses) > 1:
            for i, l in enumerate(losses):
                logging_output[f"loss_{i}"] = l.item() if not self.xla else l.detach()

        if self.infonce:
            with torch.no_grad():
                if logits.numel() == 0:
                    corr = 0
                    count = 0
                else:
                    assert logits.dim() > 1, logits.shape
                    max = logits.argmax(-1) == 0
                    min = logits.argmin(-1) == 0
                    if is_xla_tensor(logits):
                        max, min = max * mi, min * mi
                        both = max & min
                        corr = max.long().sum() - both.long().sum()
                        count = mi.sum()
                    else:
                        both = max & min
                        corr = max.long().sum().item() - both.long().sum().item()
                        count = float(max.numel())

                logging_output["correct"] = corr
                logging_output["count"] = count

        return loss, sample_size, logging_output

 x = self.compute_preds(x, y, negs)