import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

logger = logging.getLogger(__name__)
device = torch.device("cuda") if torch.cuda.is_available() else "cpu"


class LitRnn(pl.LightningModule):
    def __init__(self,
                 vocab_size,
                 hidden_size,
                 n_embd,
                 weight_decay=0.1,
                 learning_rate=3e-4,
                 betas=(0.9, 0.95)):
        super().__init__()

        self.save_hyperparameters()
        self.config = self.hparams

        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.rnn = nn.RNN(n_embd, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size, bias=False)

        logger.info("number of parameters: %e", sum(p.numel() for p in self.parameters()))

    def forward(self, idx):
        batch_size = idx.size(0)

        hidden_state = self.init_hidden_state(batch_size)
        token_embeddings = self.tok_emb(idx)

        out, hidden_state = self.rnn(token_embeddings, hidden_state)
        out = self.linear(out)

        return out, hidden_state

    def init_hidden_state(self, batch_size):
        hidden = torch.zeros(1, batch_size, self.config.hidden_size)
        return hidden.to(device)

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        out, hidden = self.forward(idx)

        # FIXME: why do we need all these .view(-1)?
        loss = F.cross_entropy(out.view(-1, out.size(-1)), targets.view(-1))

        result = pl.TrainResult(minimize=loss, checkpoint_on=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result

    def configure_optimizers(self):
        # create the optimizer
        no_decay = ["bias", "LayerNorm.weight"]
        params_decay = [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)]
        params_nodecay = [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)]
        optim_groups = [
            {"params": params_decay, "weight_decay": self.hparams.weight_decay},
            {"params": params_nodecay, "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=self.hparams.learning_rate, betas=self.hparams.betas)
        return optimizer
