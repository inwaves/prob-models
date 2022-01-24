import argparse
import time
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from mingpt.lr_decay import LearningRateDecayCallback
from addition_dataset import AdditionDataset
from mingpt.utils import set_seed
from rnn.utils import sample
from rnn.model import LitRnn


def give_exam_generalised(dataset, seqlen=2, batch_size=32, max_batches=-1):
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(model.device)

        # The first numbers, the all of which are decoys except the first and last.
        input_sequence = x[:, :seqlen * args.ndigits]

        # Predict the next number, at most `args.ndigits + 1` digits.
        output_sequence = sample(model, input_sequence, args.ndigits + 1)

        # Extract the last number.
        d3 = output_sequence[:, -(args.ndigits + 1):]

        # A list of factors of 10 in decreasing order, e.g. 10^3, 10^2, 10^1, 10^0.
        factors = torch.tensor([[10 ** i for i in range(args.ndigits + 1)][::-1]]).to(model.device)

        # Decode the numbers by multiplying digits with factors.
        # The first colon in each list index is so that the nested list format is preserved.
        d1i = (input_sequence[:, :args.ndigits] * factors[:, 1:]).sum(1)
        d2i = (input_sequence[:, (seqlen-1) * args.ndigits:seqlen * args.ndigits] * factors[:, 1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i

        # The ground truth is the sum of the first two numbers.
        correct = (d3i_pred == d3i_gt).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                pass
                # print(f"RNN claims that {d1i[i]:03d} + {d2i[i]:03d} = {d3i_pred[i]:03d} "
                #       f"(gt is {d3i_gt[i]:03d}; {judge})")

        if 0 <= max_batches <= b + 1:
            break

    print(f"final score: {np.sum(results):d}/{len(results):d} = {100 * np.mean(results):.2f}% correct")
    return np.mean(results)


if __name__ == '__main__':
    # set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    parser = argparse.ArgumentParser()

    # The length of the input sequence, where seqlen - 2
    # of the numbers are decoys (i.e. not used in the addition).
    parser.add_argument("--seqlen", type=int, default=2, help="sequence length")
    parser.add_argument("--epochs", type=int, default=20, help="number of epochs")
    parser.add_argument("--ndigits", type=int, default=2, help="number of digits")
    parser.add_argument("--hidden_size", type=int, default=305, help="number of digits")
    args = parser.parse_args()

    # Make deterministic.
    set_seed(42)

    # Create a dataset for e.g. 2-digit addition.
    args.ndigits = 2

    train_dataset = AdditionDataset(ndigit=args.ndigits, seqlen=args.seqlen, split='train')
    test_dataset = AdditionDataset(ndigit=args.ndigits, seqlen=args.seqlen, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=0)
    val_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=0)

    # Initialise an RNN.
    model = LitRnn(vocab_size=train_dataset.vocab_size, hidden_size=args.hidden_size, n_embd=128, learning_rate=6e-4)

    lr_decay = LearningRateDecayCallback(learning_rate=6e-4, warmup_tokens=1024,
                                         final_tokens=50 * len(train_dataset) * (args.ndigits + 1))

    # Train the RNN.
    tic = time.time()
    trainer = Trainer(max_epochs=args.epochs, callbacks=[lr_decay])
    trainer.fit(model, train_dataloader, val_dataloader)
    toc = time.time()

    # training set: how well did we memorize?
    train_results = [give_exam_generalised(train_dataset, seqlen=args.seqlen, batch_size=1024, max_batches=10)
                     for _ in range(3)]

    # test set: how well did we generalize?
    test_results = [give_exam_generalised(test_dataset, seqlen=args.seqlen, batch_size=1024, max_batches=-1)
                    for _ in range(3)]

    with open("./logs/log.txt", "a+") as f:
        f.write(f"GRU {args.ndigits:d}-digit addition, epochs: {args.epochs} seqlen {args.seqlen}: "
                f"{100*np.mean(train_results):.2f}\t % train, {100*np.mean(test_results):.2f}\t% test. "
                f"Time elapsed: {toc-tic}s.\n")
