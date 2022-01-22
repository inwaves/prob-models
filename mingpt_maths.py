import argparse
import time
import logging
import numpy as np
import torch

from torch.utils.data import DataLoader
from pytorch_lightning import Trainer

from mingpt.lr_decay import LearningRateDecayCallback
from addition_dataset import AdditionDataset
from mingpt.utils import set_seed, sample
from mingpt.model import GPT


def give_exam(dataset, batch_size=32, max_batches=-1):
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(model.device)
        d1d2 = x[:, :ndigit * 2]  # A list comprising the first two numbers, `ndigit` digits each.
        d1d2d3 = sample(model, d1d2, ndigit + 1)  # Predict the next number, at most `ndigit + 1` digits.
        d3 = d1d2d3[:, -(ndigit + 1):]  # Extract the last number.

        # A list of factors of 10 in decreasing order, e.g. 10^3, 10^2, 10^1, 10^0.
        factors = torch.tensor([[10 ** i for i in range(ndigit + 1)][::-1]]).to(model.device)

        # Decode the numbers by multiplying digits with factors.
        # The first colon in each list index is so that the nested list format is preserved.
        d1i = (d1d2[:, :ndigit] * factors[:, 1:]).sum(1)
        d2i = (d1d2[:, ndigit:ndigit * 2] * factors[:, 1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)

        # The ground truth is the sum of the first two numbers.
        d3i_gt = d1i + d2i
        correct = (d3i_pred == d3i_gt).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                pass
                # print(f"GPT claims that {d1i[i]:03d} + {d2i[i]:03d} = {d3i_pred[i]:03d} "
                #       f"(gt is {d3i_gt[i]:03d}; {judge})")

        if 0 <= max_batches <= b + 1:
            break

    print(f"final score: {np.sum(results):d}/{len(results):d} = {100 * np.mean(results):.2f}% correct")


def give_exam_generalised(dataset, seqlen=2, batch_size=32, max_batches=-1):
    results = []
    loader = DataLoader(dataset, batch_size=batch_size)
    for b, (x, y) in enumerate(loader):
        x = x.to(model.device)
        input_sequence = x[:, :seqlen * ndigit]  # The first three numbers, the second of which is a decoy.
        output_sequence = sample(model, input_sequence, ndigit + 1)
        d3 = output_sequence[:, -(ndigit + 1):]
        factors = torch.tensor([[10 ** i for i in range(ndigit + 1)][::-1]]).to(model.device)

        # decode the integers from individual digits
        d1i = (input_sequence[:, :ndigit] * factors[:, 1:]).sum(1)
        d2i = (input_sequence[:, (seqlen-1) * ndigit:seqlen * ndigit] * factors[:, 1:]).sum(1)
        d3i_pred = (d3 * factors).sum(1)
        d3i_gt = d1i + d2i
        correct = (d3i_pred == d3i_gt).cpu()
        for i in range(x.size(0)):
            results.append(int(correct[i]))
            judge = 'YEP!!!' if correct[i] else 'NOPE'
            if not correct[i]:
                pass
                # print(f"GPT claims that {d1i[i]:03d} + {d2i[i]:03d} = {d3i_pred[i]:03d} "
                #       f"(gt is {d3i_gt[i]:03d}; {judge})")

        if 0 <= max_batches <= b + 1:
            break

    print(f"Single run: final score: {np.sum(results):d}/{len(results):d} = {100 * np.mean(results):.2f}% correct\n")
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
    parser.add_argument("--epochs", type=int, default=20, help="sequence length")

    args = parser.parse_args()

    # make deterministic
    set_seed(42)

    # create a dataset for e.g. 2-digit addition
    ndigit = 2

    train_dataset = AdditionDataset(ndigit=ndigit, seqlen=args.seqlen, split='train')
    test_dataset = AdditionDataset(ndigit=ndigit, seqlen=args.seqlen, split='test')
    train_dataloader = DataLoader(train_dataset, batch_size=128, num_workers=10)
    val_dataloader = DataLoader(test_dataset, batch_size=128, num_workers=10)

    # initialize a baby GPT model
    model = GPT(vocab_size=train_dataset.vocab_size,
                block_size=train_dataset.block_size,
                n_layer=2,
                n_head=4,
                n_embd=128,
                learning_rate=6e-4)

    lr_decay = LearningRateDecayCallback(learning_rate=6e-4, warmup_tokens=1024,
                                         final_tokens=50 * len(train_dataset) * (ndigit + 1))

    # Train the model.
    tic = time.time()
    trainer = Trainer(gpus=2, distributed_backend="ddp", precision=16, max_epochs=args.epochs, callbacks=[lr_decay])
    trainer.fit(model, train_dataloader, val_dataloader)
    toc = time.time()
    print(f"Training took {toc - tic:.2f} seconds.")

    # training set: how well did we memorize?
    train_results = [give_exam_generalised(train_dataset, seqlen=args.seqlen, batch_size=1024, max_batches=10)
                     for _ in range(3)]

    # test set: how well did we generalize?
    test_results = [give_exam_generalised(test_dataset, seqlen=args.seqlen, batch_size=1024, max_batches=-1)
                    for _ in range(3)]

    with open("./logs/log.txt", "a+") as f:
        f.write(f"GPT {ndigit:d}-digit addition, epochs: {args.epochs} seqlen {args.seqlen}: "
                f"{100*np.mean(train_results):.2f}% train, {100*np.mean(test_results):.2f}% test. "
                f"Time elapsed: {toc - tic}s.\n")