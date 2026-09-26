"""Makes the updatable Core ML head for MLUpdateTask.

MLUpdateTask only works on the old NeuralNetwork format, so the head (Linear 68 -> 5) is
built directly as a NeuralNetwork with the trained weights, softmax on top, and only the
inner product marked updatable. Loss = cross entropy, optimizer = plain SGD, no momentum,
same as finetune() in enroll.py. Backbone stays in the ML Program embedding model.

epochs and mini batch size get overridden at update time from swift (full batch, 30 steps).

run: python updatable.py
"""
import argparse
import os

import coremltools as ct
import torch
from coremltools.models import datatypes
from coremltools.models.neural_network import NeuralNetworkBuilder, SgdParams

from model import EMB, N_CLS, N_RR, BeatNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="runs/baseline/model.pt")
    ap.add_argument("--out", default="runs/coreml/ecg_head_updatable.mlmodel")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--steps", type=int, default=30)
    a = ap.parse_args()

    model = BeatNet()
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"))
    W = model.head.fc.weight.detach().numpy().astype("float32")  # (5, 68)
    bias = model.head.fc.bias.detach().numpy().astype("float32")

    d = EMB + N_RR
    b = NeuralNetworkBuilder([("features", datatypes.Array(d))], [("probs", datatypes.Array(N_CLS))])
    b.add_inner_product("fc", W, bias, input_channels=d, output_channels=N_CLS, has_bias=True,
                        input_name="features", output_name="logits")
    b.add_softmax("softmax", "logits", "probs")
    b.make_updatable(["fc"])
    b.set_categorical_cross_entropy_loss(name="loss", input="probs")
    b.set_sgd_optimizer(SgdParams(lr=a.lr, batch=1))
    b.set_epochs(a.steps)
    # full batch = every enrollment beat, count differs per patient, so allow a range not just 1
    mb = b.spec.neuralNetwork.updateParams.optimizer.sgdOptimizer.miniBatchSize
    mb.range.minValue, mb.range.maxValue = 1, 4096

    m = ct.models.MLModel(b.spec)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    m.save(a.out)
    print("training inputs:", [(f.name, f.type.WhichOneof("Type")) for f in b.spec.description.trainingInput])
    print(f"saved {a.out}")


if __name__ == "__main__":
    main()
