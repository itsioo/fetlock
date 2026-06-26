import torch

from fetlock.config import AblationSpec, EncoderSpec, HeadSpec, OptimSpec
from fetlock.forge import build_teacher
from fetlock.paddock import synthesize_cohort
from fetlock.paddock.stalls import SupervisedStalls
from fetlock.shoeing import (
    EarlyStopping,
    build_optimizer,
    load_checkpoint,
    save_checkpoint,
    set_seed,
)
from fetlock.soundness import roc_auc
from fetlock.temper import sex_aware_loss


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="max")
    stopper.step(0.80)
    assert not stopper.should_stop
    stopper.step(0.79)
    stopper.step(0.78)
    assert stopper.should_stop


def test_checkpoint_roundtrip_restores_weights(tmp_path):
    net = build_teacher(EncoderSpec(embed_dim=16), HeadSpec(embed_dim=16), AblationSpec())
    optimizer = build_optimizer(net.parameters(), OptimSpec())
    path = tmp_path / "ckpt.pt"
    save_checkpoint(path, net, optimizer, epoch=3, seed=11, extra={"val_auc": 0.9})
    payload = load_checkpoint(path)
    assert payload["epoch"] == 3
    assert payload["seed"] == 11
    assert payload["extra"]["val_auc"] == 0.9
    fresh = build_teacher(EncoderSpec(embed_dim=16), HeadSpec(embed_dim=16), AblationSpec())
    fresh.load_state_dict(payload["model"])
    for left, right in zip(net.state_dict().values(), fresh.state_dict().values()):
        assert torch.equal(left, right)


def test_teacher_overfits_one_batch():
    set_seed(0)
    subjects = synthesize_cohort(24, 100, 2.56, seed=0)
    dataset = SupervisedStalls(subjects, 128, 64, (0, 1, 2, 3, 4, 5))
    count = min(48, len(dataset))
    signals = torch.stack([dataset[i]["signal"] for i in range(count)])
    labels = torch.stack([dataset[i]["label"] for i in range(count)]).float()
    sexes = torch.stack([dataset[i]["sex"] for i in range(count)])
    net = build_teacher(
        EncoderSpec(stages=(8, 16, 16), embed_dim=24), HeadSpec(embed_dim=24), AblationSpec()
    )
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)
    net.train()
    for _ in range(150):
        optimizer.zero_grad()
        out = net(signals, sexes)
        sex_aware_loss(out, labels, HeadSpec())["loss"].backward()
        optimizer.step()
    net.eval()
    with torch.no_grad():
        scores = torch.sigmoid(net(signals, sexes)["logit"]).numpy()
    assert roc_auc(scores, labels.numpy().astype("int64")) > 0.95
