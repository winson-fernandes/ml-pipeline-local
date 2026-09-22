import torch

from src.common.model_def import HeartDiseaseNN


def test_forward_pass_returns_two_logits_per_sample():
    model = HeartDiseaseNN(input_size=19)
    model.eval()

    batch = torch.randn(4, 19)
    output = model(batch)

    assert output.shape == (4, 2)


def test_forward_pass_works_with_a_single_sample_in_eval_mode():
    model = HeartDiseaseNN(input_size=19)
    model.eval()

    single_sample = torch.randn(1, 19)
    output = model(single_sample)

    assert output.shape == (1, 2)
