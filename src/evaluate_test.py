from evaluate import compute_confidence


def test_compute_confidence() -> None:
    accuracies = [0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8, 0.8]
    conf = compute_confidence(accuracies)
    assert conf == 0.0
