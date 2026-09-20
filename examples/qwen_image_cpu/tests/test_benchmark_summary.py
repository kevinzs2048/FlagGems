import pytest
from qwen_image_cpu.benchmark_summary import summarize_requests


def row(seconds, pixels="same"):
    return {
        "timings": {"resident request": seconds},
        "pixel_sha256": pixels,
        "native_golden_pixel_equal": True,
    }


def test_middle_image_difference_is_not_hidden_by_equal_endpoints():
    result = summarize_requests([row(1), row(2, "different"), row(3)])
    assert result["repeatability_checked"]
    assert not result["all_pixels_identical"]
    assert result["timings"]["resident request"] == {
        "samples": 3,
        "mean_seconds": 2,
        "median_seconds": 2,
        "stdev_seconds": 1,
        "min_seconds": 1,
        "max_seconds": 3,
    }


def test_single_request_does_not_claim_repeatability_or_variance():
    result = summarize_requests([row(1)])
    assert not result["repeatability_checked"]
    assert result["all_pixels_identical"] is None
    assert result["timings"]["resident request"]["stdev_seconds"] is None


def test_all_samples_and_golden_status_are_checked():
    samples = [row(1), row(2), row(3)]
    samples[1]["native_golden_pixel_equal"] = False
    samples[0]["timings"]["only_first"] = 5
    result = summarize_requests(samples)
    assert result["all_pixels_identical"]
    assert not result["all_native_golden_pixel_equal"]
    assert "only_first" not in result["timings"]


def test_empty_samples_rejected():
    with pytest.raises(ValueError):
        summarize_requests([])
