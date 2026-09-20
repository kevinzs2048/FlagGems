"""Summarize explicitly counted CPU requests, without changing thread settings."""

import statistics


def summarize_requests(rows):
    if not rows:
        raise ValueError("At least one measured request is required")
    timings = {}
    # Only summarize phases that actually ran in every request.
    for key in sorted(set.intersection(*(set(row["timings"]) for row in rows))):
        values = [row["timings"][key] for row in rows]
        timings[key] = {
            "samples": len(values),
            "mean_seconds": statistics.mean(values),
            "median_seconds": statistics.median(values),
            "stdev_seconds": statistics.stdev(values) if len(values) > 1 else None,
            "min_seconds": min(values),
            "max_seconds": max(values),
        }
    hashes = [row["pixel_sha256"] for row in rows]
    return {
        "measured_requests": len(rows),
        "timings": timings,
        "repeatability_checked": len(rows) > 1,
        "all_pixels_identical": (
            all(value == hashes[0] for value in hashes) if len(rows) > 1 else None
        ),
        "all_native_golden_pixel_equal": all(
            row["native_golden_pixel_equal"] for row in rows
        ),
    }
