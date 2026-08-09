from flag_gems.runtime.backend.device import DeviceDetector


def test_get_vendor_from_legacy_flaggems_vendor_name(monkeypatch):
    for key in (
        "GEMS_VENDOR",
        "FLAGGEMS_VENDOR",
        "GEMS_BACKEND",
        "FLAGGEMS_BACKEND",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("FLAGGEMS_VENDOR_NAME", "ARM")

    assert DeviceDetector._get_vendor_from_env(None) == "arm"
