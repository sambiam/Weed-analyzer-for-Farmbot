from farmbot_vision.weed_settings import WeedSettings, WeedSettingsStore


def test_weed_settings_default_off_and_persist_atomically(tmp_path):
    store = WeedSettingsStore(tmp_path / "weed_settings.json")
    assert store.load().enabled is False
    values = WeedSettings(enabled=True, automatic_creation=True, minimum_area_mm2=40)
    store.save(values)
    assert store.load() == values
