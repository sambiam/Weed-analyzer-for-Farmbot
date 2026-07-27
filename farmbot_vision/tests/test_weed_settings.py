from farmbot_vision.weed_settings import WeedSettings, WeedSettingsStore


def test_weed_settings_default_off_and_persist_atomically(tmp_path):
    store = WeedSettingsStore(tmp_path / "weed_settings.json")
    assert store.load().enabled is False
    values = WeedSettings(enabled=True, automatic_creation=True, minimum_area_mm2=40)
    store.save(values)
    assert store.load() == values


def test_weed_settings_expose_independent_detection_training_and_automation_guards():
    values = WeedSettings()
    assert values.shape_filter_enabled is True
    assert values.crop_protection_enabled is True
    assert values.temporal_confirmation_enabled is True
    assert values.recommendation_min_observations < values.automatic_min_observations
    assert values.visual_verifier_enabled is False
    assert values.visual_verifier_shadow_mode is True
    assert values.visual_verifier_required_for_automatic is True
    assert values.minimum_confidence < values.automatic_creation_confidence
