import rewet.Input.Settings


def get_rewet_hydraulic_basic_setting():
    settings = rewet.Input.Settings.Settings()
    settings_dict = settings.process.settings
    # settings_dict.update(settings.scenario.settings)

    return settings_dict
