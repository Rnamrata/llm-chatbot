import sys
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_whisper(monkeypatch):
    """A stand-in for the whisper module, so no model is downloaded or loaded"""
    module = MagicMock()
    module.load_model.return_value = 'the-model'
    monkeypatch.setitem(sys.modules, 'whisper', module)
    return module


@pytest.fixture
def file_manager(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    # Imported here, not at the top of the file: conftest.py purges the project's modules
    # between tests, so a top-level import could be a different copy than the one patched below.
    from src.modules.file_manager import FileManager
    return FileManager(MagicMock(), MagicMock(), MagicMock())


def test_creating_a_file_manager_does_not_load_the_speech_model(fake_whisper, file_manager):
    fake_whisper.load_model.assert_not_called()


def test_the_speech_model_is_loaded_when_first_needed(fake_whisper, file_manager):
    assert file_manager.whisper_model == 'the-model'

    fake_whisper.load_model.assert_called_once()


def test_the_speech_model_is_loaded_only_once(fake_whisper, file_manager):
    file_manager.whisper_model
    file_manager.whisper_model

    assert fake_whisper.load_model.call_count == 1


def test_the_configured_speech_model_is_the_one_loaded(fake_whisper, file_manager, monkeypatch):
    module = sys.modules[type(file_manager).__module__]
    monkeypatch.setattr(module.config, 'WHISPER_MODEL', 'tiny')

    file_manager.whisper_model

    fake_whisper.load_model.assert_called_once_with('tiny')


def test_the_default_speech_model_is_base(fake_whisper, file_manager):
    file_manager.whisper_model

    fake_whisper.load_model.assert_called_once_with('base')


def test_importing_the_file_manager_does_not_import_whisper(monkeypatch):
    # If the module imported whisper at the top, this poisoned entry would make it fail
    monkeypatch.setitem(sys.modules, 'whisper', None)
    monkeypatch.delitem(sys.modules, 'src.modules.file_manager', raising=False)

    __import__('src.modules.file_manager')
    