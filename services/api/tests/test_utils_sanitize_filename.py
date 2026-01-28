from services.api.app.utils import sanitize_filename


def test_sanitize_strips_traversal():
    assert sanitize_filename("../../secret.wav") == "secret.wav"
    assert sanitize_filename("..\\..\\secret.wav") == "secret.wav"


def test_sanitize_replaces_unsafe_chars():
    assert sanitize_filename("my file (final).wav") == "my_file_final.wav"
    assert sanitize_filename("a<>b?.mp3") == "a_b_.mp3" or sanitize_filename("a<>b?.mp3") == "a_b.mp3"


def test_sanitize_never_empty():
    assert sanitize_filename("") == "file.bin"
    assert sanitize_filename(None) == "file.bin"
    assert sanitize_filename("   ") == "file.bin"
    assert sanitize_filename("...") == "file.bin"
    assert sanitize_filename("__") == "file.bin"
