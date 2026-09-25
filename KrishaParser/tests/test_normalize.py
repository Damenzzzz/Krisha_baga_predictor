from krisha.parsers import normalize as N


def test_to_int():
    assert N.to_int("350 000 ₸") == 350000
    assert N.to_int("74 м²") == 74
    assert N.to_int(None) is None
    assert N.to_int("нет цифр") is None


def test_parse_rooms():
    assert N.parse_rooms("3-комнатная квартира · 74 м²") == 3
    assert N.parse_rooms("2-комнатная") == 2
    assert N.parse_rooms(None) is None


def test_parse_floor():
    assert N.parse_floor("1 из 5") == (1, 5)
    assert N.parse_floor("3-комнатная · 74 м² · 1/5 этаж") == (1, 5)
    assert N.parse_floor("этаж не указан") == (None, None)


def test_parse_areas():
    a = N.parse_areas("74 м², Площадь кухни — 9 м²")
    assert a["area_total"] == 74.0
    assert a["area_kitchen"] == 9.0
    assert N.parse_areas(None)["area_total"] is None
