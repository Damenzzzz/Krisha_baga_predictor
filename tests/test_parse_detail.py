from conftest import read_fixture

from krisha.parsers import parse_detail


def test_parse_detail_core_fields():
    res = parse_detail(read_fixture("detail_673910188.html"),
                       url="https://krisha.kz/a/show/673910188")
    assert res.parsed is True
    f = res.fields
    assert f["price_kzt"] == 350000
    assert f["rooms"] == 3
    assert f["area_total"] == 74.0
    assert f["floor"] == 1
    assert f["floors_total"] == 5
    assert f["year_built"] == 1975
    assert f["building_type"] == "кирпичный"
    assert f["area_kitchen"] == 9.0
    assert f["rent_period"] == "month"
    assert f["deal_type"] == "rent"
    # geo from window.data.advert.map
    assert 43.0 < f["lat"] < 44.0
    assert 76.0 < f["lon"] < 77.0


def test_parse_detail_photos():
    res = parse_detail(read_fixture("detail_673910188.html"))
    assert res.fields["photos_count"] == len(res.photos) > 0
    assert all(p["url"].startswith("http") for p in res.photos)


def test_parse_detail_never_raises_on_garbage():
    res = parse_detail("<html><body>no window.data here</body></html>")
    assert res.parsed is False
    assert res.fields.get("price_kzt") is None
