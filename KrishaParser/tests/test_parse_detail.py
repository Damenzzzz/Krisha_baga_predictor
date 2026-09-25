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
    assert f["bathroom"] == "разделен"  # current form is present alongside legacy fields
    assert f["balcony"] == "балкон: нет; лоджия: 1"
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


def test_definition_list_fields_and_empty_duplicates():
    res = parse_detail('''<div class="offer__parameters">
        <dl><dt data-name="house.year">Год постройки</dt><dd>2007</dd></dl>
        <dl><dt data-name="flat.building">Тип дома</dt><dd>монолитный</dd></dl>
        <dl><dt data-name="flat.toilet">Санузел</dt><dd>совмещённый</dd></dl>
        <dl><dt data-name="flat.balcony">Балкон</dt><dd>балкон</dd></dl>
        <dl><dt data-name="flat.balcony">Балкон</dt><dd></dd></dl>
        <dl><dd>Без названия</dd></dl>
    </div>''')
    assert res.fields["year_built"] == 2007
    assert res.fields["building_type"] == "монолитный"
    assert res.fields["bathroom"] == "совмещённый"
    assert res.fields["balcony"] == "балкон"


def test_live_rental_20260923_names():
    res = parse_detail(read_fixture("detail_rental_20260923_params.html"))
    assert res.fields["bathroom"] == "совмещен"  # not the separate `bathroom`=ванна field
    assert res.fields["balcony"] == "балкон: 1"
    assert res.fields["complex_name"] == "Шахристан"
    assert res.fields["floor"] == res.fields["floors_total"] == 16
    assert "year_built" not in res.fields  # not exposed on this live page
    assert "building_type" not in res.fields


def test_balcony_and_loggia_counts_keep_their_meaning():
    res = parse_detail('''<div class="offer__parameters">
        <dl><dt data-name="balcony_count">Балкон</dt><dd>нет</dd></dl>
        <dl><dt data-name="loggia_count">Лоджия</dt><dd>2</dd></dl>
    </div>''')
    assert res.fields["balcony"] == "балкон: нет; лоджия: 2"
