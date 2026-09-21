from conftest import read_fixture

from krisha.parsers import parse_list


def test_parse_list_ids_and_pagination():
    page = parse_list(read_fixture("list_kaskelen.html"))
    # window.data.search.ids -> 20 canonical results
    assert len(page.ids) >= 1
    assert all(isinstance(i, int) for i in page.ids)
    assert page.current_page == 1
    assert page.nb_total > 0
    assert page.page_count >= 1


def test_parse_list_cards_have_urls():
    page = parse_list(read_fixture("list_kaskelen.html"))
    assert page.cards, "expected some parsed cards"
    sample = next(iter(page.cards.values()))
    assert sample.url.startswith("https://krisha.kz/a/show/")
    assert sample.id > 0
