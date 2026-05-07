import os
from unittest.mock import MagicMock, patch
import pytest
from bs4 import BeautifulSoup
from document_qa.grobid_processors import (
    GrobidProcessor,
    GrobidServiceError,
    get_xml_nodes_body,
    get_xml_nodes_figures,
    get_xml_nodes_header,
)
from tests.resources import TEST_DATA_PATH


def test_get_xml_nodes_body_paragraphs():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    nodes = get_xml_nodes_body(soup, use_paragraphs=True)

    assert len(nodes) == 70


def test_get_xml_nodes_body_sentences():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.sentences.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_body(soup, use_paragraphs=False)

    assert len(children) == 327


def test_get_xml_nodes_figures():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_figures(soup)

    assert len(children) == 13


def test_get_xml_nodes_header_paragraphs():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.paragraphs.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup)

    assert sum([len(child) for k, child in children.items()]) == 8


def test_get_xml_nodes_header_sentences():
    with open(os.path.join(TEST_DATA_PATH, "2312.07559.sentences.tei.xml"), 'r') as fo:
        soup = BeautifulSoup(fo, 'xml')

    children = get_xml_nodes_header(soup, use_paragraphs=False)

    assert sum([len(child) for k, child in children.items()]) == 15

def test_grobid_service_error_default_status_code():
    error = GrobidServiceError("Something went wrong")
    assert error.status_code is None
    assert str(error) == "Something went wrong"


def test_grobid_service_error_stores_status_code():
    error = GrobidServiceError("Bad gateway", status_code=502)
    assert error.status_code == 502
    assert "Bad gateway" in str(error)

@pytest.fixture
def grobid_processor():
    with patch("document_qa.grobid_processors.GrobidClient") as mock_client_class:
        mock_client = MagicMock()
        mock_client_class.return_value = mock_client
        processor = GrobidProcessor("http://fake-url", ping_server=False)
        yield processor


# Connection/timeout failures
def test_process_structure_raises_on_connection_error(grobid_processor):
    grobid_processor.grobid_client.process_pdf.side_effect = ConnectionError(
        "Connection refused"
    )
    with pytest.raises(GrobidServiceError) as exc_info:
        grobid_processor.process_structure("fake.pdf")

    assert "did not respond" in str(exc_info.value).lower()
    assert exc_info.value.status_code is None


def test_process_structure_raises_on_timeout(grobid_processor):
    grobid_processor.grobid_client.process_pdf.side_effect = TimeoutError(
        "Request timed out"
    )
    with pytest.raises(GrobidServiceError) as exc_info:
        grobid_processor.process_structure("fake.pdf")

    assert exc_info.value.status_code is None


#  Non-200 HTTP status codes
def test_process_structure_raises_on_503_status(grobid_processor):
    grobid_processor.grobid_client.process_pdf.return_value = ("fake.pdf", 503, None)

    with pytest.raises(GrobidServiceError) as exc_info:
        grobid_processor.process_structure("fake.pdf")

    assert exc_info.value.status_code == 503
    assert "503" in str(exc_info.value)


def test_process_structure_raises_on_500_status(grobid_processor):
    grobid_processor.grobid_client.process_pdf.return_value = ("fake.pdf", 500, None)

    with pytest.raises(GrobidServiceError) as exc_info:
        grobid_processor.process_structure("fake.pdf")

    assert exc_info.value.status_code == 500
    assert "500" in str(exc_info.value)


def test_process_structure_raises_on_404_status(grobid_processor):
    grobid_processor.grobid_client.process_pdf.return_value = ("fake.pdf", 404, None)

    with pytest.raises(GrobidServiceError) as exc_info:
        grobid_processor.process_structure("fake.pdf")

    assert exc_info.value.status_code == 404