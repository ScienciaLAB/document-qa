"""GROBID-based processors for scientific text extraction.

This module provides processors that interact with GROBID services to:

- **Extract structured text** from scientific PDFs (:class:`GrobidProcessor`)
  — parses TEI-XML into passages with section labels and PDF coordinates.
- **Annotate physical quantities** (:class:`GrobidQuantitiesProcessor`)
  — identifies measurements via the grobid-quantities service.
- **Annotate materials** (:class:`GrobidMaterialsProcessor`)
  — identifies material mentions via grobid-superconductors.
- **Aggregate NER results** (:class:`GrobidAggregationProcessor`)
  — combines quantity and material annotations with overlap pruning.

"""

import re
from collections import OrderedDict
from html import escape
from pathlib import Path

import dateparser
import grobid_tei_xml
from bs4 import BeautifulSoup
from grobid_client.grobid_client import GrobidClient


def get_span_start(type, title=None):
    """Return an opening ``<span>`` tag for an annotation of the given *type*."""
    title_ = ' title="' + title + '"' if title is not None else ""
    return '<span class="label ' + type + '"' + title_ + '>'


def get_span_end():
    return '</span>'


def get_rs_start(type):
    return '<rs type="' + type + '">'


def get_rs_end():
    return '</rs>'


def has_space_between_value_and_unit(quantity):
    return quantity['offsetEnd'] < quantity['rawUnit']['offsetStart']


def decorate_text_with_annotations(text, spans, tag="span"):
    """Wrap recognised entity spans in markup tags.

    Produces either HTML (``<span class="label …">``) or TEI-XML
    (``<rs type="…">``) depending on *tag*.

    Args:
        text: The original plain-text string.
        spans: List of span dicts with at least ``offset_start``,
            ``offset_end``, and ``type`` keys.
        tag: ``"span"`` (default) for HTML output, ``"rs"`` for XML.

    Returns:
        str: The text with inline annotation markup.
    """
    sorted_spans = list(sorted(spans, key=lambda item: item['offset_start']))
    annotated_text = ""
    start = 0
    for span in sorted_spans:
        type = span['type'].replace("<", "").replace(">", "")
        if 'unit_type' in span and span['unit_type'] is not None:
            type = span['unit_type'].replace(" ", "_")
        annotated_text += escape(text[start: span['offset_start']])
        title = span['quantified'] if 'quantified' in span else None
        annotated_text += get_span_start(type, title) if tag == "span" else get_rs_start(type)
        annotated_text += escape(text[span['offset_start']: span['offset_end']])
        annotated_text += get_span_end() if tag == "span" else get_rs_end()

        start = span['offset_end']
    annotated_text += escape(text[start: len(text)])
    return annotated_text


def get_parsed_value_type(quantity):
    if 'parsedValue' in quantity and 'structure' in quantity['parsedValue']:
        return quantity['parsedValue']['structure']['type']


class BaseProcessor(object):
    """Shared post-processing logic for all GROBID-derived processors.

    Fixes common character-encoding artefacts produced by PDF extraction
    (e.g. ``À`` → ``-``, ``¼`` → ``=``).  All processor subclasses
    inherit :meth:`post_process` from here.
    """

    patterns = [
        r'\d+e\d+'
    ]

    def post_process(self, text):
        """Clean encoding artefacts and normalise special characters.

        Args:
            text: Raw extracted text from GROBID.

        Returns:
            str: Cleaned text.
        """
        output = text.replace('À', '-')
        output = output.replace('¼', '=')
        output = output.replace('þ', '+')
        output = output.replace('Â', 'x')
        output = output.replace('$', '~')
        output = output.replace('−', '-')
        output = output.replace('–', '-')

        for pattern in self.patterns:
            output = re.sub(pattern, lambda match: match.group().replace('e', '-'), output)

        return output


class GrobidProcessor(BaseProcessor):
    """Extract structured text and coordinates from PDFs via GROBID.

    Sends a PDF to a running GROBID server, parses the returned TEI-XML,
    and produces a list of passage dicts with text content, section labels,
    and bounding-box coordinates for each paragraph.

    Args:
        grobid_url: Full URL of the GROBID server
            (e.g. ``"https://grobid.example.com"``).
        ping_server: If ``True`` (default), verify the server is alive
            on init.

    Raises:
        ServerUnavailableException: If *ping_server* is ``True`` and the
            GROBID server does not respond.

    """

    def __init__(self, grobid_url, ping_server=True):
        grobid_client = GrobidClient(
            grobid_server=grobid_url,
            batch_size=5,
            coordinates=["p", "title", "persName"],
            sleep_time=5,
            timeout=60,
            check_server=ping_server
        )
        self.grobid_client = grobid_client

    def process_structure(self, input_path, coordinates=False):
        """Send a PDF to GROBID and return structured content.

        Args:
            input_path: Path to the PDF file.
            coordinates: If ``True``, include bounding-box coordinate
                strings in each passage (needed for PDF highlighting).

        Returns:
            dict or None: A dict with keys:

            - ``"biblio"`` — bibliographic metadata (title, authors, DOI, …).
            - ``"passages"`` — list of passage dicts, each containing
              ``text``, ``type``, ``section``, ``subSection``,
              ``passage_id``, and ``coordinates``.
            - ``"filename"`` — stem of the PDF filename.

            Returns ``None`` if GROBID returns a non-200 status.
        """
        pdf_file, status, text = self.grobid_client.process_pdf("processFulltextDocument",
                                                                input_path,
                                                                consolidate_header=True,
                                                                consolidate_citations=False,
                                                                segment_sentences=False,
                                                                tei_coordinates=coordinates,
                                                                include_raw_citations=False,
                                                                include_raw_affiliations=False,
                                                                generateIDs=True)

        if status != 200:
            return

        document_object = self.parse_grobid_xml(text, coordinates=coordinates)
        document_object['filename'] = Path(pdf_file).stem.replace(".tei", "")

        return document_object

    def process_single(self, input_file):
        doc = self.process_structure(input_file)

        for paragraph in doc['passages']:
            entities = self.process_single_text(paragraph['text'])
            paragraph['spans'] = entities

        return doc

    def parse_grobid_xml(self, text, coordinates=False):
        """Parse GROBID TEI-XML into a structured passage dict.

        Extracts title, abstract, body paragraphs, back-matter, and
        figure descriptions from the XML, post-processes encoding
        artefacts, and attaches coordinate metadata.

        Args:
            text: Raw TEI-XML string returned by GROBID.
            coordinates: Whether to extract ``coords`` attributes.

        Returns:
            dict: ``{"biblio": {…}, "passages": […]}``
        """
        output_data = OrderedDict()

        doc_biblio = grobid_tei_xml.parse_document_xml(text)
        biblio = {
            "doi": doc_biblio.header.doi if doc_biblio.header.doi is not None else "",
            "authors": ", ".join([author.full_name for author in doc_biblio.header.authors]),
            "title": doc_biblio.header.title,
            "hash": doc_biblio.pdf_md5
        }
        try:
            year = dateparser.parse(doc_biblio.header.date).year
            biblio["publication_year"] = year
        except:
            pass

        output_data['biblio'] = biblio
        passages = []
        output_data['passages'] = passages
        passage_type = "paragraph"

        soup = BeautifulSoup(text, 'xml')
        blocks_header = get_xml_nodes_header(soup, use_paragraphs=True)

        # passages.append({
        #     "text": f"authors: {biblio['authors']}",
        #     "type": passage_type,
        #     "section": "<header>",
        #     "subSection": "<authors>",
        #     "passage_id": "hauthors",
        #     "coordinates": ";".join([node['coords'] if coordinates and node.has_attr('coords') else "" for node in
        #                              blocks_header['authors']])
        # })

        passages.append({
            "text": self.post_process(" ".join([node.text for node in blocks_header['title']])),
            "type": passage_type,
            "section": "<header>",
            "subSection": "<title>",
            "passage_id": "htitle",
            "coordinates": ";".join([node['coords'] if coordinates and node.has_attr('coords') else "" for node in
                                     blocks_header['title']])
        })

        passages.append({
            "text": self.post_process(
                ''.join(node.text for node in blocks_header['abstract'] for text in node.find_all(text=True) if
                        text.parent.name != "ref" or (
                                text.parent.name == "ref" and text.parent.attrs[
                            'type'] != 'bibr'))),
            "type": passage_type,
            "section": "<header>",
            "subSection": "<abstract>",
            "passage_id": "habstract",
            "coordinates": ";".join([node['coords'] if coordinates and node.has_attr('coords') else "" for node in
                                     blocks_header['abstract']])
        })

        text_blocks_body = get_xml_nodes_body(soup, verbose=False, use_paragraphs=True)
        text_blocks_body.extend(get_xml_nodes_back(soup, verbose=False, use_paragraphs=True))

        use_paragraphs = True
        if not use_paragraphs:
            passages.extend([
                {
                    "text": self.post_process(''.join(text for text in sentence.find_all(text=True) if
                                                      text.parent.name != "ref" or (
                                                              text.parent.name == "ref" and text.parent.attrs[
                                                          'type'] != 'bibr'))),
                    "type": passage_type,
                    "section": "<body>",
                    "subSection": "<paragraph>",
                    "passage_id": str(paragraph_id),
                    "coordinates": paragraph['coords'] if coordinates and sentence.has_attr('coords') else ""
                }
                for paragraph_id, paragraph in enumerate(text_blocks_body) for
                sentence_id, sentence in enumerate(paragraph)
            ])
        else:
            passages.extend([
                {
                    "text": self.post_process(''.join(text for text in paragraph.find_all(text=True) if
                                                      text.parent.name != "ref" or (
                                                              text.parent.name == "ref" and text.parent.attrs[
                                                          'type'] != 'bibr'))),
                    "type": passage_type,
                    "section": "<body>",
                    "subSection": "<paragraph>",
                    "passage_id": str(paragraph_id),
                    "coordinates": paragraph['coords'] if coordinates and paragraph.has_attr('coords') else ""
                }
                for paragraph_id, paragraph in enumerate(text_blocks_body)
            ])

        text_blocks_figures = get_xml_nodes_figures(soup, verbose=False)

        if not use_paragraphs:
            passages.extend([
                {
                    "text": self.post_process(''.join(text for text in sentence.find_all(text=True) if
                                                      text.parent.name != "ref" or (
                                                              text.parent.name == "ref" and text.parent.attrs[
                                                          'type'] != 'bibr'))),
                    "type": passage_type,
                    "section": "<body>",
                    "subSection": "<figure>",
                    "passage_id": str(paragraph_id) + str(sentence_id),
                    "coordinates": sentence['coords'] if coordinates and 'coords' in sentence else ""
                }
                for paragraph_id, paragraph in enumerate(text_blocks_figures) for
                sentence_id, sentence in enumerate(paragraph)
            ])
        else:
            passages.extend([
                {
                    "text": self.post_process(''.join(text for text in paragraph.find_all(text=True) if
                                                      text.parent.name != "ref" or (
                                                              text.parent.name == "ref" and text.parent.attrs[
                                                          'type'] != 'bibr'))),
                    "type": passage_type,
                    "section": "<body>",
                    "subSection": "<figure>",
                    "passage_id": str(paragraph_id),
                    "coordinates": paragraph['coords'] if coordinates and paragraph.has_attr('coords') else ""
                }
                for paragraph_id, paragraph in enumerate(text_blocks_figures)
            ])

        return output_data


class GrobidQuantitiesProcessor(BaseProcessor):
    """NER processor for physical quantities (measurements, units).

    Wraps the `grobid-quantities <https://github.com/kermitt2/grobid-quantities>`_
    service to identify and normalise measurements in text.

    Args:
        grobid_quantities_client: A configured quantities API client
    """

    def __init__(self, grobid_quantities_client):
        self.grobid_quantities_client = grobid_quantities_client

    def process(self, text) -> list:
        """Extract quantity spans from *text*.

        Args:
            text: Plain text to analyse.

        Returns:
            list[dict]: Span dicts with ``offset_start``, ``offset_end``,
            ``type`` (``"property"``), and optional ``unit_type`` /
            ``quantified`` keys.
        """
        status, result = self.grobid_quantities_client.process_text(text.strip())

        if status != 200:
            result = {}

        spans = []

        if 'measurements' in result:
            found_measurements = self.parse_measurements_output(result)

            for m in found_measurements:
                item = {
                    "text": text[m['offset_start']:m['offset_end']],
                    'offset_start': m['offset_start'],
                    'offset_end': m['offset_end']
                }

                if 'raw' in m and m['raw'] != item['text']:
                    item['text'] = m['raw']

                if 'quantified_substance' in m:
                    item['quantified'] = m['quantified_substance']

                if 'type' in m:
                    item["unit_type"] = m['type']

                item['type'] = 'property'
                # if 'raw_value' in m:
                #     item['raw_value'] = m['raw_value']

                spans.append(item)

        return spans

    @staticmethod
    def parse_measurements_output(result):
        measurements_output = []

        for measurement in result['measurements']:
            type = measurement['type']
            measurement_output_object = {}
            quantity_type = None
            has_unit = False
            parsed_value_type = None

            if 'quantified' in measurement:
                if 'normalizedName' in measurement['quantified']:
                    quantified_substance = measurement['quantified']['normalizedName']
                    measurement_output_object["quantified_substance"] = quantified_substance

            if 'measurementOffsets' in measurement:
                measurement_output_object["offset_start"] = measurement["measurementOffsets"]['start']
                measurement_output_object["offset_end"] = measurement["measurementOffsets"]['end']
            else:
                # If there are no offsets we skip the measurement
                continue

            # if 'measurementRaw' in measurement:
            #     measurement_output_object['raw_value'] = measurement['measurementRaw']

            if type == 'value':
                quantity = measurement['quantity']

                parsed_value = GrobidQuantitiesProcessor.get_parsed(quantity)
                if parsed_value:
                    measurement_output_object['parsed'] = parsed_value

                normalized_value = GrobidQuantitiesProcessor.get_normalized(quantity)
                if normalized_value:
                    measurement_output_object['normalized'] = normalized_value

                raw_value = GrobidQuantitiesProcessor.get_raw(quantity)
                if raw_value:
                    measurement_output_object['raw'] = raw_value

                if 'type' in quantity:
                    quantity_type = quantity['type']

                if 'rawUnit' in quantity:
                    has_unit = True

                parsed_value_type = get_parsed_value_type(quantity)

            elif type == 'interval':
                if 'quantityMost' in measurement:
                    quantityMost = measurement['quantityMost']
                    if 'type' in quantityMost:
                        quantity_type = quantityMost['type']

                    if 'rawUnit' in quantityMost:
                        has_unit = True

                    parsed_value_type = get_parsed_value_type(quantityMost)

                if 'quantityLeast' in measurement:
                    quantityLeast = measurement['quantityLeast']

                    if 'type' in quantityLeast:
                        quantity_type = quantityLeast['type']

                    if 'rawUnit' in quantityLeast:
                        has_unit = True

                    parsed_value_type = get_parsed_value_type(quantityLeast)

            elif type == 'listc':
                quantities = measurement['quantities']

                if 'type' in quantities[0]:
                    quantity_type = quantities[0]['type']

                if 'rawUnit' in quantities[0]:
                    has_unit = True

                parsed_value_type = get_parsed_value_type(quantities[0])

            if quantity_type is not None or has_unit:
                measurement_output_object['type'] = quantity_type

            if parsed_value_type is None or parsed_value_type not in ['ALPHABETIC', 'TIME']:
                measurements_output.append(measurement_output_object)

        return measurements_output

    @staticmethod
    def get_parsed(quantity):
        parsed_value = parsed_unit = None
        if 'parsedValue' in quantity and 'parsed' in quantity['parsedValue']:
            parsed_value = quantity['parsedValue']['parsed']
        if 'parsedUnit' in quantity and 'name' in quantity['parsedUnit']:
            parsed_unit = quantity['parsedUnit']['name']

        if parsed_value and parsed_unit:
            if has_space_between_value_and_unit(quantity):
                return str(parsed_value) + str(parsed_unit)
            else:
                return str(parsed_value) + " " + str(parsed_unit)

    @staticmethod
    def get_normalized(quantity):
        normalized_value = normalized_unit = None
        if 'normalizedQuantity' in quantity:
            normalized_value = quantity['normalizedQuantity']
        if 'normalizedUnit' in quantity and 'name' in quantity['normalizedUnit']:
            normalized_unit = quantity['normalizedUnit']['name']

        if normalized_value and normalized_unit:
            if has_space_between_value_and_unit(quantity):
                return str(normalized_value) + " " + str(normalized_unit)
            else:
                return str(normalized_value) + str(normalized_unit)

    @staticmethod
    def get_raw(quantity):
        raw_value = raw_unit = None
        if 'rawValue' in quantity:
            raw_value = quantity['rawValue']
        if 'rawUnit' in quantity and 'name' in quantity['rawUnit']:
            raw_unit = quantity['rawUnit']['name']

        if raw_value and raw_unit:
            if has_space_between_value_and_unit(quantity):
                return str(raw_value) + " " + str(raw_unit)
            else:
                return str(raw_value) + str(raw_unit)


class GrobidMaterialsProcessor(BaseProcessor):
    """NER processor for material mentions (chemical compounds, etc.).

    Wraps the `grobid-superconductors <https://github.com/lfoppiano/grobid-superconductors>`_
    service.

    Args:
        grobid_superconductors_client: A configured
            :class:`~document_qa.ner_client_generic.NERClientGeneric` instance.
    """

    def __init__(self, grobid_superconductors_client):
        self.grobid_superconductors_client = grobid_superconductors_client

    def process(self, text):
        """Extract material-mention spans from *text*.

        Args:
            text: Plain text to analyse.

        Returns:
            list[dict]: Span dicts with ``offset_start``, ``offset_end``,
            ``type`` (``"material"``), and optional ``formula`` keys.
        """
        preprocessed_text = text.strip()
        status, result = self.grobid_superconductors_client.process_text(preprocessed_text,
                                                                         "processText_disable_linking")

        if status != 200:
            result = {}

        spans = []

        if 'passages' in result:
            materials = self.parse_superconductors_output(result, preprocessed_text)

            for m in materials:
                item = {"text": preprocessed_text[m['offset_start']:m['offset_end']]}

                item['offset_start'] = m['offset_start']
                item['offset_end'] = m['offset_end']

                if 'formula' in m:
                    item["formula"] = m['formula']

                item['type'] = 'material'
                item['raw_value'] = m['text']

                spans.append(item)

        return spans

    def parse_materials(self, text):
        status, result = self.grobid_superconductors_client.process_texts(text.strip(), "parseMaterials")

        if status != 200:
            result = []

        results = []
        for position_material in result:
            compositions = []
            for material in position_material:
                if 'resolvedFormulas' in material:
                    for resolved_formula in material['resolvedFormulas']:
                        if 'formulaComposition' in resolved_formula:
                            compositions.append(resolved_formula['formulaComposition'])
                elif 'formula' in material:
                    if 'formulaComposition' in material['formula']:
                        compositions.append(material['formula']['formulaComposition'])
            results.append(compositions)

        return results

    def parse_material(self, text):
        status, result = self.grobid_superconductors_client.process_text(text.strip(), "parseMaterial")

        if status != 200:
            result = []

        compositions = self.output_info(result)

        return compositions

    def output_info(self, result):
        compositions = []
        for material in result:
            if 'resolvedFormulas' in material:
                for resolved_formula in material['resolvedFormulas']:
                    if 'formulaComposition' in resolved_formula:
                        compositions.append(resolved_formula['formulaComposition'])
            elif 'formula' in material:
                if 'formulaComposition' in material['formula']:
                    compositions.append(material['formula']['formulaComposition'])
            if 'name' in material:
                compositions.append(material['name'])
        return compositions

    @staticmethod
    def parse_superconductors_output(result, original_text):
        materials = []

        for passage in result['passages']:
            sentence_offset = original_text.index(passage['text'])
            if 'spans' in passage:
                spans = passage['spans']
                for material_span in filter(lambda s: s['type'] == '<material>', spans):
                    text_ = material_span['text']

                    base_material_information = {
                        "text": text_,
                        "offset_start": sentence_offset + material_span['offset_start'],
                        'offset_end': sentence_offset + material_span['offset_end']
                    }

                    materials.append(base_material_information)

        return materials


class GrobidAggregationProcessor(GrobidQuantitiesProcessor, GrobidMaterialsProcessor):
    """Combined NER processor that merges quantity and material annotations.

    Runs both :class:`GrobidQuantitiesProcessor` and
    :class:`GrobidMaterialsProcessor`, then prunes overlapping spans so
    that the output is clean and non-overlapping.

    Args:
        grobid_quantities_client: Optional quantities API client.
        grobid_superconductors_client: Optional materials NER client.

    Either or both clients may be ``None``; only the provided services
    will be called.
    """

    def __init__(self, grobid_quantities_client=None, grobid_superconductors_client=None):
        if grobid_quantities_client:
            self.gqp = GrobidQuantitiesProcessor(grobid_quantities_client)
        if grobid_superconductors_client:
            self.gmp = GrobidMaterialsProcessor(grobid_superconductors_client)

    def process_single_text(self, text):
        """Run both NER services on *text* and return merged, deduplicated spans.

        Args:
            text: Plain text to process.

        Returns:
            list[dict]: Non-overlapping span dicts sorted by offset.
        """
        extracted_quantities_spans = self.process_properties(text)
        extracted_materials_spans = self.process_materials(text)
        all_entities = extracted_quantities_spans + extracted_materials_spans
        entities = self.prune_overlapping_annotations(all_entities)
        return entities

    def process_properties(self, text):
        if self.gqp:
            return self.gqp.process(text)
        else:
            return []

    def process_materials(self, text):
        if self.gmp:
            return self.gmp.process(text)
        else:
            return []

    @staticmethod
    def box_to_dict(box, color=None, type=None, border=None):
        """Convert a GROBID coordinate list into an annotation dict.

        Args:
            box: List or tuple of ``[page, x, y, width, height]``.
            color: Optional hex colour string for the annotation.
            type: Optional annotation type label.
            border: Optional border style (e.g. ``"dotted"``).

        Returns:
            dict: Annotation dict suitable for ``streamlit-pdf-viewer``,
            or empty dict if *box* is invalid.
        """

        if box is None or box == "" or len(box) < 5:
            return {}

        item = {"page": box[0], "x": box[1], "y": box[2], "width": box[3], "height": box[4]}
        if color:
            item['color'] = color

        if type:
            item['type'] = type

        if border:
            item['border'] = border

        return item

    @staticmethod
    def prune_overlapping_annotations(entities: list) -> list:
        """Remove overlapping spans, keeping the most informative one.

        When two spans overlap, the longer span is preferred.  Adjacent
        spans of the same type may be merged (e.g. a split decimal number).

        Args:
            entities: List of span dicts with ``offset_start``,
                ``offset_end``, ``type``, and ``text`` keys.

        Returns:
            list[dict]: Pruned, non-overlapping spans sorted by offset.
        """
        # Sorting by offsets
        sorted_entities = sorted(entities, key=lambda d: d['offset_start'])

        if len(entities) <= 1:
            return sorted_entities

        to_be_removed = []

        previous = None
        first = True

        for current in sorted_entities:
            if first:
                first = False
                previous = current
                continue

            if previous['offset_start'] < current['offset_start'] \
                    and previous['offset_end'] < current['offset_end'] \
                    and (previous['offset_end'] < current['offset_start'] \
                         and not (previous['text'] == "-" and current['text'][0].isdigit())):
                previous = current
                continue

            if previous['offset_end'] < current['offset_end']:
                if current['type'] == previous['type']:
                    # Type is the same
                    if current['offset_start'] == previous['offset_end']:
                        if current['type'] == 'property':
                            if current['text'].startswith("."):
                                print(
                                    f"Merging. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")
                                # current entity starts with a ".", suspiciously look like a truncated value
                                to_be_removed.append(previous)
                                current['text'] = previous['text'] + current['text']
                                current['raw_value'] = current['text']
                                current['offset_start'] = previous['offset_start']
                            elif previous['text'].endswith(".") and current['text'][0].isdigit():
                                print(
                                    f"Merging. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")
                                # previous entity ends with ".", current entity starts with a number
                                to_be_removed.append(previous)
                                current['text'] = previous['text'] + current['text']
                                current['raw_value'] = current['text']
                                current['offset_start'] = previous['offset_start']
                            elif previous['text'].startswith("-"):
                                print(
                                    f"Merging. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")
                                # previous starts with a `-`, sherlock this is another truncated value
                                current['text'] = previous['text'] + current['text']
                                current['raw_value'] = current['text']
                                current['offset_start'] = previous['offset_start']
                                to_be_removed.append(previous)
                            else:
                                print("Other cases to be considered: ", previous, current)
                        else:
                            if current['text'].startswith("-"):
                                print(
                                    f"Merging. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")
                                # previous starts with a `-`, sherlock this is another truncated value
                                current['text'] = previous['text'] + current['text']
                                current['raw_value'] = current['text']
                                current['offset_start'] = previous['offset_start']
                                to_be_removed.append(previous)
                            else:
                                print("Other cases to be considered: ", previous, current)

                    elif previous['text'] == "-" and current['text'][0].isdigit():
                        print(
                            f"Merging. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")
                        # previous starts with a `-`, sherlock this is another truncated value
                        current['text'] = previous['text'] + " " * (current['offset_start'] - previous['offset_end']) + \
                                          current['text']
                        current['raw_value'] = current['text']
                        current['offset_start'] = previous['offset_start']
                        to_be_removed.append(previous)
                    else:
                        print(
                            f"Overlapping. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")

                        # take the largest one
                        if len(previous['text']) > len(current['text']):
                            to_be_removed.append(current)
                        elif len(previous['text']) < len(current['text']):
                            to_be_removed.append(previous)
                        else:
                            to_be_removed.append(previous)
                elif current['type'] != previous['type']:
                    print(
                        f"Overlapping. {current['text']} <{current['type']}> with {previous['text']} <{previous['type']}>")

                    if len(previous['text']) > len(current['text']):
                        to_be_removed.append(current)
                    elif len(previous['text']) < len(current['text']):
                        to_be_removed.append(previous)
                    else:
                        if current['type'] == "material":
                            to_be_removed.append(previous)
                        else:
                            to_be_removed.append(current)
                previous = current

            elif previous['offset_end'] > current['offset_end']:
                to_be_removed.append(current)
                # the previous goes after the current, so we keep the previous and we discard the current
            else:
                if current['type'] == "material":
                    to_be_removed.append(previous)
                else:
                    to_be_removed.append(current)
                previous = current

        new_sorted_entities = [e for e in sorted_entities if e not in to_be_removed]

        return new_sorted_entities


class XmlProcessor(BaseProcessor):
    def __init__(self):
        super().__init__()

    def process_structure(self, input_file):
        text = ""
        with open(input_file, encoding='utf-8') as fi:
            text = fi.read()

        output_data = self.parse_xml(text)
        output_data['filename'] = Path(input_file).stem.replace(".tei", "")

        return output_data

    # def process_single(self, input_file):
    #     doc = self.process_structure(input_file)
    #
    #     for paragraph in doc['passages']:
    #         entities = self.process_single_text(paragraph['text'])
    #         paragraph['spans'] = entities
    #
    #     return doc

    def process(self, text):
        output_data = OrderedDict()
        soup = BeautifulSoup(text, 'xml')
        text_blocks_children = get_children_list_supermat(soup, verbose=False)

        passages = []
        output_data['passages'] = passages
        passages.extend([
            {
                "text": self.post_process(''.join(text for text in sentence.find_all(text=True) if
                                                  text.parent.name != "ref" or (
                                                          text.parent.name == "ref" and text.parent.attrs[
                                                      'type'] != 'bibr'))),
                "type": "paragraph",
                "section": "<body>",
                "subSection": "<paragraph>",
                "passage_id": str(paragraph_id) + str(sentence_id)
            }
            for paragraph_id, paragraph in enumerate(text_blocks_children) for
            sentence_id, sentence in enumerate(paragraph)
        ])

        return output_data


def get_children_list_supermat(soup, use_paragraphs=False, verbose=False):
    children = []

    child_name = "p" if use_paragraphs else "s"
    for child in soup.tei.children:
        if child.name == 'teiHeader':
            pass
            children.append(child.find_all("title"))
            children.extend([subchild.find_all(child_name) for subchild in child.find_all("abstract")])
            children.extend([subchild.find_all(child_name) for subchild in child.find_all("ab", {"type": "keywords"})])
        elif child.name == 'text':
            children.extend([subchild.find_all(child_name) for subchild in child.find_all("body")])

    if verbose:
        print(str(children))

    return children


def get_children_list_grobid(soup: object, use_paragraphs: object = True, verbose: object = False) -> object:
    children = []

    child_name = "p" if use_paragraphs else "s"
    for child in soup.TEI.children:
        if child.name == 'teiHeader':
            pass
            # children.extend(child.find_all("title", attrs={"level": "a"}, limit=1))
            # children.extend([subchild.find_all(child_name) for subchild in child.find_all("abstract")])
        elif child.name == 'text':
            children.extend([subchild.find_all(child_name) for subchild in child.find_all("body")])
            children.extend([subchild.find_all("figDesc") for subchild in child.find_all("body")])

    if verbose:
        print(str(children))

    return children


def get_xml_nodes_header(soup: object, use_paragraphs: bool = True) -> list:
    sub_tag = "p" if use_paragraphs else "s"

    header_elements = {
        "authors": [persNameNode for persNameNode in soup.teiHeader.find_all("persName")],
        "abstract": [p_in_abstract for abstractNodes in soup.teiHeader.find_all("abstract") for p_in_abstract in
                     abstractNodes.find_all(sub_tag)],
        "title": [soup.teiHeader.fileDesc.title]
    }

    return header_elements


def get_xml_nodes_body(soup: object, use_paragraphs: bool = True, verbose: bool = False) -> list:
    nodes = []
    tag_name = "p" if use_paragraphs else "s"
    for child in soup.TEI.children:
        if child.name == 'text':
            # nodes.extend([subchild.find_all(tag_name) for subchild in child.find_all("body")])
            nodes.extend(
                [subsubchild for subchild in child.find_all("body") for subsubchild in subchild.find_all(tag_name)])

    if verbose:
        print(str(nodes))

    return nodes


def get_xml_nodes_back(soup: object, use_paragraphs: bool = True, verbose: bool = False) -> list:
    nodes = []
    tag_name = "p" if use_paragraphs else "s"
    for child in soup.TEI.children:
        if child.name == 'text':
            nodes.extend(
                [subsubchild for subchild in child.find_all("back") for subsubchild in subchild.find_all(tag_name)])

    if verbose:
        print(str(nodes))

    return nodes


def get_xml_nodes_figures(soup: object, verbose: bool = False) -> list:
    children = []
    for child in soup.TEI.children:
        if child.name == 'text':
            children.extend(
                [subchild for subchilds in child.find_all("body") for subchild in subchilds.find_all("figDesc")])

    if verbose:
        print(str(children))

    return children
