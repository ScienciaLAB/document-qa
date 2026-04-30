import os
import re
from hashlib import blake2b
from tempfile import NamedTemporaryFile

import dotenv
import streamlit as st
from grobid_quantities.quantities import QuantitiesAPI
from langchain.memory import ConversationBufferMemory
from langchain_openai import ChatOpenAI
from streamlit_pdf_viewer import pdf_viewer

from document_qa.custom_embeddings import ModalEmbeddings
from document_qa.document_qa_engine import DocumentQAEngine, DataStorage
from document_qa.grobid_processors import GrobidAggregationProcessor, decorate_text_with_annotations, GrobidServiceError
from document_qa.ner_client_generic import NERClientGeneric

dotenv.load_dotenv(override=True)

API_MODELS = {
    "microsoft/Phi-4-mini-instruct": os.environ["PHI_URL"],
    "Qwen/Qwen3-0.6B": os.environ["QWEN_URL"]
}

API_EMBEDDINGS = {
    'intfloat/multilingual-e5-large-instruct-modal': os.environ['EMBEDS_URL']
}

if 'rqa' not in st.session_state:
    st.session_state['rqa'] = {}

if 'model' not in st.session_state:
    st.session_state['model'] = None

if 'api_keys' not in st.session_state:
    st.session_state['api_keys'] = {}

if 'doc_id' not in st.session_state:
    st.session_state['doc_id'] = None

if 'loaded_embeddings' not in st.session_state:
    st.session_state['loaded_embeddings'] = None

if 'hash' not in st.session_state:
    st.session_state['hash'] = None

if 'git_rev' not in st.session_state:
    st.session_state['git_rev'] = "unknown"
    if os.path.exists("revision.txt"):
        with open("revision.txt", 'r') as fr:
            from_file = fr.read()
            st.session_state['git_rev'] = from_file if len(from_file) > 0 else "unknown"

if "messages" not in st.session_state:
    st.session_state.messages = []

if 'ner_processing' not in st.session_state:
    st.session_state['ner_processing'] = False

if 'uploaded' not in st.session_state:
    st.session_state['uploaded'] = False

if 'memory' not in st.session_state:
    st.session_state['memory'] = None

if 'binary' not in st.session_state:
    st.session_state['binary'] = None

if 'annotations' not in st.session_state:
    st.session_state['annotations'] = None

if 'should_show_annotations' not in st.session_state:
    st.session_state['should_show_annotations'] = True

if 'pdf' not in st.session_state:
    st.session_state['pdf'] = None

if 'embeddings' not in st.session_state:
    st.session_state['embeddings'] = None

if 'scroll_to_first_annotation' not in st.session_state:
    st.session_state['scroll_to_first_annotation'] = False

st.set_page_config(
    page_title="Scientific Document Insights Q/A",
    page_icon="📝",
    initial_sidebar_state="expanded",
    layout="wide",
    menu_items={
        'Get Help': 'https://github.com/lfoppiano/document-qa',
        'Report a bug': "https://github.com/lfoppiano/document-qa/issues",
        'About': "Upload a scientific article in PDF, ask questions, get insights."
    }
)

st.markdown(
    """
        <style>
               .block-container {
                    padding-top: 3rem;
                    padding-bottom: 1rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """,
    unsafe_allow_html=True
)


def new_file():
    st.session_state['loaded_embeddings'] = None
    st.session_state['doc_id'] = None
    st.session_state['uploaded'] = True
    st.session_state['annotations'] = []
    if st.session_state['memory']:
        st.session_state['memory'].clear()


def clear_memory():
    st.session_state['memory'].clear()


# @st.cache_resource
def init_qa(model_name, embeddings_name):
    st.session_state['memory'] = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    chat = ChatOpenAI(
        model=model_name,
        temperature=0.0,
        base_url=API_MODELS[model_name],
        api_key=os.environ.get('API_KEY')
    )

    embeddings = ModalEmbeddings(
        url=API_EMBEDDINGS[embeddings_name],
        model_name=embeddings_name,
        api_key=os.environ.get('EMBEDS_API_KEY')
    )

    storage = DataStorage(embeddings)
    return DocumentQAEngine(chat, storage, grobid_url=os.environ['GROBID_URL'], memory=st.session_state['memory'])


@st.cache_resource
def init_ner():
    quantities_client = QuantitiesAPI(os.environ['GROBID_QUANTITIES_URL'], check_server=True)

    materials_client = NERClientGeneric(ping=True)
    config_materials = {
        'grobid': {
            "server": os.environ['GROBID_MATERIALS_URL'],
            'sleep_time': 5,
            'timeout': 60,
            'url_mapping': {
                'processText_disable_linking': "/service/process/text?disableLinking=True",
                # 'processText_disable_linking': "/service/process/text"
            }
        }
    }

    materials_client.set_config(config_materials)

    gqa = GrobidAggregationProcessor(grobid_quantities_client=quantities_client,
                                     grobid_superconductors_client=materials_client)
    return gqa


gqa = init_ner()


def get_file_hash(fname):
    hash_md5 = blake2b()
    with open(fname, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def play_old_messages(container):
    if st.session_state['messages']:
        for message in st.session_state['messages']:
            if message['role'] == 'user':
                container.chat_message("user").markdown(message['content'])
            elif message['role'] == 'assistant':
                if mode == "LLM":
                    container.chat_message("assistant").markdown(message['content'], unsafe_allow_html=True)
                else:
                    container.chat_message("assistant").write(message['content'])


# is_api_key_provided = st.session_state['api_key']

with st.sidebar:
    st.title("📝 Document Q/A")
    st.markdown("Upload a scientific article in PDF, ask questions, get insights.")
    st.markdown(
        ":warning: [Usage disclaimer](https://github.com/lfoppiano/document-qa?tab=readme-ov-file#disclaimer-on-data-security-and-privacy-%EF%B8%8F) :warning: ")
    st.markdown("LM and Embeddings are powered by [Modal.com](https://modal.com/)")

    st.divider()
    st.session_state['model'] = model = st.selectbox(
        "Model:",
        options=API_MODELS.keys(),
        index=(list(API_MODELS.keys())).index(
            os.environ["DEFAULT_MODEL"]) if "DEFAULT_MODEL" in os.environ and os.environ["DEFAULT_MODEL"] else 0,
        placeholder="Select model",
        help="Select the LLM model:",
        disabled=st.session_state['doc_id'] is not None or st.session_state['uploaded']
    )

    st.session_state['embeddings'] = embedding_name = st.selectbox(
        "Embeddings:",
        options=API_EMBEDDINGS.keys(),
        index=(list(API_EMBEDDINGS.keys())).index(
            os.environ["DEFAULT_EMBEDDING"]) if "DEFAULT_EMBEDDING" in os.environ and os.environ[
            "DEFAULT_EMBEDDING"] else 0,
        placeholder="Select embedding",
        help="Select the Embedding function:",
        disabled=st.session_state['doc_id'] is not None or st.session_state['uploaded']
    )

    api_key = os.environ['API_KEY']

    if model not in st.session_state['rqa'] or model not in st.session_state['api_keys']:
        with st.spinner("Preparing environment"):
            st.session_state['rqa'][model] = init_qa(model, st.session_state['embeddings'])
            st.session_state['api_keys'][model] = api_key

left_column, right_column = st.columns([5, 4])
right_column = right_column.container(border=True)
left_column = left_column.container(border=True)

with right_column:
    uploaded_file = st.file_uploader(
        "Upload a scientific article",
        type=("pdf"),
        on_change=new_file,
        disabled=st.session_state['model'] is not None and st.session_state['model'] not in
                 st.session_state['api_keys'],
        help="The full-text is extracted using Grobid."
    )

    placeholder = st.empty()
    messages = st.container(height=300)

    question = st.chat_input(
        "Ask something about the article",
        # placeholder="Can you give me a short summary?",
        disabled=not uploaded_file
    )

query_modes = {
    "llm": "LLM Q/A",
    "embeddings": "Embeddings",
    "question_coefficient": "Question coefficient"
}

with st.sidebar:
    st.header("Settings")
    mode = st.radio(
        "Query mode",
        ("llm", "embeddings", "question_coefficient"),
        disabled=not uploaded_file,
        index=0,
        horizontal=True,
        format_func=lambda x: query_modes[x],
        help="LLM will respond the question, Embedding will show the "
             "relevant paragraphs to the question in the paper. "
             "Question coefficient attempt to estimate how effective the question will be answered."
    )
    st.session_state['scroll_to_first_annotation'] = st.checkbox(
        "Scroll to context",
        help='The PDF viewer will automatically scroll to the first relevant passage in the document.'
    )
    st.session_state['ner_processing'] = st.checkbox(
        "Identify materials and properties.",
        help='The LLM responses undergo post-processing to extract physical quantities, measurements, and materials mentions.'
    )

    # Add a checkbox for showing annotations
    # st.session_state['show_annotations'] = st.checkbox("Show annotations", value=True)
    # st.session_state['should_show_annotations'] = st.checkbox("Show annotations", value=True)

    chunk_size = st.slider("Text chunks size", -1, 2000, value=-1,
                           help="Size of chunks in which split the document. -1: use paragraphs, > 0 paragraphs are aggregated.",
                           disabled=uploaded_file is not None)
    if chunk_size == -1:
        context_size = st.slider("Context size (paragraphs)", 3, 20, value=10,
                                 help="Number of paragraphs to consider when answering a question",
                                 disabled=not uploaded_file)
    else:
        context_size = st.slider("Context size (chunks)", 3, 10, value=4,
                                 help="Number of chunks to consider when answering a question",
                                 disabled=not uploaded_file)

    st.divider()

    st.header("Documentation")
    st.markdown("https://github.com/lfoppiano/document-qa")
    st.markdown(
        """Upload a scientific article as PDF document. Once the spinner stops, you can proceed to ask your questions.""")

    if st.session_state['git_rev'] != "unknown":
        st.markdown("**Revision number**: [" + st.session_state[
            'git_rev'] + "](https://github.com/lfoppiano/document-qa/commit/" + st.session_state['git_rev'] + ")")

if uploaded_file and not st.session_state.loaded_embeddings:
    if model not in st.session_state['api_keys']:
        st.error("Before uploading a document, you must enter the API key. ")
        st.stop()

    with left_column:
        try:
            with st.spinner('Reading file, calling Grobid, and creating in-memory embeddings...'):
                binary = uploaded_file.getvalue()
                tmp_file = NamedTemporaryFile()
                tmp_file.write(bytearray(binary))
                st.session_state['binary'] = binary

                st.session_state['doc_id'] = hash = st.session_state['rqa'][model].create_memory_embeddings(
                    tmp_file.name,
                    chunk_size=chunk_size,
                    perc_overlap=0.1
                )
                st.session_state['loaded_embeddings'] = True
                st.session_state.messages = []
        except GrobidServiceError as exc:
            message = str(exc).strip() or "Grobid is not responding"
            status = f" (status {exc.status_code})" if exc.status_code else ""
            st.session_state['doc_id'] = None
            st.session_state['loaded_embeddings'] = False
            st.session_state['uploaded'] = False
            st.error(f"{message} Please try later.")
            st.stop()


def rgb_to_hex(rgb):
    return "#{:02x}{:02x}{:02x}".format(*rgb)


def generate_color_gradient(num_elements):
    # Define warm and cold colors in RGB format
    warm_color = (255, 165, 0)  # Orange
    cold_color = (0, 0, 255)  # Blue

    # Generate a linear gradient of colors
    color_gradient = [
        rgb_to_hex(tuple(int(warm * (1 - i / num_elements) + cold * (i / num_elements)) for warm, cold in
                         zip(warm_color, cold_color)))
        for i in range(num_elements)
    ]

    return color_gradient


with right_column:
    if st.session_state.loaded_embeddings and question and len(question) > 0 and st.session_state.doc_id:
        st.session_state.messages.append({"role": "user", "mode": mode, "content": question})

        for message in st.session_state.messages:
            # with messages.chat_message(message["role"]):
            if message['mode'] == "llm":
                messages.chat_message(message["role"]).markdown(message["content"], unsafe_allow_html=True)
            elif message['mode'] == "embeddings":
                messages.chat_message(message["role"]).write(message["content"])
            elif message['mode'] == "question_coefficient":
                messages.chat_message(message["role"]).markdown(message["content"], unsafe_allow_html=True)
        if model not in st.session_state['rqa']:
            st.error("The API Key for the " + model + " is  missing. Please add it before sending any query. `")
            st.stop()

        text_response = None
        if mode == "embeddings":
            with placeholder:
                with st.spinner("Fetching the relevant context..."):
                    text_response, coordinates = st.session_state['rqa'][model].query_storage(
                        question,
                        st.session_state.doc_id,
                        context_size=context_size
                    )
        elif mode == "llm":
            with placeholder:
                with st.spinner("Generating LLM response..."):
                    _, text_response, coordinates = st.session_state['rqa'][model].query_document(
                        question,
                        st.session_state.doc_id,
                        context_size=context_size
                    )

        elif mode == "question_coefficient":
            with st.spinner("Estimate question/context relevancy..."):
                text_response, coordinates = st.session_state['rqa'][model].analyse_query(
                    question,
                    st.session_state.doc_id,
                    context_size=context_size
                )

        annotations = [[GrobidAggregationProcessor.box_to_dict([cs for cs in c.split(",")]) for c in coord_doc]
                       for coord_doc in coordinates]
        gradients = generate_color_gradient(len(annotations))
        for i, color in enumerate(gradients):
            for annotation in annotations[i]:
                annotation['color'] = color
                if i == 0:
                    annotation['border'] = "dotted"

        st.session_state['annotations'] = [annotation for annotation_doc in annotations for annotation in
                                           annotation_doc]

        if not text_response:
            st.error("Something went wrong. Contact info AT sciencialab.com to report the issue through GitHub.")

        if mode == "llm":
            if st.session_state['ner_processing']:
                with st.spinner("Processing NER on LLM response..."):
                    entities = gqa.process_single_text(text_response)
                    decorated_text = decorate_text_with_annotations(text_response.strip(), entities)
                    decorated_text = decorated_text.replace('class="label material"', 'style="color:green"')
                    decorated_text = re.sub(r'class="label[^"]+"', 'style="color:orange"', decorated_text)
                    text_response = decorated_text
            messages.chat_message("assistant").markdown(text_response, unsafe_allow_html=True)
        else:
            messages.chat_message("assistant").write(text_response)
        st.session_state.messages.append({"role": "assistant", "mode": mode, "content": text_response})

    elif st.session_state.loaded_embeddings and st.session_state.doc_id:
        play_old_messages(messages)

with left_column:
    if st.session_state['binary']:
        with st.container(height=600):
            pdf_viewer(
                input=st.session_state['binary'],
                annotation_outline_size=2,
                annotations=st.session_state['annotations'] if st.session_state['annotations'] else [],
                render_text=True,
                scroll_to_annotation=1 if (st.session_state['annotations'] and st.session_state[
                    'scroll_to_first_annotation']) else None
            )
