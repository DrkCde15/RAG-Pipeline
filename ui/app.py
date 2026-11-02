import os

import httpx
import streamlit as st

from ui.api_client import APIError, RAGAPIClient

st.set_page_config(page_title="RAG Pipeline", page_icon="R", layout="wide")


@st.cache_resource
def get_api_client() -> RAGAPIClient:
    return RAGAPIClient(os.getenv("API_BASE_URL", "http://localhost:8000"))


def show_api_error(error: Exception) -> None:
    st.error(f"Não foi possível concluir a operação: {error}")


api = get_api_client()
st.title("RAG Pipeline")
st.caption("Consulte seus documentos com respostas fundamentadas e fontes.")

try:
    health = api.health()
    if health.get("api") == "ok":
        st.sidebar.success("API conectada")
    else:
        st.sidebar.warning(f"Estado da API: {health}")
except (APIError, httpx.RequestError) as error:
    st.sidebar.error(f"API indisponível: {error}")

ingestion_tab, chat_tab, documents_tab = st.tabs(["Ingestão", "Chat", "Documentos"])

with ingestion_tab:
    st.subheader("Adicionar um documento")
    st.write("Envie um arquivo PDF, TXT ou Markdown para indexá-lo no Qdrant.")
    uploaded_file = st.file_uploader(
        "Arquivo",
        type=["pdf", "txt", "md", "markdown"],
        key="document_upload",
    )
    if st.button("Ingerir documento", type="primary", disabled=uploaded_file is None):
        try:
            result = api.ingest_file(uploaded_file.name, uploaded_file.getvalue())
            st.success(f"Documento processado: {result['chunks_indexed']} chunks indexados.")
            st.code(result["document_ids"][0])
        except (APIError, httpx.RequestError, KeyError, IndexError) as error:
            show_api_error(error)

with chat_tab:
    st.subheader("Pergunte aos seus documentos")
    with st.form("chat_form"):
        question = st.text_area(
            "Pergunta",
            placeholder="Ex.: Quais são os principais tópicos deste documento?",
            max_chars=4000,
        )
        submit_question = st.form_submit_button("Perguntar", type="primary")

    if submit_question:
        if not question.strip():
            st.warning("Digite uma pergunta antes de enviar.")
        else:
            with st.spinner("Pesquisando documentos e gerando resposta..."):
                try:
                    result = api.chat(question.strip())
                    st.markdown("#### Resposta")
                    st.write(result["answer"])
                    st.caption(f"Tempo de processamento: {result['processing_time']:.2f} s")
                    st.markdown("#### Fontes")
                    if not result["sources"]:
                        st.info("Nenhuma fonte relevante foi encontrada.")
                    for index, source in enumerate(result["sources"], start=1):
                        page = f" — página {source['page']}" if source.get("page") else ""
                        with st.expander(
                            f"[{index}] {source['filename']}{page} · score {source['score']:.3f}"
                        ):
                            st.write(source["excerpt"])
                            st.caption(f"ID do chunk: {source['chunk_id']}")
                except (APIError, httpx.RequestError, KeyError, TypeError, ValueError) as error:
                    show_api_error(error)

with documents_tab:
    st.subheader("Documentos indexados")
    try:
        documents = api.list_documents()
        if not documents:
            st.info("Ainda não há documentos indexados. Envie um na aba Ingestão.")
        for document in documents:
            with st.container(border=True):
                details, action = st.columns([5, 1])
                with details:
                    st.markdown(f"**{document['filename']}**")
                    st.caption(
                        f"{document['chunk_count']} chunks · "
                        f"adicionado em {document['ingestion_timestamp']}"
                    )
                    st.code(document["document_id"])
                with action:
                    if st.button(
                        "Remover",
                        key=f"delete_{document['document_id']}",
                        type="secondary",
                    ):
                        try:
                            result = api.delete_document(document["document_id"])
                            if result["deleted"]:
                                st.success(f"{document['filename']} removido.")
                            else:
                                st.warning("O documento já não estava indexado.")
                            st.rerun()
                        except (APIError, httpx.RequestError, KeyError) as error:
                            show_api_error(error)
    except (APIError, httpx.RequestError, KeyError, TypeError) as error:
        show_api_error(error)
