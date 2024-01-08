import streamlit as st
import os
import tempfile
from process_text import question_answer
from datetime import datetime

#formato_ptb = "%d/%m/%Y %H:%M:%S"
formato_ptb = "%H:%M:%S"

# Configurações da página
st.set_page_config(
    page_title="LangChain - Perguntas e Respostas - Q&A",
    layout='wide'
)

# Sidebar
st.sidebar.write("**Configurações**")
api_key = st.sidebar.text_input("OpenAI API Key", type='password')
file_input = st.sidebar.file_uploader("Arquivo")
chunk = st.sidebar.slider("Chunk", min_value=0, max_value=100, step=5, value=5)
chain_type = st.sidebar.radio("Tipo Chain", options=['stuff', 'map_reduce', 'refine', 'map_rerank'])

# Body
st.header("LangChain :speech_balloon: - Perguntas e Respostas - Q&A", divider='gray')

if 'historico' not in st.session_state:
    st.session_state.historico = [{"role" : "assistant", "content" : f"{datetime.now().__format__(formato_ptb)} Olá. Sou seu assistente virtual ChatBot. Como posso te ajudar?"}]

for h in st.session_state.historico:
    st.chat_message(h["role"]).write(h["content"])

# Entrada de texto para a pergunta
pergunta = st.chat_input(placeholder="Coloque aqui a sua pergunta")

#função que preenche o histórico do chat
if pergunta:
    data_hora_pergunta = datetime.now().__format__(formato_ptb)

    with st.spinner("Em processamento"):
        st.session_state.historico.append({"role" : "user", "content" : f"{data_hora_pergunta} {pergunta}"})

        if len(api_key) == 0:
            st.session_state.historico.append({"role" : "assistant", "content" : f"{data_hora_pergunta} **ATENÇÃO**, Para utilizar o chat é necessário preencher a *OpenAI API Key*!"})

        else:            
            temp_file_path = os.path.join(tempfile.gettempdir(), "temp.pdf")

            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_input.read())
                
            # configurar a chave API da OpenAI
            os.environ["OPENAI_API_KEY"] = api_key
            
            # executar a função de perguntas e respostas
            try:
                resultados = question_answer(file_path=temp_file_path, query=pergunta,
                        chain_type=chain_type,
                        n_chunks=chunk)
            except Exception as e:
                st.error(f"Ocorreu um erro interno: {e}") 
                
            # Atualiza a variável chat apenas com a última pergunta
            data_hora_resposta = datetime.now().__format__(formato_ptb)
            st.session_state.historico.append({"role" : "assistant", "content" : f"{data_hora_resposta} {resultados['result']}"})
                    
            # atualiza a tela para exibir a resposta no histórico da conversa
        st.rerun()


