import streamlit as st
import os
import tempfile
from process_text import question_answer
from datetime import datetime


# Configurações da página
st.set_page_config(
    page_title="LangChain - Perguntas e Respostas - Q&A",
    layout='wide'
)

# Define a custom HTML template with a script to scroll the text area
scroll_script = """
<script>
  var textArea = document.getElementById('text_area_chat');
  textArea.scrollTop = textArea.scrollHeight;
</script>
"""

# Sidebar
st.sidebar.write("**Configurações**")
api_key = st.sidebar.text_input("OpenAI API Key", type='password')
file_input = st.sidebar.file_uploader("Arquivo")
chunk = st.sidebar.slider("Chunk", min_value=0, max_value=100, step=5, value=5)
chain_type = st.sidebar.radio("Tipo Chain", options=['stuff', 'map_reduce', 'refine', 'map_rerank'])

# Body
st.header("LangChain :speech_balloon: - Perguntas e Respostas - Q&A", divider='gray')

if 'historico' not in st.session_state:
    st.session_state.historico = [f"{datetime.now().__format__('%d/%m/%Y %H:%M:%S')} 😊 Olá. Sou seu assistente virtual ChatBot. Como posso te ajudar?"]
    
st.text_area("Chat", value="".join(st.session_state.historico), height=400, key='text_area_chat')


# Entrada de texto para a pergunta
pergunta = st.text_input(label="Envie sua pergunta", placeholder="Coloque aqui a sua pergunta")

# Botão para enviar
btn_enviar = st.button("Enviar")

#função que preenche o histórico do chat
if (btn_enviar and pergunta and api_key):
    data_hora_pergunta = datetime.now().__format__("%d/%m/%Y %H:%M:%S")
    if len(api_key) == 0:
        st.sidebar.error("'OpenAI API Key' é obrigatório.")
        
    else:
        
        with st.spinner("Em processamento"):
        
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
            data_hora_resposta = datetime.now().__format__("%d/%m/%Y %H:%M:%S")
            st.session_state.historico.insert(0, f"{data_hora_resposta} 😊 Atendente: {resultados['result']}\n\n{data_hora_pergunta} 🌲 Usuário: {pergunta}\n\n")
            
            # Inclua o script para rolar a área de texto até o final
            st.markdown(scroll_script, unsafe_allow_html=True)
            
            # atualiza a tela para exibir a resposta no histórico da conversa
            st.rerun()
            
        #limpa o campo pergunta


