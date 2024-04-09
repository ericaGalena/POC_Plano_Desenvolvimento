import streamlit as st

def intro():
    import streamlit as st

    st.write("Plano de Desenvolvimento 👈")
    st.sidebar.success("Selecione o módulo desejado.")

    st.markdown(
        """
        Esse é um coach de carreiras que vai te ajudar no seu Desenvovimento Profissional. 
    """
    )

def upload_curriculo():
  import streamlit as st
  import os
  import google.generativeai as genai
  from langchain_community.document_loaders import PyPDFLoader
  from langchain_text_splitters import CharacterTextSplitter
  from langchain.vectorstores import Chroma
  from langchain_google_genai import ChatGoogleGenerativeAI
  from langchain_google_genai import GoogleGenerativeAIEmbeddings
  from langchain.prompts import PromptTemplate
  from langchain.chains.combine_documents import create_stuff_documents_chain
  from langchain.chains import create_retrieval_chain




  # Initialize Gemini-Pro
  genai.configure(api_key='AIzaSyBMdfvPoNMPilhfGIP-e-VDnnasZ-uHi7A')




  # Gemini uses 'model' for assistant; Streamlit uses 'assistant'
  def role_to_streamlit(role):
    if role == "model":
      return "assistant"
    else:
      return role




  uploaded_file = st.file_uploader("Selecione o arquivo do seu currículo:", type="pdf")


  if uploaded_file :


    temp_file = "./temp.pdf"
    with open(temp_file, "wb") as file:
      file.write(uploaded_file.getvalue())
      file_name = uploaded_file.name


    loader = PyPDFLoader(temp_file)
 
    text_splitter = CharacterTextSplitter(
      separator=".",
      chunk_size=700,
      chunk_overlap=50,
      length_function=len,
    is_separator_regex=False,
    )
    pages = loader.load_and_split(text_splitter)

    os.environ["GOOGLE_API_KEY"]='AIzaSyBMdfvPoNMPilhfGIP-e-VDnnasZ-uHi7A'
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectordb=Chroma.from_documents(pages,embeddings)
    retriever = vectordb.as_retriever(search_kwargs={"k": 30})



    #Create the retrieval chain
    template = """
      Você é um coach de carreira. Sua função é indicar quais cursos, formações, mentorias um
      colbarador deve fazer para atingir um objetivo.
      O documento que você recebeu é um currículo exportado a partir do Linkedln. Extrai as informações sobre a experiência da pessoa.
      context: {context}
      input: {input}
      answer:
    """
    prompt = PromptTemplate.from_template(template)
    combine_docs_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)



    #Invoke the retrieval chain
    response=retrieval_chain.invoke({"input":"Qual experiência da Erica?"})
    st.markdown("Qual experiência da Erica?")


    #Print the answer to the question
    st.markdown(response["answer"])
    st.markdown("Quais as habilidades da Erica?")




    #Invoke the retrieval chain
    response=retrieval_chain.invoke({"input":"Quais as habilidades da Erica?"})


    #Print the answer to the question
    st.markdown(response["answer"])
    habilidades=response["answer"]
 
  
def escolha_objetivos():
  genre = st.radio(
     "Qual seu objetivo? :question: ",
    ["Progredir na sua carreira :dart:", "Mudar de carreira :thought_balloon:", "Desenvolvimento Pessoal  :brain:"],
    index=None,
  )

  st.write("Você selecionou:", genre)


page_names_to_funcs = {
    "—": intro,
    "Upload do currículo": upload_curriculo,
    "Escolha de objetivos": escolha_objetivos
}

demo_name = st.sidebar.selectbox("Escolha um módulo:", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()



