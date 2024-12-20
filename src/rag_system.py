import os
from typing import List, Dict
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class TranscriptRAG:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings()
        # Breaking up the text into smaller pieces (chunks) so it's easier to search through and answer questions.
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2500, #Size of each chunk, feel free to modify this. Make sure it's not too small or too large.
            #The cost associated with each call to the LLM and the speed of the system depends on this.
            chunk_overlap=200, #Overlap between chunks, this can be modified as well
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
        # Using GPT-4 for better answers
        self.llm = ChatOpenAI(temperature=0.7, model="gpt-4-turbo-preview")
        # Keeping track of the covnersation history
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

    def prepare_documents(self, transcription: str, speakers_data: Dict):
        """
        Chunking the transrcipt into smaller pieces.
        """
        documents = []
        
        # Process full transcription
        full_transcript_chunks = self.text_splitter.create_documents(
            [transcription],
            metadatas=[{"source": "full_transcript", "type": "complete"}]
        )
        documents.extend(full_transcript_chunks)
        
        # Process speaker-wise transcripts
        for speaker, data in speakers_data.items():
            speaker_chunks = self.text_splitter.create_documents(
                [data["text"]],
                metadatas=[{
                    "source": "speaker_transcript",
                    "speaker": speaker,
                    "type": "speaker_specific"
                }]
            )
            documents.extend(speaker_chunks)
        
        return documents

    def create_vector_store(self, documents):
        """
     Store our text chunks in a Chrome vector database. This can be changed to any other vector database like FAISS etc.
        """
        return Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory="./data/vectorstore"
        )

    def setup_retriever(self, vector_store):
        """
        Set up the system that will find relevant chunks.
        """
        base_retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": 4, # Number of chunks to retrieve, modify this as needed
                "score_threshold": 0.5 #Threshold for similarity score, feel free to change this, the higher you make this, the more relevant chunks you'll get.
            }
        )
        return base_retriever

    def setup_qa_chain(self, retriever):
        """
        Create the question answering sytsem
        This combines everything (finding relevant text + generating answers)
        """
        
        # Custom prompt template for better context utilization
        prompt_template = """
        You are an AI assistant analyzing a conversation transcript. Use the following pieces of context to answer the question. 
        If you don't know the answer, just say that you don't know. Don't try to make up an answer. 
        
        Context: {context}
        
        Chat History: {chat_history}
        
        Question: {question}
        
        When answering:
        1. If the context mentions specific speakers, include who said what. Be specific about which speaker said what. e.g 'Speaker A: "Hello, how are you?"
        2. Provide direct quotes when relevant
        3. Be concise but comprehensive
        4. If the answer requires multiple points, use a numbered list
        5. Search the web or use external knowledge for background information about a particular topic, but make sure to demarcate this as background information clearly!
        6. If no relevant information was found in the transcript, only provide this background information but mention that not relevant information was found in the transcript!
        Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "chat_history", "question"]
        )

        # Conversation buffer memory to keep track of the convo histroy
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            output_key="answer",
            return_messages=True
        )

        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": PROMPT},
            return_source_documents=True,
            verbose=True
        )

    def query(self, qa_chain, question: str) -> Dict:
        """
        Takes a question and returns both an answer and where it found the information
        """
        result = qa_chain({"question": question})
        
        # Extract source information
        sources = []
        for doc in result.get("source_documents", []):
            source_info = {
                "text": doc.page_content,
                "metadata": doc.metadata
            }
            sources.append(source_info)
        
        return {
            "answer": result["answer"],
            "sources": sources
        }

def initialize_rag_system(transcription: str, speakers_data: Dict):
    """
    Setting up the whole system
    """
    rag = TranscriptRAG()
    documents = rag.prepare_documents(transcription, speakers_data)
    vector_store = rag.create_vector_store(documents)
    retriever = rag.setup_retriever(vector_store)
    qa_chain = rag.setup_qa_chain(retriever)
    
    return rag, qa_chain 