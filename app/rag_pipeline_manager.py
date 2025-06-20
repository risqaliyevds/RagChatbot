"""
Chat Service and LLM Integration
===============================

Handles chat logic, LLM initialization, and RAG chain creation.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import Document

from .models import ChatMessage
from .qdrant_manager import SimpleQdrantVectorStore

logger = logging.getLogger(__name__)


def init_llm(config: Dict[str, Any]) -> ChatOpenAI:
    """Initialize the language model"""
    try:
        llm = ChatOpenAI(
            openai_api_base=config.get("vllm_chat_endpoint", "http://localhost:8000/v1"),
            openai_api_key=config.get("vllm_api_key", "EMPTY"),
            model_name=config.get("chat_model", "google/gemma-3-12b-it"),
            temperature=0.7,
            max_tokens=512,
            request_timeout=60,  # Increased timeout for vLLM
            max_retries=3,       # Add retry logic
        )
        logger.info(f"Initialized LLM: {config.get('chat_model', 'google/gemma-3-12b-it')}")
        logger.info(f"vLLM endpoint: {config.get('vllm_chat_endpoint', 'http://localhost:8000/v1')}")
        return llm
    except Exception as e:
        logger.error(f"Failed to initialize LLM: {e}")
        raise


def get_qa_prompt() -> PromptTemplate:
    """Get the QA prompt template"""

    template = """Siz veb-sayt yordamchi chatbotisiz. Foydalanuvchilarga sayt bo'yicha navigatsiya qilish va kerakli ma'lumotlarni topishda yordam berasiz.

KONTEKST: {context}
SUHBAT TARIXI: {chat_history}
FOYDALANUVCHI XABARI: {question}

JAVOB BERISH QOIDALARI:
1. TILNI ANIQLASH: Foydalanuvchi qaysi tilda yozgan bo'lsa, o'sha tilda javob bering
   - O'zbek (lotin/kirill), Rus, Ingliz tillarini qo'llab-quvvatlaysiz

2. SALOMLASHISH VA XUSHMUOMALA:
   - Salom, rahmat, xayr kabi so'zlarga mos javob bering
   - Sayt kontekstida professional ohangda gapiring

3. JAVOB BERISH MANTIQ - QATTIQ QOIDALAR:
   - FAQAT KONTEKST va SUHBAT TARIXIdagi sayt ma'lumotlaridan foydalaning
   - KONTEKST bo'sh bo'lsa: FAQAT salomlashish, rahmat, xayrlashish kabi oddiy muloqotga javob bering
   - KONTEKST bo'sh bo'lganda HECH QANDAY umumiy ma'lumot (paytaxt, tarix, fan va boshqalar) BERMANG
   - Agar savolga javob saytda yo'q bo'lsa: "Bunday ma'lumotlar bizning bazamizda yo'q. Iltimos, kerakli hujjatlarni bazaga yuklang, ushanda sizga aniqroq javob bera olaman."
   - Navigatsiya yordam: faqat kontekstdagi sahifa, link, bo'limlarni ko'rsating

4. JAVOB SIFATI:
   - Aniq, qisqa va amaliy
   - Faqat sayt ma'lumotlari: sahifa linkları, bo'lim nomlari, xizmat tavsiflarini bering
   - Qadamlar bo'yicha yo'l-yo'riq (agar sayt kontekstida bo'lsa)
   - Sayt ichida qidirishga yordam bering

5. VEB-SAYT YORDAMCHISI MAXSUS VAZIFALAR:
   - Salomlashish → "Saytimizga xush kelibsiz! Sizga sayt bo'yicha qanday yordam bera olaman?"
   - Rahmat → "Arzimaydi! Sayt haqida boshqa savollaringiz bo'lsa, so'rang"
   - Xayrlashish → "Xayr! Sayt haqida keyinroq savollaringiz bo'lsa, qaytib keling"
   - KONTEKST bo'sh + sayt tashqaridagi savol → "Kechirasiz, men faqat bazamizdagi ma'lumotlar bo'yicha yordam bera olaman. Kerakli hujjatlarni bazaga yuklang."
   - Noaniq savol → "Saytning qaysi bo'limi haqida ma'lumot olmoqchisiz?"

JAVOB:"""

    return PromptTemplate(
        template=template,
        input_variables=["context", "chat_history", "question"]
    )


def format_docs(docs: List[Document]) -> str:
    """Format documents for the prompt"""
    return "\n\n".join(doc.page_content for doc in docs)


def clean_response(response: str) -> str:
    """Clean and format the response"""
    # Remove any unwanted prefixes or suffixes
    response = response.strip()
    
    # Remove common AI assistant prefixes
    prefixes_to_remove = [
        "JAVOB:",
        "Javob:",
        "Assistant:",
        "AI:",
        "Bot:",
    ]
    
    for prefix in prefixes_to_remove:
        if response.startswith(prefix):
            response = response[len(prefix):].strip()
    
    return response


def format_chat_history(messages: List[ChatMessage]) -> str:
    """Format chat history for the prompt"""
    if not messages:
        return "Suhbat yangi boshlandi."
    
    formatted_history = []
    for msg in messages[-5:]:  # Keep only last 5 messages
        role_name = "Foydalanuvchi" if msg.role == "user" else "Yordamchi"
        formatted_history.append(f"{role_name}: {msg.content}")
    
    return "\n".join(formatted_history)


def create_qa_chain(vectorstore: SimpleQdrantVectorStore, llm: ChatOpenAI, prompt: PromptTemplate, config: Dict[str, Any]):
    """Create QA chain function"""
    
    def qa_function(question: str, chat_history: List[ChatMessage] = None) -> str:
        """Process a question and return an answer"""
        try:
            # Search for relevant documents
            top_k = config.get("top_k", 3)
            docs = vectorstore.search(question, k=top_k)
            
            # Format context (empty string if no docs found)
            context = format_docs(docs) if docs else ""
            
            # Format chat history
            history_text = format_chat_history(chat_history or [])
            
            # Create prompt
            formatted_prompt = prompt.format(
                context=context,
                chat_history=history_text,
                question=question
            )
            
            logger.debug(f"Generated prompt for question: {question[:50]}...")
            
            # Get response from LLM
            try:
                response = llm.invoke(formatted_prompt)
                answer = response.content if hasattr(response, 'content') else str(response)
                
                # Clean response
                answer = clean_response(answer)
                
                # Add source information if enabled
                if config.get("include_sources", False):
                    sources = set()
                    for doc in docs:
                        source = doc.metadata.get("source", "Unknown")
                        if source != "Unknown":
                            sources.add(source)
                    
                    if sources:
                        answer += f"\n\nManbalar: {', '.join(list(sources)[:3])}"
                
                logger.debug(f"Generated answer: {answer[:100]}...")
                return answer
                
            except Exception as e:
                logger.error(f"LLM invocation failed: {e}")
                return "Kechirasiz, hozir javob tayyorlashda muammo yuz berdi. Iltimos, qayta urinib ko'ring."
        
        except Exception as e:
            logger.error(f"QA function failed: {e}")
            return "Kechirasiz, savolni qayta ishlashda xatolik yuz berdi. Iltimos, qayta urinib ko'ring."
    
    return qa_function


class ChatService:
    """Service for handling chat operations"""
    
    def __init__(self, qa_chain, db_manager):
        self.qa_chain = qa_chain
        self.db_manager = db_manager
    
    async def process_message(self, user_id: str, message: str, chat_id: str = None) -> Dict[str, Any]:
        """Process a user message and return response"""
        try:
            # Get or create chat session
            session = self.db_manager.get_or_create_session(user_id, chat_id)
            if not session:
                raise Exception("Failed to create chat session")
            
            chat_id = session["chat_id"]
            
            # Add user message to database
            self.db_manager.add_message(chat_id, "user", message)
            
            # Get chat history for context
            messages = self.db_manager.get_chat_messages(chat_id, limit=10)
            chat_history = [
                ChatMessage(
                    role=msg["role"],
                    content=msg["content"],
                    timestamp=datetime.fromisoformat(msg["timestamp"]) if msg.get("timestamp") else datetime.now(timezone.utc)
                )
                for msg in messages
            ]
            
            # Generate response using QA chain
            response = self.qa_chain(message, chat_history[:-1])  # Exclude current message
            
            # Add bot response to database
            self.db_manager.add_message(chat_id, "assistant", response)
            
            return {
                "chat_id": chat_id,
                "user_id": user_id,
                "message": response,
                "timestamp": datetime.now(timezone.utc)
            }
            
        except Exception as e:
            logger.error(f"Failed to process message: {e}")
            raise
    
    def create_new_chat(self, user_id: str) -> Dict[str, Any]:
        """Create a new chat session"""
        try:
            chat_id = self.db_manager.create_new_chat_for_user(user_id)
            if not chat_id:
                raise Exception("Failed to create new chat")
            
            # Get session details
            session = self.db_manager.get_chat_session(chat_id)
            if not session:
                raise Exception("Failed to retrieve created session")
            
            welcome_message = "Assalomu alaykum! Men sizning professional yordamchingiz \"Assistent Bot\"man. Qanday yordam bera olaman?"
            
            # Add welcome message
            self.db_manager.add_message(chat_id, "assistant", welcome_message)
            
            return {
                "chat_id": chat_id,
                "user_id": user_id,
                "message": welcome_message,
                "created_at": session["created_at"],
                "last_activity": session["last_activity"]
            }
            
        except Exception as e:
            logger.error(f"Failed to create new chat: {e}")
            raise
    
    def get_user_session_status(self, user_id: str) -> Dict[str, Any]:
        """Get user session status"""
        try:
            session = self.db_manager.get_active_user_session(user_id)
            
            if session:
                # Check if session is expired (more than 1 hour of inactivity)
                last_activity = datetime.fromisoformat(session["last_activity"])
                # Make both datetimes timezone-aware for comparison
                if last_activity.tzinfo is None:
                    last_activity = last_activity.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                time_diff = now - last_activity
                session_expired = time_diff.total_seconds() > 3600  # 1 hour
                
                return {
                    "user_id": user_id,
                    "has_active_session": True,
                    "active_chat_id": session["chat_id"],
                    "last_activity": session["last_activity"],
                    "session_expired": session_expired
                }
            else:
                return {
                    "user_id": user_id,
                    "has_active_session": False,
                    "active_chat_id": None,
                    "last_activity": None,
                    "session_expired": False
                }
                
        except Exception as e:
            logger.error(f"Failed to get user session status: {e}")
            raise
    
    def get_chat_history(self, user_id: str, chat_id: str) -> List[Dict[str, Any]]:
        """Get chat history for a specific chat"""
        try:
            # Verify the chat belongs to the user
            session = self.db_manager.get_chat_session(chat_id)
            if not session or session["user_id"] != user_id:
                raise Exception("Chat not found or access denied")
            
            messages = self.db_manager.get_chat_messages(chat_id)
            return messages
            
        except Exception as e:
            logger.error(f"Failed to get chat history: {e}")
            raise
    
    def get_user_chats(self, user_id: str) -> List[Dict[str, Any]]:
        """Get all chats for a user"""
        try:
            chats = self.db_manager.get_user_chats(user_id)
            return chats
            
        except Exception as e:
            logger.error(f"Failed to get user chats: {e}")
            raise 