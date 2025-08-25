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
    """Initialize the language model with fallback support"""
    
    # For immediate functionality, use mock LLM directly
    # TODO: Enable external vLLM when service is available
    logger.info("🔧 Using local mock LLM for immediate functionality")
    return _create_mock_llm(config)
    
    # Commented out external LLM code for now
    """
    try:
        # First, try to initialize with external vLLM
        vllm_endpoint = config.get("vllm_chat_endpoint", "http://localhost:8000/v1")
        
        # Fix common URL formatting issues
        if not vllm_endpoint.startswith('http'):
            vllm_endpoint = f"http://{vllm_endpoint}"
        
        # Quick connection test first
        try:
            import requests
            test_url = vllm_endpoint.replace('/v1', '/health')  # Try health endpoint first
            response = requests.get(test_url, timeout=3)
            if response.status_code != 200:
                raise ConnectionError(f"vLLM service not available (status: {response.status_code})")
        except Exception as conn_e:
            logger.warning(f"⚠️ vLLM service not accessible: {conn_e}")
            return _create_mock_llm(config)
        
        llm = ChatOpenAI(
            openai_api_base=vllm_endpoint,
            openai_api_key=config.get("vllm_api_key", "EMPTY"),
            model_name=config.get("chat_model", "google/gemma-3-12b-it"),
            temperature=config.get("llm_temperature", 0.7),
            max_tokens=config.get("llm_max_tokens", 2048),
            request_timeout=5,  # Short timeout for quick failure
            max_retries=1,  # Only one retry
        )
        
        # Quick test with very short timeout
        try:
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("LLM test timeout")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(3)  # 3 second timeout
            
            test_response = llm.invoke("test")
            signal.alarm(0)  # Cancel timeout
            
            logger.info(f"✅ External LLM connected successfully: {config.get('chat_model', 'google/gemma-3-12b-it')}")
            logger.info(f"vLLM endpoint: {vllm_endpoint}")
            return llm
            
        except Exception as test_e:
            signal.alarm(0)  # Cancel timeout
            logger.warning(f"⚠️ External LLM test failed: {test_e}")
            # Fall back to local mock LLM
            return _create_mock_llm(config)
            
    except Exception as e:
        logger.warning(f"⚠️ LLM initialization failed: {e} - Using local mock LLM")
        return _create_mock_llm(config)
    """


def _create_mock_llm(config: Dict[str, Any]):
    """Create a mock LLM for local operation"""
    
    class MockLLM:
        """Professional mock LLM that provides intelligent responses"""
        
        def __init__(self, config):
            self.config = config
            self.model_name = config.get("chat_model", "local-mock-llm")
            logger.info(f"🔧 Initialized Mock LLM: {self.model_name}")
        
        def invoke(self, prompt: str):
            """Generate intelligent response based on prompt analysis"""
            
            class MockResponse:
                def __init__(self, content):
                    self.content = content
            
            # Extract the actual question from the prompt
            lines = prompt.split('\n')
            question = ""
            context = ""
            
            for line in lines:
                if line.startswith('FOYDALANUVCHI XABARI:'):
                    question = line.replace('FOYDALANUVCHI XABARI:', '').strip()
                elif line.startswith('KONTEKST:'):
                    context = line.replace('KONTEKST:', '').strip()
            
            # Generate intelligent response based on question analysis
            response = self._generate_intelligent_response(question, context)
            return MockResponse(response)
        
        def _generate_intelligent_response(self, question: str, context: str) -> str:
            """Generate contextually appropriate responses"""
            question_lower = question.lower().strip()
            
            # Greeting responses
            greetings = ['salom', 'hello', 'assalomu alaykum', 'hi', 'привет', 'салом', 'hey']
            if any(greeting in question_lower for greeting in greetings):
                return "Assalomu alaykum! Men sizning AI yordamchingizman. Qanday yordam bera olaman?"
            
            # Thank you responses
            thanks = ['rahmat', 'thank', 'спасибо', 'рахмат', 'tnx', 'thanks', 'tashakkur']
            if any(thank in question_lower for thank in thanks):
                return "Arzimaydi! Yana savollaringiz bo'lsa, bemalol so'rang."
            
            # Goodbye responses
            goodbyes = ['hayr', 'bye', 'xayr', 'до свидания', 'goodbye', 'alvido']
            if any(goodbye in question_lower for goodbye in goodbyes):
                return "Xayr! Keyingi safar ko'rishguncha!"
            
            # Help requests
            help_words = ['yordam', 'help', 'помощь', 'көмек', 'yardım', 'qanday']
            if any(word in question_lower for word in help_words) and 'qanday yordam' in question_lower:
                return "Albatta yordam beraman! Men quyidagi vazifalarni bajara olaman:\n• Savollaringizga javob berish\n• Ma'lumot qidirish\n• Hujjatlarni tahlil qilish\n• Maslahat berish\n\nQanday yordam kerak?"
            
            # Check if context is actually relevant (not just any context)
            if context and context.strip():
                # Simple relevance check - if question keywords appear in context
                question_keywords = [word for word in question_lower.split() if len(word) > 3]
                context_lower = context.lower()
                
                # Count how many question keywords appear in context
                relevance_score = sum(1 for keyword in question_keywords if keyword in context_lower)
                
                # If context is relevant (at least some keywords match)
                if relevance_score > 0 or len(question_keywords) == 0:
                    # Extract the most relevant part of context
                    context_sentences = context.split('.')
                    relevant_part = context
                    
                    # Try to find the most relevant sentence
                    if question_keywords and context_sentences:
                        best_sentence = ""
                        best_score = 0
                        for sentence in context_sentences:
                            sentence_score = sum(1 for keyword in question_keywords if keyword in sentence.lower())
                            if sentence_score > best_score:
                                best_score = sentence_score
                                best_sentence = sentence.strip()
                        
                        if best_sentence:
                            relevant_part = best_sentence
                    
                    # Generate response based on question type
                    if any(word in question_lower for word in ['nima', 'what', 'что']):
                        return f"Savolingizga javob:\n\n{relevant_part}\n\nQo'shimcha ma'lumot kerak bo'lsa, aniqroq savol bering."
                    elif any(word in question_lower for word in ['qayerda', 'where', 'где', 'qaysi']):
                        return f"Ma'lumotlarga ko'ra:\n\n{relevant_part}\n\nBu ma'lumot sizga yordam berdimi?"
                    elif any(word in question_lower for word in ['qachon', 'when', 'когда']):
                        return f"Vaqt bo'yicha ma'lumot:\n\n{relevant_part}\n\nYana tafsilotlar kerakmi?"
                    elif any(word in question_lower for word in ['nega', 'why', 'почему', 'nima uchun']):
                        return f"Sababi quyidagicha:\n\n{relevant_part}\n\nTushunarlimi yoki qo'shimcha tushuntirish kerakmi?"
                    elif any(word in question_lower for word in ['qanday', 'how', 'как']):
                        return f"Jarayon quyidagicha:\n\n{relevant_part}\n\nBatafsil ma'lumot kerakmi?"
                    else:
                        return f"Sizning savolingizga javob:\n\n{relevant_part}\n\nQo'shimcha savollaringiz bo'lsa, so'rang."
                else:
                    # Context exists but not relevant to the question
                    return self._generate_no_context_response(question)
            else:
                # No context available
                return self._generate_no_context_response(question)
        
        def _generate_no_context_response(self, question: str) -> str:
            """Generate response when no relevant context is available"""
            question_lower = question.lower().strip()
            
            # Weather related
            if any(word in question_lower for word in ['weather', 'ob-havo', 'погода', 'havo']):
                return "Men ob-havo ma'lumotlariga ega emasman. Ob-havo haqida ma'lumot olish uchun tegishli hujjatlarni yuklang yoki ob-havo xizmatlari veb-saytlaridan foydalaning."
            
            # News/current events
            elif any(word in question_lower for word in ['news', 'yangilik', 'новости', 'today', 'bugun']):
                return "Men joriy yangiliklar va voqealar haqida ma'lumotga ega emasman. Yangiliklar uchun tegishli manbalardan foydalaning yoki kerakli hujjatlarni yuklang."
            
            # Technical questions
            elif any(word in question_lower for word in ['kod', 'code', 'program', 'dastur', 'algorithm']):
                return "Texnik savolingizga javob berish uchun tegishli texnik hujjatlar yoki kod namunalarini yuklang. Men faqat yuklangan hujjatlar asosida javob bera olaman."
            
            # General knowledge questions
            elif any(word in question_lower for word in ['capital', 'poytaxt', 'столица', 'president', 'prezident']):
                return "Men umumiy ma'lumotlar bazasiga ega emasman. Faqat siz yuklagan hujjatlardagi ma'lumotlar asosida javob bera olaman. Kerakli ma'lumotlarni o'z ichiga olgan hujjatlarni yuklang."
            
            # Default response
            else:
                return f"'{question}' savolingizga javob berish uchun tegishli hujjatlar yuklanmagan. Iltimos:\n\n1. Mavzu bo'yicha hujjatlarni yuklang\n2. Savolingizni yuklangan hujjatlar mazmuniga mos ravishda bering\n3. Yoki boshqa mavzuda savol bering\n\nMen faqat yuklangan hujjatlar asosida javob bera olaman."
    
    return MockLLM(config)


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


def format_chat_history(messages: List[ChatMessage], config: Dict[str, Any] = None) -> str:
    """Format chat history for the prompt"""
    if not messages:
        return "Suhbat yangi boshlandi."
    
    formatted_history = []
    # Keep only recent messages based on configuration
    context_limit = config.get("chat_history_context_limit", 5) if config else 5
    for msg in messages[-context_limit:]:
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
            
            # IMPROVED: Higher threshold for better relevance filtering
            score_threshold = config.get("vector_search_score_threshold", 0.6)  # Increased from 0.2
            
            docs = vectorstore.search(
                question, 
                k=top_k,  # Number of documents to retrieve
                score_threshold=score_threshold  # Higher threshold for better relevance
            )
            
            # Format context (empty string if no docs found)
            context = format_docs(docs) if docs else ""
            
            # Debug logging for search results
            if docs:
                logger.info(f"Found {len(docs)} relevant documents")
                for i, doc in enumerate(docs):
                    content_preview = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    score = doc.metadata.get('score', 'N/A')
                    source = doc.metadata.get('source', 'Unknown')
                    logger.debug(f"Doc {i+1}: Score={score}, Source={source}, Content='{content_preview}'")
            else:
                logger.warning(f"No relevant documents found for query: {question[:50]}...")
            
            # Format chat history
            history_text = format_chat_history(chat_history or [], config)
            
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
            
            # Check if RAG functionality is available
            if self.qa_chain is None:
                # Limited functionality mode - simple responses
                response = self._generate_fallback_response(message)
            else:
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
    
    def _generate_fallback_response(self, message: str) -> str:
        """Generate a fallback response when RAG functionality is not available"""
        message_lower = message.lower().strip()
        
        # Simple greeting responses
        greetings = ['salom', 'hello', 'assalomu alaykum', 'hi', 'привет', 'салом']
        if any(greeting in message_lower for greeting in greetings):
            return "Assalomu alaykum! Hozir sistema to'liq ishlamayapti, lekin sizga yordam berishga harakat qilaman."
        
        # Thank you responses
        thanks = ['rahmat', 'thank', 'спасибо', 'рахмат']
        if any(thank in message_lower for thank in thanks):
            return "Arzimaydi! Sistema to'liq tiklanganda yanada yaxshi xizmat ko'rsata olaman."
        
        # Goodbye responses
        goodbyes = ['hayr', 'bye', 'xayr', 'до свидания']
        if any(goodbye in message_lower for goodbye in goodbyes):
            return "Xayr! Sistema to'liq ishlaganda qaytib keling."
        
        # Default response for any other message
        return "Kechirasiz, hozir sistema to'liq ishlamayapti. Iltimos, keyinroq qayta urinib ko'ring yoki administratorga murojaat qiling."
    
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
    
    def get_user_session_status(self, user_id: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Get user session status"""
        try:
            session = self.db_manager.get_active_user_session(user_id)
            
            if session:
                # Check if session is expired based on configuration
                last_activity = datetime.fromisoformat(session["last_activity"])
                # Make both datetimes timezone-aware for comparison
                if last_activity.tzinfo is None:
                    last_activity = last_activity.replace(tzinfo=timezone.utc)
                now = datetime.now(timezone.utc)
                time_diff = now - last_activity
                session_timeout_seconds = (config.get("session_timeout_hours", 1) if config else 1) * 3600
                session_expired = time_diff.total_seconds() > session_timeout_seconds
                
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