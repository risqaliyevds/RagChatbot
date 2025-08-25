#!/usr/bin/env python3
"""
Gradio Test Interface for Platform Assistant RAG Chatbot
========================================================

Simple Gradio interface to test the enhanced RAG chatbot with:
- User management
- Chat history tracking
- Platform functionality assistance

This is the main Gradio interface moved from the root app folder.
"""

import gradio as gr
import requests
import json
import uuid
import os
import logging
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from app.config import get_config

# Initialize logger
logger = logging.getLogger(__name__)

# Configuration
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8081")
DEFAULT_USER_ID = "test_user_001"

# Debug print
print(f"🔧 DEBUG: API_BASE_URL = {API_BASE_URL}")
print(f"🔧 DEBUG: Environment API_BASE_URL = {os.getenv('API_BASE_URL', 'NOT SET')}")

class GradioRAGClient:
    """Gradio RAG клиенти"""
    
    def __init__(self, base_url: str = API_BASE_URL, config: Dict[str, Any] = None):
        self.base_url = base_url
        self.current_user_id = DEFAULT_USER_ID
        self.current_chat_id = None
        self.config = config or {}
        
        # Configuration for timeouts and limits
        self.client_timeout = self.config.get("gradio_client_timeout", 60)
        self.fresh_init_timeout = self.config.get("fresh_init_timeout", 60) 
        self.http_timeout = self.config.get("http_client_timeout", 10)
        self.error_text_limit = self.config.get("error_text_limit", 500)
        
    def check_health(self) -> dict:
        """Тизим соғлиғини текшириш"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=self.http_timeout)
            if response.status_code == 200:
                return {"status": "✅ Соғлом", "data": response.json()}
            else:
                return {"status": f"❌ Хато {response.status_code}", "data": response.text}
        except Exception as e:
            return {"status": f"❌ Уланиш хатоси", "data": str(e)}
    
    def send_message(self, message: str, user_id: str = None, chat_id: str = None) -> dict:
        """Хабар юбориш"""
        try:
            payload = {
                "user_id": user_id or self.current_user_id,
                "message": message
            }
            
            if chat_id:
                payload["chat_id"] = chat_id
            
            print(f"🔧 DEBUG: Sending message to {self.base_url}/v1/chat")
            print(f"🔧 DEBUG: Payload: {payload}")
                
            response = requests.post(
                f"{self.base_url}/v1/chat",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.client_timeout
            )
            
            print(f"🔧 DEBUG: Response status: {response.status_code}")
            
            if response.status_code == 200:
                data = response.json()
                self.current_chat_id = data.get("chat_id")
                print(f"🔧 DEBUG: Success response: {data}")
                return {
                    "success": True,
                    "response": data.get("message", ""),
                    "chat_id": data.get("chat_id", ""),
                    "timestamp": data.get("timestamp", "")
                }
            else:
                error_text = response.text[:self.error_text_limit]  # Use configurable limit
                print(f"🔧 DEBUG: Error response: {error_text}")
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {error_text}"
                }
                
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Серверга уланиш хатоси: {str(e)}"
            print(f"🔧 DEBUG: Connection error: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except requests.exceptions.Timeout as e:
            error_msg = f"Вақт тугашу хатоси: {str(e)}"
            print(f"🔧 DEBUG: Timeout error: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
        except Exception as e:
            error_msg = f"Умумий хато: {str(e)}"
            print(f"🔧 DEBUG: General error: {error_msg}")
            return {
                "success": False,
                "error": error_msg
            }
    
    def get_chat_history(self, chat_id: str, user_id: str = None) -> dict:
        """Чат тарихини олиш"""
        try:
            payload = {
                "user_id": user_id or self.current_user_id,
                "chat_id": chat_id
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/history",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}
    
    def create_new_chat(self, user_id: str = None) -> dict:
        """Янги чат яратиш (New Chat button functionality)"""
        try:
            payload = {
                "user_id": user_id or self.current_user_id
            }
            
            response = requests.post(
                f"{self.base_url}/v1/chat/new",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                data = response.json()
                self.current_chat_id = data.get("chat_id")
                return {
                    "success": True,
                    "chat_id": data.get("chat_id", ""),
                    "message": data.get("message", ""),
                    "created_at": data.get("created_at", ""),
                    "last_activity": data.get("last_activity", "")
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}: {response.text}"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": f"Хато: {str(e)}"
            }

    def get_user_session_status(self, user_id: str = None) -> dict:
        """Фойдаланувчи сессия ҳолатини текшириш"""
        try:
            payload = {
                "user_id": user_id or self.current_user_id
            }
            
            response = requests.post(
                f"{self.base_url}/v1/user/session-status",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def get_user_chats(self, user_id: str = None) -> dict:
        """Фойдаланувчи чатларини олиш"""
        try:
            uid = user_id or self.current_user_id
            response = requests.get(
                f"{self.base_url}/v1/user/{uid}/chats",
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def upload_document(self, file_path: str) -> dict:
        """Ҳужжат юклаш"""
        try:
            # Read the file content first to avoid file handle issues
            with open(file_path, 'rb') as f:
                file_content = f.read()
            
            # Now upload with the file content
            files = {
                'file': (os.path.basename(file_path), file_content, 'application/octet-stream')
            }
            
            response = requests.post(
                f"{self.base_url}/v1/documents/upload-with-progress",
                files=files,
                timeout=self.client_timeout  # Use longer timeout for file upload
            )
            
            if response.status_code == 200:
                initial_data = response.json()
                upload_id = initial_data.get("upload_id")
                
                if not upload_id:
                    return {"success": False, "error": "No upload ID received"}
                
                # Poll the progress endpoint until completion
                import time
                max_wait_time = 300  # 5 minutes maximum
                poll_interval = 1  # Check every second
                elapsed_time = 0
                
                while elapsed_time < max_wait_time:
                    try:
                        progress_response = requests.get(
                            f"{self.base_url}/v1/documents/upload-progress/{upload_id}",
                            timeout=self.http_timeout
                        )
                        
                        if progress_response.status_code == 200:
                            progress_data = progress_response.json()
                            stage = progress_data.get("stage", "")
                            
                            if stage == "completed":
                                # Extract completion data from details
                                details = progress_data.get("details", {})
                                return {
                                    "success": True,
                                    "data": {
                                        "upload_id": upload_id,
                                        "filename": details.get("original_filename", ""),
                                        "file_size": len(file_content),  # Use actual file size
                                        "chunks_added": details.get("chunks_added", 0),
                                        "processing_time": details.get("processing_time", 0),
                                        "message": progress_data.get("message", "Successfully uploaded")
                                    }
                                }
                            elif stage == "error":
                                error_msg = progress_data.get("message", "Upload failed")
                                return {"success": False, "error": error_msg}
                            
                            # Still processing, wait and try again
                            time.sleep(poll_interval)
                            elapsed_time += poll_interval
                        else:
                            return {"success": False, "error": f"Progress check failed: HTTP {progress_response.status_code}"}
                    
                    except Exception as e:
                        return {"success": False, "error": f"Progress check error: {str(e)}"}
                
                return {"success": False, "error": "Upload timed out"}
                
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def list_documents(self) -> dict:
        """Ҳужжатлар рўйхатини олиш"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/documents/list",
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def delete_document(self, filename: str) -> dict:
        """Ҳужжатни ўчириш"""
        try:
            payload = {"filename": filename}
            
            response = requests.delete(
                f"{self.base_url}/v1/documents/delete",
                json=payload,
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def get_document_stats(self) -> dict:
        """Ҳужжат статистикасини олиш"""
        try:
            response = requests.get(
                f"{self.base_url}/v1/documents/stats",
                timeout=self.http_timeout
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}

    def fresh_initialize_system(self) -> dict:
        """Тизимни янги ҳолатга келтириш (барча маълумотлар ўчириш)"""
        try:
            response = requests.post(
                f"{self.base_url}/v1/system/fresh-init",
                timeout=self.fresh_init_timeout  # Use longer timeout for initialization
            )
            
            if response.status_code == 200:
                return {"success": True, "data": response.json()}
            else:
                return {"success": False, "error": f"HTTP {response.status_code}: {response.text}"}
                
        except Exception as e:
            return {"success": False, "error": f"Хато: {str(e)}"}



# Initialize client globally for reuse
config = get_config()
client = GradioRAGClient(config=config.__dict__ if hasattr(config, '__dict__') else config)

# Test connection on startup
def test_connection():
    """Test initial connection to the API"""
    print("🔍 Testing connection to API...")
    health = client.check_health()
    if "✅" in health.get("status", ""):
        print("✅ Connection to API successful!")
    else:
        print(f"❌ Connection to API failed: {health}")
    return health

# Test connection immediately
startup_health = test_connection()

def format_chat_history(messages: List[dict]) -> str:
    """Чат тарихини форматлаш"""
    if not messages:
        return "Чат тарихи бўш"
    
    formatted = []
    for msg in messages:
        role = "👤 Фойдаланувчи" if msg.get("role") == "user" else "🤖 Ассистент"
        content = msg.get("content", "")
        timestamp = msg.get("timestamp", "")
        if timestamp:
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                time_str = dt.strftime("%H:%M:%S")
            except:
                time_str = timestamp[:8] if len(timestamp) > 8 else timestamp
        else:
            time_str = ""
        
        formatted.append(f"**{role}** {time_str}\n{content}\n")
    
    return "\n".join(formatted)

def chat_interface(message: str, history: List[dict], user_id: str, chat_id: str) -> Tuple[List[dict], str, str]:
    """Чат интерфейси"""
    if not message.strip():
        return history, "", chat_id
    
    # Update client user ID
    client.current_user_id = user_id or DEFAULT_USER_ID
    
    # Send message
    result = client.send_message(message, user_id, chat_id)
    
    if result.get("success"):
        response = result.get("response", "Жавоб олинмади")
        new_chat_id = result.get("chat_id", chat_id)
        
        # Add to history in messages format
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": response})
        
        return history, "", new_chat_id
    else:
        error_msg = f"❌ Хато: {result.get('error', 'Номаълум хато')}"
        history.append({"role": "user", "content": message})
        history.append({"role": "assistant", "content": error_msg})
        return history, "", chat_id

def check_system_health() -> str:
    """Тизим соғлиғини текшириш"""
    health = client.check_health()
    status = health.get("status", "Номаълум")
    data = health.get("data", {})
    
    if isinstance(data, dict):
        details = f"""
**Ҳолат:** {status}
**Хабар:** {data.get('message', 'Маълумот йўқ')}
**Qdrant:** {data.get('qdrant_status', 'Маълумот йўқ')}
"""
    else:
        details = f"**Ҳолат:** {status}\n**Маълумот:** {data}"
    
    return details

def load_chat_history(chat_id: str, user_id: str) -> str:
    """Чат тарихини юклаш"""
    if not chat_id.strip():
        return "Чат ID киритинг"
    
    result = client.get_chat_history(chat_id, user_id)
    
    if result.get("success"):
        data = result.get("data", {})
        messages = data.get("messages", [])
        total = data.get("total_messages", 0)
        
        if messages:
            formatted = format_chat_history(messages)
            return f"**Жами хабарлар:** {total}\n\n{formatted}"
        else:
            return "Бу чатда хабарлар йўқ"
    else:
        return f"❌ Хато: {result.get('error', 'Номаълум хато')}"

def load_user_chats(user_id: str) -> str:
    """Фойдаланувчи чатларини юклаш"""
    result = client.get_user_chats(user_id)
    
    if result.get("success"):
        data = result.get("data", {})
        chats = data.get("chats", [])
        total = data.get("total_chats", 0)
        
        if chats:
            formatted = []
            for chat in chats:
                chat_id = chat.get("chat_id", "")
                created = chat.get("created_at", "")
                last_activity = chat.get("last_activity", "")
                msg_count = chat.get("message_count", 0)
                
                try:
                    created_dt = datetime.fromisoformat(created.replace('Z', '+00:00'))
                    created_str = created_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    created_str = created
                
                try:
                    activity_dt = datetime.fromisoformat(last_activity.replace('Z', '+00:00'))
                    activity_str = activity_dt.strftime("%Y-%m-%d %H:%M")
                except:
                    activity_str = last_activity
                
                formatted.append(f"""
**Чат ID:** `{chat_id}`
**Яратилган:** {created_str}
**Сўнгги фаолият:** {activity_str}
**Хабарлар сони:** {msg_count}
---""")
            
            return f"**Жами чатлар:** {total}\n\n" + "\n".join(formatted)
        else:
            return "Чатлар топилмади"
    else:
        return f"❌ Хато: {result.get('error', 'Номаълум хато')}"

def new_chat(user_id: str) -> Tuple[List[dict], str, str]:
    """Янги чат бошлаш (improved with API call)"""
    # Update client user ID
    client.current_user_id = user_id or DEFAULT_USER_ID
    
    # Create new chat via API
    result = client.create_new_chat(user_id)
    
    if result.get("success"):
        new_chat_id = result.get("chat_id", "")
        welcome_message = result.get("message", "Янги чат яратилди!")
        
        # Initialize with welcome message
        initial_history = [
            {"role": "assistant", "content": welcome_message}
        ]
        
        return initial_history, new_chat_id, f"✅ Янги чат яратилди: {new_chat_id}"
    else:
        error_msg = f"❌ Янги чат яратишда хато: {result.get('error', 'Номаълум хато')}"
        return [], "", error_msg

def check_user_session_status(user_id: str) -> str:
    """Фойдаланувчи сессия ҳолатини текшириш"""
    result = client.get_user_session_status(user_id)
    
    if result.get("success"):
        data = result.get("data", {})
        has_active = data.get("has_active_session", False)
        active_chat_id = data.get("active_chat_id", "")
        last_activity = data.get("last_activity", "")
        session_expired = data.get("session_expired", False)
        
        if has_active:
            return f"""
**✅ Фаол сессия мавжуд**
**Чат ID:** `{active_chat_id}`
**Сўнгги фаолият:** {last_activity}
"""
        elif session_expired:
            return f"""
**⏰ Сессия муддати тугаган**
**Сўнгги фаолият:** {last_activity}
**Ҳолат:** Янги чат яратиш керак (1 соатдан кўп вақт ўтган)
"""
        else:
            return "**🆕 Янги фойдаланувчи** - ҳали чат яратилмаган"
    else:
        return f"❌ Хато: {result.get('error', 'Номаълум хато')}"


def upload_document_handler(file, progress=gr.Progress()) -> str:
    """Ҳужжат юклаш ишловчиси прогресс билан"""
    if file is None:
        return "❌ Файл танланмаган"
    
    try:
        # Initialize progress tracking
        progress(0, desc="Файлни тайёрлаш...")
        
        # Get file size for better progress tracking
        import os
        file_size = os.path.getsize(file.name) if os.path.exists(file.name) else 0
        
        # Convert file size to readable format
        if file_size < 1024:
            size_str = f"{file_size} байт"
        elif file_size < 1024 * 1024:
            size_str = f"{file_size / 1024:.1f} КБ"
        else:
            size_str = f"{file_size / (1024 * 1024):.1f} МБ"
        
        # Stage 1: File validation
        progress(0.05, desc="Файлни текшириш...")
        import time
        time.sleep(0.2)  # Small delay for visual feedback
        
        # Stage 2: File reading
        progress(0.15, desc=f"Файлни ўқиш ({size_str})...")
        time.sleep(0.3)
        
        # Stage 3: Starting upload
        progress(0.25, desc="Серверга юклаш...")
        
        # Simulate realistic upload progress based on file size
        upload_stages = [
            (0.3, "Файлни серверга юклаш..."),
            (0.35, "Серверда қабул қилинмоқда..."),
            (0.4, "Файл текширилмоқда..."),
        ]
        
        for stage_progress, stage_desc in upload_stages:
            progress(stage_progress, desc=stage_desc)
            # Adjust timing based on file size
            if file_size > 5 * 1024 * 1024:  # Files larger than 5MB
                time.sleep(0.5)
            else:
                time.sleep(0.2)
        
        # Upload with progress simulation
        result = client.upload_document(file.name)
        
        # Stage 4: Processing on server (varies by file type)
        file_ext = os.path.splitext(file.name)[1].lower()
        if file_ext == '.pdf':
            progress(0.5, desc="PDF файлини тахлил қилиш...")
            time.sleep(0.8)
        elif file_ext in ['.docx', '.doc']:
            progress(0.5, desc="Word ҳужжатини ишлаш...")
            time.sleep(0.6)
        else:
            progress(0.5, desc="Матн файлини ишлаш...")
            time.sleep(0.4)
        
        # Stage 5: Document processing stages
        processing_stages = [
            (0.6, "Матнни чиқариш..."),
            (0.7, "Ҳужжатни бўлаклаш..."),
            (0.8, "Метаданларни тайёрлаш..."),
        ]
        
        for stage_progress, stage_desc in processing_stages:
            progress(stage_progress, desc=stage_desc)
            time.sleep(0.3)
        
        # Stage 6: Creating embeddings (most time-consuming)
        embedding_stages = [
            (0.85, "Векторлар яратиш..."),
            (0.9, "Векторларни сақлаш..."),
            (0.95, "Индекслаш..."),
        ]
        
        for stage_progress, stage_desc in embedding_stages:
            progress(stage_progress, desc=stage_desc)
            # Longer delay for embedding creation (realistic timing)
            if file_size > 1024 * 1024:  # Files larger than 1MB
                time.sleep(0.5)
            else:
                time.sleep(0.3)
        
        if result.get("success"):
            progress(1.0, desc="✅ Муваффақиятли тугатилди!")
            
            data = result.get("data", {})
            filename = data.get("filename", "")
            file_size = data.get("file_size", 0)
            chunks_added = data.get("chunks_added", 0)
            processing_time = data.get("processing_time", 0)
            message = data.get("message", "")
            
            # Debug logging
            logger.info(f"Upload completion data: filename={filename}, size={file_size}, chunks={chunks_added}")
            
            # Convert file size to readable format
            if file_size < 1024:
                size_str = f"{file_size} байт"
            elif file_size < 1024 * 1024:
                size_str = f"{file_size / 1024:.1f} КБ"
            else:
                size_str = f"{file_size / (1024 * 1024):.1f} МБ"
            
            return f"""
**✅ Ҳужжат муваффақиятли юкланди!**

**Файл номи:** {filename}
**Файл ҳажми:** {size_str}
**Қўшилган чанклар:** {chunks_added}
**Ишлов бериш вақти:** {processing_time} сония

**Хабар:** {message}

🎉 **Энди сиз юкланган ҳужжат бўйича саволлар бера оласиз!**
"""
        else:
            progress(1.0, desc="❌ Хато юз берди")
            error_msg = result.get("error", "Номаълум хато")
            return f"❌ Ҳужжат юклашда хато: {error_msg}"
            
    except Exception as e:
        progress(1.0, desc="❌ Хато юз берди")
        return f"❌ Хато: {str(e)}"


def list_documents_handler() -> str:
    """Ҳужжатлар рўйхатини кўрсатиш ишловчиси"""
    try:
        result = client.list_documents()
        
        if result.get("success"):
            data = result.get("data", {})
            files = data.get("files", [])
            total_files = data.get("total_files", 0)
            total_size = data.get("total_size", 0)
            
            if not files:
                return "📂 Ҳужжатлар папкаси бўш"
            
            # Convert total size to readable format
            if total_size < 1024:
                total_size_str = f"{total_size} байт"
            elif total_size < 1024 * 1024:
                total_size_str = f"{total_size / 1024:.1f} КБ"
            else:
                total_size_str = f"{total_size / (1024 * 1024):.1f} МБ"
            
            output = f"""
## 📂 Ҳужжатлар рўйхати

**Жами файллар:** {total_files}
**Жами ҳажм:** {total_size_str}

---

"""
            
            for i, file_info in enumerate(files, 1):
                filename = file_info.get("filename", "")
                file_size = file_info.get("file_size", 0)
                created_at = file_info.get("created_at", "")
                modified_at = file_info.get("modified_at", "")
                file_extension = file_info.get("file_extension", "")
                
                # Debug logging for individual files
                logger.info(f"Processing file info for display: {filename}, size={file_size}")
                
                # Convert file size to readable format
                if file_size < 1024:
                    size_str = f"{file_size} байт"
                elif file_size < 1024 * 1024:
                    size_str = f"{file_size / 1024:.1f} КБ"
                else:
                    size_str = f"{file_size / (1024 * 1024):.1f} МБ"
                
                # Format dates
                try:
                    created_date = datetime.fromisoformat(created_at.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                    modified_date = datetime.fromisoformat(modified_at.replace('Z', '+00:00')).strftime("%Y-%m-%d %H:%M")
                except:
                    created_date = created_at[:16] if len(created_at) >= 16 else created_at
                    modified_date = modified_at[:16] if len(modified_at) >= 16 else modified_at
                
                # Get file type emoji
                emoji = "📄"
                if file_extension == ".pdf":
                    emoji = "📕"
                elif file_extension in [".docx", ".doc"]:
                    emoji = "📘"
                elif file_extension == ".txt":
                    emoji = "📝"
                elif file_extension == ".md":
                    emoji = "📋"
                elif file_extension == ".py":
                    emoji = "🐍"
                
                output += f"""
### {emoji} {i}. {filename}

**Ҳажм:** {size_str} | **Тур:** {file_extension.upper()} | **Яратилди:** {created_date} | **Ўзгартирилди:** {modified_date}

---
"""
            
            return output
            
        else:
            error_msg = result.get("error", "Номаълум хато")
            return f"❌ Ҳужжатлар рўйхатини олишда хато: {error_msg}"
            
    except Exception as e:
        return f"❌ Хато: {str(e)}"


def delete_document_handler(filename: str) -> str:
    """Ҳужжатни ўчириш ишловчиси"""
    if not filename or filename.strip() == "":
        return "❌ Файл номи киритилмаган"
    
    try:
        result = client.delete_document(filename.strip())
        
        if result.get("success"):
            data = result.get("data", {})
            message = data.get("message", "")
            deleted_filename = data.get("filename", filename)
            embeddings_deleted = data.get("embeddings_deleted", 0)
            
            output = f"""
**✅ Ҳужжат муваффақиятли ўчирилди!**

**Файл номи:** {deleted_filename}

**Ўчирилган векторлар:** {embeddings_deleted}

**Хабар:** {message}
"""
            
            if embeddings_deleted > 0:
                output += "\n\n**🧹 Векторлар базаси тозаланди:** Файлга тааллуқли барча маълумотлар тизимдан олинди."
            else:
                output += "\n\n**⚠️ Векторлар топилмади:** Ушбу файл учун векторлар топилмади (илгари ишланмаган файл бўлиши мумкин)."
            
            return output
            
        else:
            error_msg = result.get("error", "Номаълум хато")
            return f"❌ Ҳужжатни ўчиришда хато: {error_msg}"
            
    except Exception as e:
        return f"❌ Хато: {str(e)}"


def get_document_stats_handler() -> str:
    """Ҳужжат статистикасини кўрсатиш ишловчиси"""
    try:
        result = client.get_document_stats()
        
        if result.get("success"):
            data = result.get("data", {})
            doc_stats = data.get("document_stats", {})
            vector_stats = data.get("vector_stats", {})
            
            total_docs = doc_stats.get("total_documents", 0)
            active_docs = doc_stats.get("active_documents", 0)
            inactive_docs = doc_stats.get("inactive_documents", 0)
            total_size = doc_stats.get("total_size", 0)
            total_chunks = doc_stats.get("total_chunks", 0)
            
            total_points = vector_stats.get("total_points", 0)
            total_files = vector_stats.get("total_files", 0)
            
            # Convert total size to readable format
            if total_size < 1024:
                size_str = f"{total_size} байт"
            elif total_size < 1024 * 1024:
                size_str = f"{total_size / 1024:.1f} КБ"
            else:
                size_str = f"{total_size / (1024 * 1024):.1f} МБ"
            
            return f"""
## 📊 Ҳужжат статистикаси

### 📄 Ҳужжатлар
- **Жами ҳужжатлар:** {total_docs}
- **Фаол ҳужжатлар:** {active_docs}
- **Ўчирилган ҳужжатлар:** {inactive_docs}
- **Жами ҳажм:** {size_str}
- **Жами чанклар:** {total_chunks}

### 🔗 Векторлар
- **Жами векторлар:** {total_points}
- **Файллар билан алоқа:** {total_files}

### 📈 Ўртача кўрсаткичлар
- **Файл бошига чанклар:** {total_chunks / active_docs if active_docs > 0 else 0:.1f}
- **Файл бошига векторлар:** {total_points / total_files if total_files > 0 else 0:.1f}

---

*Эътибор: Агар чанклар ёки векторлар 0 га тенг бўлса, бу ҳужжатлар тўлиқ ишлаб бўлинмаганини кўрсатиши мумкин.*
"""
        else:
            error_msg = result.get("error", "Номаълум хато")
            return f"❌ Статистикани олишда хато: {error_msg}"
            
    except Exception as e:
        return f"❌ Хато: {str(e)}"


def fresh_initialize_handler() -> str:
    """Тизимни янги ҳолатга келтириш ишловчиси"""
    try:
        result = client.fresh_initialize_system()
        
        if result.get("success"):
            data = result.get("data", {})
            details = data.get("details", {})
            warning = data.get("warning", "")
            
            postgresql_cleanup = details.get("postgresql_cleanup", False)
            qdrant_cleanup = details.get("qdrant_cleanup", False)
            database_migration = details.get("database_migration", False)
            qdrant_collection = details.get("qdrant_collection", False)
            
            cleanup_status = "✅" if postgresql_cleanup else "❌"
            qdrant_status = "✅" if qdrant_cleanup else "❌"
            migration_status = "✅" if database_migration else "❌"
            collection_status = "✅" if qdrant_collection else "❌"
            
            return f"""
## 🧨 Тизим янги ҳолатга келтирилди!

### ⚠️ Муҳим огоҳлантириш:
{warning}

### 🔄 Бажарилган амаллар:
- **PostgreSQL маълумотлари тозаланди:** {cleanup_status}
- **Qdrant коллекцияси тозаланди:** {qdrant_status} 
- **Маълумотлар базаси яратилди:** {migration_status}
- **Векторлар коллекцияси яратилди:** {collection_status}

### ✅ Натижа:
Тизим тўлиқ тозаланди ва янги ҳолатга келтирилди. Энди янги ҳужжатлар юклашингиз мумкин.

### 📝 Кейинги қадамлар:
1. Керакли ҳужжатларни қайта юкланг
2. Чат ва фойдаланувчилар маълумотлари ҳам тозаланди
3. Статистикани текшириб кўринг

---

*Эътибор: Барча олдинги маълумотлар ҳамишагалик ўчирилди!*
"""
        else:
            return f"❌ Янги ҳолатга келтиришда хато: {result.get('error', 'Номаълум хато')}"
            
    except Exception as e:
        return f"❌ Хато: {str(e)}"




# Create Gradio interface
with gr.Blocks(title="Platform Assistant RAG Chatbot", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🤖 Платформа Ассистенти RAG Чатбот
    
    Бу платформа функционаллигини ўрганиш учун ёрдамчи чатбот. 
    Фойдаланувчи ID ва чат тарихи автоматик бошқарилади.
    """)
    
    with gr.Tab("💬 Чат"):
        with gr.Row():
            with gr.Column(scale=2):
                user_id_input = gr.Textbox(
                    label="Фойдаланувчи ID",
                    value=DEFAULT_USER_ID,
                    placeholder="Фойдаланувчи идентификатори"
                )
                chat_id_input = gr.Textbox(
                    label="Чат ID (ихтиёрий)",
                    placeholder="Мавжуд чатни давом эттириш учун ID киритинг"
                )
                
            with gr.Column(scale=1):
                health_btn = gr.Button("🏥 Тизим соғлиғи", variant="secondary")
                new_chat_btn = gr.Button("🆕 Янги чат", variant="primary")
                session_status_btn = gr.Button("📊 Сессия ҳолати", variant="secondary")
        
        chatbot = gr.Chatbot(
            label="Чат",
            height=400,
            placeholder="Саволингизни ёзинг...",
            type='messages'
        )
        
        with gr.Row():
            msg_input = gr.Textbox(
                label="Хабар",
                placeholder="Саволингизни ёзинг...",
                scale=4
            )
            send_btn = gr.Button("📤 Юбориш", variant="primary", scale=1)
        
        with gr.Row():
            health_output = gr.Markdown(label="Тизим ҳолати")
            session_output = gr.Markdown(label="Сессия ҳолати")
    
    with gr.Tab("📜 Чат тарихи"):
        with gr.Row():
            history_user_id = gr.Textbox(
                label="Фойдаланувчи ID",
                value=DEFAULT_USER_ID
            )
            history_chat_id = gr.Textbox(
                label="Чат ID",
                placeholder="Тарихини кўриш учун чат ID киритинг"
            )
            load_history_btn = gr.Button("📥 Тарихни юклаш", variant="primary")
        
        history_output = gr.Markdown(label="Чат тарихи")
    
    with gr.Tab("📋 Фойдаланувчи чатлари"):
        with gr.Row():
            chats_user_id = gr.Textbox(
                label="Фойдаланувчи ID",
                value=DEFAULT_USER_ID
            )
            load_chats_btn = gr.Button("📥 Чатларни юклаш", variant="primary")
        
        chats_output = gr.Markdown(label="Фойдаланувчи чатлари")
    
    with gr.Tab("📄 Файл бошқаруви"):
        gr.Markdown("""
        ### 📄 Файл бошқаруви ва векторлаштириш
        
        Бу бўлимда сиз ҳужжатларни юклаш, рўйхатини кўриш ва ўчириш амалларини бажара оласиз.
        """)
        
        with gr.Tab("📤 Файл юклаш"):
            gr.Markdown("""
            **Қўллаб-қувватланадиган форматлар:**
            - PDF файллари (.pdf)
            - Word ҳужжатлари (.docx, .doc)
            - Матн файллари (.txt)
            - Markdown файллари (.md)
            - Python файллари (.py)
            
            **Максимал файл ҳажми:** 50 МБ
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    file_upload = gr.File(
                        label="Ҳужжат танланг",
                        file_types=[".pdf", ".docx", ".doc", ".txt", ".md", ".py"],
                        type="filepath"
                    )
                    upload_btn = gr.Button("📤 Ҳужжатни юклаш", variant="primary", size="lg")
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **Эслатма:**
                    - Ҳужжат юкланганидан сўнг у автоматик равишда ишланади
                    - Векторлаштириш жараёни бир неча дақиқа давом этиши мумкин
                    - Юкланган ҳужжатлар `documents/` папкасида сақланади
                    """)
            
            upload_output = gr.Markdown(label="Юклаш натижаси")
        
        with gr.Tab("📂 Файллар рўйхати"):
            gr.Markdown("""
            ### 📂 Мавжуд ҳужжатлар рўйхати
            
            Бу ерда сиз барча юкланган ҳужжатларни кўра оласиз.
            """)
            
            with gr.Row():
                list_files_btn = gr.Button("🔄 Рўйхатни янгилаш", variant="primary")
                
            files_list_output = gr.Markdown(label="Файллар рўйхати")
        
        with gr.Tab("🗑️ Файл ўчириш"):
            gr.Markdown("""
            ### 🗑️ Ҳужжатни ўчириш
            
            Ҳужжатни тизимдан бутунлай ўчиришучун файл номини киритинг.
            
            **Огоҳлантириш:** Бу амал қайтарилмас!
            """)
            
            with gr.Row():
                with gr.Column(scale=2):
                    filename_input = gr.Textbox(
                        label="Файл номи",
                        placeholder="Масалан: my_document.pdf",
                        info="Ўчириш учун анқо файл номини киритинг"
                    )
                    delete_btn = gr.Button("🗑️ Файлни ўчириш", variant="stop")
                
                with gr.Column(scale=1):
                    gr.Markdown("""
                    **Эслатма:**
                    - Файл номини аниқ киритинг
                    - Файл тизимдан бутунлай ўчирилади
                    - Ушбу амал қайтарилмас
                    - Файл векторлар базасидан ҳам олинади
                    """)
            
            delete_output = gr.Markdown(label="Ўчириш натижаси")
        
        with gr.Tab("📊 Статистика"):
            gr.Markdown("""
            ### 📊 Ҳужжат ва векторлар статистикаси
            
            Бу ерда тизимдаги барча ҳужжатлар ва уларнинг ишлов бериш ҳолати ҳақида батафсил маълумот олишингиз мумкин.
            """)
            
            with gr.Row():
                stats_btn = gr.Button("🔄 Статистикани янгилаш", variant="primary")
                
            stats_output = gr.Markdown(label="Тизим статистикаси")
            
            gr.Markdown("---")
            
            gr.Markdown("""
            ### 🧨 Хавфли ҳудуд: Тизимни янги ҳолатга келтириш
            
            **⚠️ ОГОҲЛАНТИРИШ:** Бу амал БАРЧА маълумотларни ҳамишагалик ўчиради!
            
            Бу дагча барча ҳужжатлар, чатлар, фойдаланувчилар маълумотлари ва векторларни тизимдан тўлиқ ўчиради ва тизимни янги ҳолатга келтиради.
            """)
            
            with gr.Row():
                fresh_init_btn = gr.Button("🧨 ТИЗИМНИ ЯНГИ ҲОЛАТГА КЕЛТИРИШ", variant="stop", size="lg")
                
            fresh_init_output = gr.Markdown(label="Янги ҳолатга келтириш натижаси")

    
    # Event handlers
    def send_message_handler(message, history, user_id, chat_id):
        return chat_interface(message, history, user_id, chat_id)
    
    # Bind events
    send_btn.click(
        send_message_handler,
        inputs=[msg_input, chatbot, user_id_input, chat_id_input],
        outputs=[chatbot, msg_input, chat_id_input]
    )
    
    msg_input.submit(
        send_message_handler,
        inputs=[msg_input, chatbot, user_id_input, chat_id_input],
        outputs=[chatbot, msg_input, chat_id_input]
    )
    
    health_btn.click(
        check_system_health,
        outputs=health_output
    )
    
    new_chat_btn.click(
        new_chat,
        inputs=[user_id_input],
        outputs=[chatbot, chat_id_input, chats_output]
    )
    
    session_status_btn.click(
        check_user_session_status,
        inputs=[user_id_input],
        outputs=session_output
    )
    
    load_history_btn.click(
        load_chat_history,
        inputs=[history_chat_id, history_user_id],
        outputs=history_output
    )
    
    load_chats_btn.click(
        load_user_chats,
        inputs=[chats_user_id],
        outputs=chats_output
    )
    
    upload_btn.click(
        upload_document_handler,
        inputs=[file_upload],
        outputs=upload_output
    )
    
    list_files_btn.click(
        list_documents_handler,
        outputs=files_list_output
    )
    
    delete_btn.click(
        delete_document_handler,
        inputs=[filename_input],
        outputs=delete_output
    )
    
    stats_btn.click(
        get_document_stats_handler,
        outputs=stats_output
    )
    
    fresh_init_btn.click(
        fresh_initialize_handler,
        outputs=fresh_init_output
    )
    


if __name__ == "__main__":
    print("🚀 Gradio интерфейсини ишга тушириш...")
    print(f"📡 API сервер: {API_BASE_URL}")
    print("🌐 Веб интерфейс: http://localhost:7860")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        show_error=True,
        debug=False
    ) 