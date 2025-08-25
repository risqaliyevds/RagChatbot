"""
Pydantic Models for the RAG Chatbot Application
==============================================

All data models used throughout the application for request/response validation
and data serialization.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    """Чат хабари модели"""
    role: str = Field(..., description="Хабар юборувчининг роли")
    content: str = Field(..., description="Хабар мазмуни")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Хабар вақти")


class ChatSession(BaseModel):
    """Чат сессияси модели"""
    chat_id: str = Field(..., description="Чат идентификатори")
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    messages: List[ChatMessage] = Field(default_factory=list, description="Чат хабарлари")
    created_at: datetime = Field(default_factory=datetime.now, description="Яратилган вақт")
    last_activity: datetime = Field(default_factory=datetime.now, description="Сўнгги фаолият")


class ChatRequest(BaseModel):
    """Чат сўрови модели"""
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    chat_id: Optional[str] = Field(None, description="Чат идентификатори (янги чат учун бўш қолдиринг)")
    message: str = Field(..., description="Фойдаланувчи хабари")


class ChatResponse(BaseModel):
    """Чат жавоби модели"""
    chat_id: str = Field(..., description="Чат идентификатори")
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    message: str = Field(..., description="Бот жавоби")
    timestamp: datetime = Field(default_factory=datetime.now, description="Жавоб вақти")


class ChatHistoryRequest(BaseModel):
    """Чат тарихи сўрови модели"""
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    chat_id: str = Field(..., description="Чат идентификатори")


class ChatCompletionRequest(BaseModel):
    """Чат тўлдириш сўрови модели"""
    model: str = Field(default="google/gemma-3-12b-it", description="Ишлатиладиган модел")
    messages: List[ChatMessage] = Field(..., description="Хабарлар рўйхати")
    max_tokens: Optional[int] = Field(default=150, description="Ишлаб чиқариладиган максимал токенлар")
    temperature: Optional[float] = Field(default=0.7, description="Намуна олиш ҳарорати")
    stream: Optional[bool] = Field(default=False, description="Жавобларни оқим тарзида юбориш")


class ChatCompletionResponse(BaseModel):
    """Чат тўлдириш жавоби модели"""
    id: str = Field(..., description="Тўлдириш учун ноёб идентификатор")
    object: str = Field(default="chat.completion", description="Объект тури")
    created: int = Field(..., description="Яратилган Unix вақт белгиси")
    model: str = Field(..., description="Тўлдириш учун ишлатилган модел")
    choices: List[Dict[str, Any]] = Field(..., description="Тўлдириш танловлари рўйхати")
    usage: Dict[str, int] = Field(..., description="Токен ишлатиш маълумоти")


class NewChatRequest(BaseModel):
    """Янги чат яратиш сўрови модели"""
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")


class NewChatResponse(BaseModel):
    """Янги чат яратиш жавоби модели"""
    chat_id: str = Field(..., description="Янги чат идентификатори")
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    message: str = Field(..., description="Хуш келибсиз хабари")
    created_at: str = Field(..., description="Яратилган вақт")
    last_activity: str = Field(..., description="Сўнгги фаолият")


class UserSessionStatusRequest(BaseModel):
    """Фойдаланувчи сессия ҳолати сўрови модели"""
    user_id: str = Field(..., description="Фойдалануvчи идентификатори")


class UserSessionStatusResponse(BaseModel):
    """Фойдаланувчи сессия ҳолати жавоби модели"""
    user_id: str = Field(..., description="Фойдаланувчи идентификатори")
    has_active_session: bool = Field(..., description="Фаол сессия мавжудлиги")
    active_chat_id: Optional[str] = Field(None, description="Фаол чат идентификатори")
    last_activity: Optional[str] = Field(None, description="Сўнгги фаолият вақти")
    session_expired: bool = Field(..., description="Сессия муddaти тугаганми")


class HealthResponse(BaseModel):
    """Соғлиқни текшириш жавоб модели"""
    status: str = Field(..., description="Соғлиқ ҳолати")
    message: str = Field(..., description="Соғлиқ хабари")
    qdrant_status: str = Field(..., description="Qdrant уланиш ҳолати")
    database_status: str = Field(..., description="Database уланиш ҳолати")


class DocumentUploadResponse(BaseModel):
    """Ҳужжат юклаш жавоби модели"""
    success: bool = Field(..., description="Юклаш муваффақиятли бўлдими")
    message: str = Field(..., description="Жавоб хабари")
    filename: str = Field(..., description="Юкланган файл номи")
    file_size: int = Field(..., description="Файл ҳажми (байтларда)")
    chunks_added: int = Field(..., description="Қўшилган чанклар сони")
    processing_time: float = Field(..., description="Ишлов бериш вақти (сонияларда)")


class FileInfo(BaseModel):
    """Файл маълумотлари модели"""
    filename: str = Field(..., description="Файл номи")
    file_size: int = Field(..., description="Файл ҳажми (байтларда)")
    created_at: str = Field(..., description="Яратилган сана")
    modified_at: str = Field(..., description="Ўзгартирилган сана")
    file_extension: str = Field(..., description="Файл кенгайтмаси")


class DocumentListResponse(BaseModel):
    """Ҳужжатлар рўйхати жавоби модели"""
    success: bool = Field(..., description="Амал муваффақиятли бўлдими")
    message: str = Field(..., description="Жавоб хабари")
    files: List[FileInfo] = Field(default_factory=list, description="Файллар рўйхати")
    total_files: int = Field(..., description="Жами файллар сони")
    total_size: int = Field(..., description="Жами файллар ҳажми")


class DocumentDeleteRequest(BaseModel):
    """Ҳужжат ўчириш сўрови модели"""
    filename: str = Field(..., description="Ўчирилиши керак бўлган файл номи")


class DocumentDeleteResponse(BaseModel):
    """Ҳужжат ўчириш жавоби модели"""
    success: bool = Field(..., description="Ўчириш муваффақиятли бўлдими")
    message: str = Field(..., description="Жавоб хабари")
    filename: str = Field(..., description="Ўчирилган файл номи")
    embeddings_deleted: int = Field(default=0, description="Ўчирилган векторлар сони")


class DocumentUploadProgress(BaseModel):
    """Ҳужжат юклаш прогресс модели"""
    stage: str = Field(..., description="Ҳозирги босқич")
    progress: float = Field(..., description="Прогресс фоизи (0-100)")
    message: str = Field(..., description="Ҳозирги ҳолат хабари")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Қўшимча маълумотлар") 