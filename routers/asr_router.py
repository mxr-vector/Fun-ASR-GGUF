import os
import shutil
import tempfile
import asyncio
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, status
from pydantic import BaseModel
from typing import Optional, List
from services import asr_service
from core.config import settings
from core.logger import logger
from core.response import R

router = APIRouter(
    prefix="/asr",
    tags=["ASR Transcription"]
)

class TranscriptionResponse(BaseModel):
    filename: str
    text: str
    audio_id: Optional[str] = None
    error: Optional[str] = None
    language: Optional[str] = None
    segments: Optional[list] = None
    ctc_text: Optional[str] = None
    hotwords: Optional[list] = None
    timings: Optional[dict] = None

class HotwordsRequest(BaseModel):
    words: List[str]

@router.get("/hotwords", summary="获取当前热词列表")
async def get_hotwords():
    """
    获取 ASR 引擎当前加载的热词列表。
    """
    words = asr_service.get_hotwords()
    return R.success(data={"words": words, "count": len(words)})

@router.post("/hotwords", summary="更新热词列表（支持热更新）")
async def update_hotwords(request: HotwordsRequest):
    """
    更新 ASR 引擎的热词列表，会覆盖当前文件并热加载。
    """
    try:
        count = asr_service.update_hotwords(request.words)
        return R.success(msg="热词更新成功", data={"count": count})
    except Exception as e:
        logger.error(f"热词更新失败: {e}", exc_info=True)
        return R.fail(msg=f"热词更新失败: {str(e)}", code=500)

@router.post("/transcribe", response_model=R[TranscriptionResponse], summary="单个音频文件转写")
async def transcribe_audio(
    file: UploadFile = File(...),
    audio_id: Optional[str] = Form(None, description="音频的自定义ID，原样返回"),
    language: Optional[str] = Form(None, description="语言设置（None=自动检测）"),
    context: Optional[str] = Form(settings.DEFAULT_CONTEXT, description="上下文信息"),
):
    """
    接收单个音频文件并使用 Fun-ASR 进行转写。
    支持自动检测语言和提供额外的提示上下文。
    """
    if not file.filename:
        logger.warning("请求拒绝：未提供上传文件名称。")
        return R.fail(msg="未上传文件", code=400)
        
    ext = os.path.splitext(file.filename)[1].lower()
    
    # 创建临时文件统一保存上传的音频
    fd, temp_path = tempfile.mkstemp(suffix=ext)
    os.close(fd)
    
    logger.info(f"正在接收音频文件: {file.filename}")
    
    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        result_dict = await asr_service.transcribe_async(
            temp_path, 
            language=language, 
            context=context
        )
        
        response_data = TranscriptionResponse(
            audio_id=audio_id,
            filename=file.filename,
            text=result_dict.get("text", ""),
            language=language,
            segments=result_dict.get("segments", []),
            ctc_text=result_dict.get("ctc_text", ""),
            hotwords=result_dict.get("hotwords", []),
            timings=result_dict.get("timings", {})
        )
        
        return R.success(data=response_data)
        
    except Exception as e:
        logger.error(f"转写文件 {file.filename} 时出错: {e}", exc_info=True)
        return R.fail(msg=f"转写失败: {str(e)}", code=500)
    finally:
        # 清理临时文件
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.post("/transcribe/batch", response_model=R[List[TranscriptionResponse]], summary="批量音频文件转写")
async def transcribe_audio_batch(
    files: List[UploadFile] = File(...),
    audio_ids: Optional[List[str]] = Form(None, description="音频的ID列表，需与files顺序对应"),
    language: Optional[str] = Form(None, description="语言设置（None=自动检测）"),
    context: Optional[str] = Form(settings.DEFAULT_CONTEXT, description="上下文信息"),
):
    """
    接收多个音频文件并进行批量转写。由于底层使用单例和锁，
    并发上传的多个文件将被安全排队并行/串行处理，不会导致内存溢出。
    """
    if not files or len(files) == 0:
        return R.fail(msg="未上传文件", code=400)

    logger.info(f"正在接收批量音频上传: 共 {len(files)} 个文件")
    
    results = []
    
    # 内部异步函数：用于处理单个文件
    async def process_single_file(file: UploadFile, audio_id: Optional[str]) -> TranscriptionResponse:
        if not file.filename:
            return TranscriptionResponse(audio_id=audio_id, filename="unknown", text="", error="No filename provided")
            
        ext = os.path.splitext(file.filename)[1].lower()
        fd, temp_path = tempfile.mkstemp(suffix=ext)
        os.close(fd)
        
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # asr_service 内部的 asyncio.Lock() 会安全地处理重叠的并发请求
            result_dict = await asr_service.transcribe_async(
                temp_path, 
                language=language, 
                context=context
            )
            return TranscriptionResponse(
                audio_id=audio_id,
                filename=file.filename, 
                text=result_dict.get("text", ""), 
                language=language,
                segments=result_dict.get("segments", []),
                ctc_text=result_dict.get("ctc_text", ""),
                hotwords=result_dict.get("hotwords", []),
                timings=result_dict.get("timings", {})
            )
            
        except Exception as e:
            logger.error(f"批量转写文件 {file.filename} 时出错: {e}", exc_info=True)
            return TranscriptionResponse(audio_id=audio_id, filename=file.filename, text="", error=str(e))
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    # 使用 gather 并发调度所有转写任务，服务层的锁机制会自动对它们进行安全排队
    coroutines = []
    for idx, f in enumerate(files):
        aid = audio_ids[idx] if audio_ids and idx < len(audio_ids) else None
        coroutines.append(process_single_file(f, aid))

    results = await asyncio.gather(*coroutines)
    
    return R.success(data=results)

@router.get("/health", summary="生产级健康检查")
async def health_check():
    """
    检查 ASR 引擎状态及应用存活状态。
    结合 core 返回结构 R 进行规范化输出。
    """
    is_loaded = asr_service.engine is not None
    data = {
        "status": "healthy" if is_loaded else "degraded",
        "engine_loaded": is_loaded,
        "workers": asr_service._executor._max_workers
    }
    if is_loaded:
        return R.success(data=data, msg="系统运行正常")
    else:
        # 即使引擎未加载，服务依然是UP的
        return R.success(data=data, msg="系统已启动，但 ASR 引擎尚未加载")
