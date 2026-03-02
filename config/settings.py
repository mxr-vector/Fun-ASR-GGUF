import os
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # API configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # Model configuration paths
    MODEL_DIR: str = "./model"
    DATA_DIR: str = "./data"
    HOTWORDS_PATH: str = "./hot-word.txt"
    
    # Transcription settings
    SIMILAR_THRESHOLD: float = 0.6
    MAX_HOTWORDS: int = 10
    ENABLE_CTC: bool = True
    
    # Default context for ASR
    DEFAULT_CONTEXT: str = (
        "这是民航空中交通管制（ATC）地空通话录音，通话语言为中英文混合。"
        "说话人为管制员或飞行员，内容涉及起飞、落地、滑行、航向、高度、速度等指令。"
        "中文数字采用航空专用读法：0=洞，1=幺，2=两，3=三，4=四，5=五，6=六，7=拐，8=八，9=勾。"
        "英文部分使用ICAO标准术语，如CLEARED、MAINTAIN、DESCEND、CLIMB、RUNWAY、SQUAWK，"
        "以及NATO音标字母如ALFA、BRAVO、CHARLIE等。"
        "请严格保留原始口语读法，中文数字不要转为阿拉伯数字，英文术语保留原词。"
        "常见句式如：南方三五六幺请联系进近幺两幺点六五、CCA1508 DESCEND AND MAINTAIN FL090、修正海压幺洞幺三。"
        "通话中可能包含航班呼号（如国航、东方、南方、Cathay、Lufthansa）加数字编号、"
        "跑道编号、频率读报、应答机编码、高度层和航向角度。"
    )

    @property
    def encoder_onnx_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "Fun-ASR-Nano-Encoder-Adaptor.fp16.onnx")

    @property
    def ctc_onnx_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "Fun-ASR-Nano-CTC.fp16.onnx")

    @property
    def decoder_gguf_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "Fun-ASR-Nano-Decoder.fp16.gguf")

    @property
    def tokens_path(self) -> str:
        return os.path.join(self.MODEL_DIR, "tokens.txt")
        
    class Config:
        env_prefix = "ASR_"

settings = Settings()
