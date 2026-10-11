import os
import hashlib
from gtts import gTTS
from config import Config

def speak_text(text, lang="vi"):
    """
    Chuyển văn bản thành giọng nói bằng gTTS và lưu dưới dạng mp3.
    """
    if not text:
        return None
        
    # Tạo thư mục TTS nếu chưa có
    if not os.path.exists(Config.TTS_DIR):
        os.makedirs(Config.TTS_DIR)
        
    # Tạo hash của text để làm tên file và lưu cache
    text_hash = hashlib.md5(text.encode('utf-8')).hexdigest()
    filename = f"{text_hash}_{lang}.mp3"
    filepath = os.path.join(Config.TTS_DIR, filename)
    
    # Kiểm tra xem file đã tồn tại trong cache chưa
    if os.path.exists(filepath):
        return filepath
        
    try:
        # Gọi gTTS
        tts = gTTS(text=text, lang=lang, slow=False)
        tts.save(filepath)
        return filepath
    except Exception as e:
        print(f"Lỗi khi tạo TTS cho '{text}': {e}")
        return None
