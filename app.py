import streamlit as st
import os
import tempfile
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
from groq import Groq
import requests

# ----------------------------------------------
# 🔑 المفاتيح (مُدخلة مباشرة للتجربة فقط)
# ----------------------------------------------
GROQ_API_KEY = "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"
HF_TOKEN = "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"
ELEVENLABS_API_KEY = "sk_f0e90f28021076730796bc5a4fe56887b6f4b6ed5998b686"

# ----------------------------------------------
# إعدادات الصفحة
# ----------------------------------------------
st.set_page_config(page_title="مولد فيديوهات تعليمية - تجريبي", layout="wide")
st.title("🎬 مولد فيديو الدرس القصصي بالذكاء الاصطناعي (تجربة)")

st.warning("⚠️ هذه مفاتيح تجريبية فقط، استبدلها بمفاتيحك الخاصة فوراً!")

# ----------------------------------------------
# واجهة المستخدم
# ----------------------------------------------
topic = st.text_input("📝 اكتب موضوع الدرس:", placeholder="مثال: كيف تعمل الذبذبات في الفيزياء؟")
num_scenes = st.slider("عدد المشاهد", min_value=3, max_value=10, value=5)

if st.button("🚀 توليد الفيديو الآن"):
    if not topic:
        st.error("الرجاء كتابة موضوع الدرس.")
        st.stop()

    # ----------------------------------------------
    # 1. توليد السيناريو باستخدام Groq
    # ----------------------------------------------
    with st.spinner("📝 جاري كتابة السيناريو..."):
        try:
            client = Groq(api_key=GROQ_API_KEY)
            prompt = f"""
            اكتب سيناريو لدرس تعليمي قصصي مدته دقيقة واحدة عن موضوع: {topic}.
            قم بتقسيم السيناريو إلى {num_scenes} مشاهد.
            لكل مشهد: اكتب النص السردي (الذي سيتحدث به الراوي) ووصفًا تفصيليًا للصورة التي ستظهر.
            استخدم لغة عربية فصحى سهلة وجذابة للأطفال.
            """
            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            script = response.choices[0].message.content
            st.success("✅ تم كتابة السيناريو!")
            with st.expander("📖 عرض السيناريو"):
                st.write(script)
        except Exception as e:
            st.error(f"خطأ في توليد السيناريو: {e}")
            st.stop()

    # ----------------------------------------------
    # 2. تحويل النص إلى كلام باستخدام ElevenLabs (مع الاحتياطي gTTS)
    # ----------------------------------------------
    with st.spinner("🔊 جاري تحويل النص إلى صوت..."):
        try:
            # استخراج النص السردي
            lines = script.split('\n')
            narration_text = ""
            for line in lines:
                if "المشهد" in line or "راوي" in line:
                    if ":" in line:
                        narration_text += line.split(":", 1)[1].strip() + " "
            if not narration_text:
                narration_text = script

            # محاولة استخدام ElevenLabs أولاً
            try:
                url = "https://api.elevenlabs.io/v1/text-to-speech/21m00Tcm4TlvDq8ikWAM"
                headers = {
                    "Accept": "audio/mpeg",
                    "Content-Type": "application/json",
                    "xi-api-key": ELEVENLABS_API_KEY
                }
                data = {
                    "text": narration_text,
                    "model_id": "eleven_monolingual_v1",
                    "voice_settings": {"stability": 0.5, "similarity_boost": 0.5}
                }
                response = requests.post(url, json=data, headers=headers)
                if response.status_code == 200:
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                        tmp_audio.write(response.content)
                        audio_path = tmp_audio.name
                else:
                    raise Exception("ElevenLabs فشل، استخدام gTTS")
            except:
                # الاحتياطي: gTTS (مجاني بدون مفتاح)
                tts = gTTS(text=narration_text, lang='ar')
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                    tts.save(tmp_audio.name)
                    audio_path = tmp_audio.name
            
            st.success("✅ تم توليد الصوت!")
        except Exception as e:
            st.error(f"خطأ في توليد الصوت: {e}")
            st.stop()

    # ----------------------------------------------
    # 3. توليد الصور باستخدام Hugging Face Token
    # ----------------------------------------------
    with st.spinner("🖼️ جاري توليد الصور (قد يستغرق وقتاً)..."):
        try:
            # تحميل النموذج باستخدام التوكن
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                use_auth_token=HF_TOKEN
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            # استخراج أوصاف الصور
            image_prompts = []
            for line in lines:
                if "صورة:" in line or "وصف الصورة:" in line:
                    image_prompts.append(line.split(":", 1)[1].strip())
            if not image_prompts:
                image_prompts = [f"رسم توضيحي لدرس عن {topic}, أسلوب كرتوني تعليمي"] * num_scenes

            image_paths = []
            for i, prompt in enumerate(image_prompts[:num_scenes]):
                with st.status(f"توليد الصورة {i+1}/{num_scenes}..."):
                    image = pipe(prompt).images[0]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        image.save(tmp_img.name)
                        image_paths.append(tmp_img.name)
            st.success("✅ تم توليد الصور!")
        except Exception as e:
            st.error(f"خطأ في توليد الصور: {e}. تأكد من توفر الذاكرة.")
            st.stop()

    # ----------------------------------------------
    # 4. تجميع الفيديو النهائي
    # ----------------------------------------------
    with st.spinner("🎬 جاري تجميع الفيديو..."):
        try:
            audio_clip = AudioFileClip(audio_path)
            duration_per_clip = audio_clip.duration / len(image_paths)
            
            clips = []
            for img_path in image_paths:
                clip = ImageClip(img_path).with_duration(duration_per_clip).with_resized(height=720)
                clips.append(clip)
            
            video_clip = concatenate_videoclips(clips, method="compose")
            final_video = video_clip.with_audio(audio_clip)
            
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                final_video.write_videofile(tmp_video.name, fps=24, codec='libx264', audio_codec='aac')
                video_path = tmp_video.name
            
            st.success("✅ تم توليد الفيديو بنجاح!")
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button("📥 تحميل الفيديو", f, file_name="درس_تعليمي.mp4", mime="video/mp4")
                
        except Exception as e:
            st.error(f"خطأ في تجميع الفيديو: {e}")
            st.stop()

    st.info("💡 تم حفظ الفيديو مؤقتاً. حمّله الآن.")
