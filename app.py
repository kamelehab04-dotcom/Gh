import streamlit as st
import os
import requests
import google.generativeai as genai
from gtts import gTTS
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import tempfile

st.set_page_config(page_title="مولد فيديوهات الدروس التعليمية", layout="wide")
st.title("🎬 مولد فيديو الدرس القصصي بالذكاء الاصطناعي")

with st.sidebar:
    st.header("⚙️ الإعدادات")
    gemini_api_key = st.text_input("مفتاح Google Gemini API", type="password")
    st.markdown("---")
    st.markdown("**كيف تحصل على المفتاح؟**")
    st.markdown("- [الحصول على مفتاح Gemini](https://ai.google.dev/gemini-api/docs/api-key)")

col1, col2 = st.columns([2, 1])
with col1:
    topic = st.text_input("📝 اكتب موضوع الدرس:", placeholder="مثال: كيف تعمل الذبذبات في الفيزياء؟")
    num_scenes = st.slider("عدد المشاهد", min_value=3, max_value=10, value=5)
with col2:
    st.markdown("### 🎯 النماذج المستخدمة")
    st.markdown("- **Gemini Pro**: لتوليد السيناريو")
    st.markdown("- **gTTS**: للصوت")
    st.markdown("- **Stable Diffusion**: للصور")

if st.button("🚀 توليد الفيديو الآن"):
    if not gemini_api_key:
        st.error("الرجاء إدخال مفتاح Google Gemini API.")
        st.stop()
    if not topic:
        st.error("الرجاء كتابة موضوع الدرس.")
        st.stop()

    # 1. توليد السيناريو
    with st.spinner("📝 جاري كتابة السيناريو..."):
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-pro')   # ✅ النموذج المتاح عالميًا
            prompt = f"""
            اكتب سيناريو لدرس تعليمي قصصي مدته دقيقة واحدة عن موضوع: {topic}.
            قم بتقسيم السيناريو إلى {num_scenes} مشاهد.
            لكل مشهد: اكتب النص السردي ووصفًا تفصيليًا للصورة.
            استخدم لغة عربية فصحى سهلة وجذابة للأطفال.
            """
            response = model.generate_content(prompt)
            script = response.text
            st.success("✅ تم كتابة السيناريو!")
            with st.expander("📖 عرض السيناريو"):
                st.write(script)
        except Exception as e:
            st.error(f"خطأ في توليد السيناريو: {e}")
            st.stop()

    # 2. تحويل النص إلى صوت
    with st.spinner("🔊 جاري تحويل النص إلى صوت..."):
        try:
            lines = script.split('\n')
            narration_text = ""
            for line in lines:
                if "المشهد" in line or "راوي" in line:
                    if ":" in line:
                        narration_text += line.split(":", 1)[1].strip() + " "
            if not narration_text:
                narration_text = script
            tts = gTTS(text=narration_text, lang='ar')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tts.save(tmp_audio.name)
                audio_path = tmp_audio.name
            st.success("✅ تم توليد الصوت!")
        except Exception as e:
            st.error(f"خطأ في توليد الصوت: {e}")
            st.stop()

    # 3. توليد الصور
    with st.spinner("🖼️ جاري توليد الصور (قد يستغرق وقتًا)..."):
        try:
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            image_prompts = []
            for line in lines:
                if "صورة:" in line or "وصف الصورة:" in line:
                    image_prompts.append(line.split(":", 1)[1].strip())
            if not image_prompts:
                image_prompts = [f"رسم توضيحي لدرس عن {topic}, أسلوب كرتوني"] * num_scenes

            image_paths = []
            for i, prompt in enumerate(image_prompts[:num_scenes]):
                with st.status(f"توليد الصورة {i+1}/{num_scenes}..."):
                    image = pipe(prompt).images[0]
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                        image.save(tmp_img.name)
                        image_paths.append(tmp_img.name)
            st.success("✅ تم توليد الصور!")
        except Exception as e:
            st.error(f"خطأ في توليد الصور: {e}")
            st.stop()

    # 4. تجميع الفيديو
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
            st.success("✅ تم توليد الفيديو!")
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button("📥 تحميل الفيديو", f, file_name="درس_تعليمي.mp4", mime="video/mp4")
        except Exception as e:
            st.error(f"خطأ في تجميع الفيديو: {e}")
            st.stop()

    st.info("💡 تم حفظ الفيديو مؤقتًا. يمكنك تحميله الآن.")
