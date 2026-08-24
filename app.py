import streamlit as st
import os
import requests
import google.generativeai as genai
from gtts import gTTS

# استيرادات MoviePy الصحيحة للإصدار 2.2.1
from moviepy import ImageClip, AudioFileClip, concatenate_videoclips

from diffusers import StableDiffusionPipeline
import torch
from PIL import Image
import numpy as np
import tempfile

# ----------------------------------------------
# إعدادات الصفحة
# ----------------------------------------------
st.set_page_config(page_title="مولد فيديوهات الدروس التعليمية", layout="wide")
st.title("🎬 مولد فيديو الدرس القصصي بالذكاء الاصطناعي")

# ----------------------------------------------
# الإعدادات الجانبية (API Keys)
# ----------------------------------------------
with st.sidebar:
    st.header("⚙️ الإعدادات")
    gemini_api_key = st.text_input("مفتاح Google Gemini API", type="password")
    hf_token = st.text_input("مفتاح Hugging Face (اختياري)", type="password",
                             help="مطلوب لنماذج معينة من Hugging Face")
    st.markdown("---")
    st.markdown("**كيف تحصل على المفاتيح؟**")
    st.markdown("- [الحصول على مفتاح Gemini](https://ai.google.dev/gemini-api/docs/api-key)")
    st.markdown("- [الحصول على مفتاح Hugging Face](https://huggingface.co/settings/tokens)")

# ----------------------------------------------
# واجهة المستخدم
# ----------------------------------------------
col1, col2 = st.columns([2, 1])
with col1:
    topic = st.text_input("📝 اكتب موضوع الدرس:", placeholder="مثال: كيف تعمل الذبذبات في الفيزياء؟")
    num_scenes = st.slider("عدد المشاهد", min_value=3, max_value=10, value=5)
with col2:
    st.markdown("### 🎯 نماذج الذكاء الاصطناعي المستخدمة")
    st.markdown("- **Gemini**: لتوليد السيناريو")
    st.markdown("- **gTTS**: لتحويل النص لكلام")
    st.markdown("- **Stable Diffusion**: لتوليد الصور")

# ----------------------------------------------
# زر التشغيل الرئيسي
# ----------------------------------------------
if st.button("🚀 توليد الفيديو الآن"):
    # التحقق من المفاتيح
    if not gemini_api_key:
        st.error("الرجاء إدخال مفتاح Google Gemini API في القائمة الجانبية.")
        st.stop()
    if not topic:
        st.error("الرجاء كتابة موضوع الدرس.")
        st.stop()

    # --------------------------------
    # 1. توليد السيناريو باستخدام Gemini
    # --------------------------------
    with st.spinner("📝 جاري كتابة السيناريو القصصي..."):
        try:
            genai.configure(api_key=gemini_api_key)
            model = genai.GenerativeModel('gemini-1.5-flash')
            prompt = f"""
            اكتب سيناريو لدرس تعليمي قصصي مدته دقيقة واحدة عن موضوع: {topic}.
            قم بتقسيم السيناريو إلى {num_scenes} مشاهد.
            لكل مشهد: اكتب النص السردي (الذي سيتحدث به الراوي) ووصفًا تفصيليًا للصورة التي ستظهر.
            استخدم لغة عربية فصحى سهلة وجذابة للأطفال.
            """
            response = model.generate_content(prompt)
            script = response.text
            st.success("✅ تم كتابة السيناريو بنجاح!")
            with st.expander("📖 عرض السيناريو"):
                st.write(script)
        except Exception as e:
            st.error(f"حدث خطأ أثناء توليد السيناريو: {e}")
            st.stop()

    # --------------------------------
    # 2. تحويل النص إلى كلام (TTS)
    # --------------------------------
    with st.spinner("🔊 جاري تحويل النص إلى صوت..."):
        try:
            # استخراج النص السردي من السيناريو
            lines = script.split('\n')
            narration_text = ""
            for line in lines:
                if "المشهد" in line or "راوي" in line:
                    if ":" in line:
                        narration_text += line.split(":", 1)[1].strip() + " "
            if not narration_text:
                narration_text = script  # احتياطي

            tts = gTTS(text=narration_text, lang='ar')
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp_audio:
                tts.save(tmp_audio.name)
                audio_path = tmp_audio.name
            st.success("✅ تم توليد الصوت بنجاح!")
        except Exception as e:
            st.error(f"حدث خطأ أثناء توليد الصوت: {e}")
            st.stop()

    # --------------------------------
    # 3. توليد الصور لكل مشهد باستخدام Stable Diffusion
    # --------------------------------
    with st.spinner("🖼️ جاري توليد الصور (قد يستغرق بعض الوقت)..."):
        try:
            # تحميل النموذج (خفيف)
            pipe = StableDiffusionPipeline.from_pretrained(
                "CompVis/stable-diffusion-v1-4",
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
            )
            if torch.cuda.is_available():
                pipe = pipe.to("cuda")

            # استخراج أوصاف الصور من السيناريو
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
            st.success("✅ تم توليد الصور بنجاح!")
        except Exception as e:
            st.error(f"حدث خطأ أثناء توليد الصور: {e}. تأكد من توفر ذاكرة كافية.")
            st.stop()

    # --------------------------------
    # 4. تجميع الفيديو النهائي
    # --------------------------------
    with st.spinner("🎬 جاري تجميع الفيديو..."):
        try:
            # تحميل الصوت
            audio_clip = AudioFileClip(audio_path)
            duration_per_clip = audio_clip.duration / len(image_paths)

            # إنشاء مقاطع الصور
            clips = []
            for img_path in image_paths:
                clip = ImageClip(img_path).with_duration(duration_per_clip).with_resized(height=720)
                clips.append(clip)

            # دمج المقاطع
            video_clip = concatenate_videoclips(clips, method="compose")
            final_video = video_clip.with_audio(audio_clip)

            # حفظ الفيديو
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_video:
                final_video.write_videofile(tmp_video.name, fps=24, codec='libx264', audio_codec='aac')
                video_path = tmp_video.name

            st.success("✅ تم توليد الفيديو بنجاح!")

            # عرض الفيديو وتحميله
            st.video(video_path)
            with open(video_path, "rb") as f:
                st.download_button("📥 تحميل الفيديو", f, file_name="درس_تعليمي.mp4", mime="video/mp4")

        except Exception as e:
            st.error(f"حدث خطأ أثناء تجميع الفيديو: {e}")
            st.stop()

    # ----------------------------------------------
    # رسالة نهائية
    # ----------------------------------------------
    st.markdown("---")
    st.info("💡 تم حفظ الفيديو مؤقتًا. يمكنك تحميله الآن أو إعادة التشغيل لتجربة جديدة.")
