import os
import glob
import shutil
import zipfile
import tempfile
import pandas as pd
import streamlit as st
from pdf2image import convert_from_path

from PIL import Image

from pipeline import MangaTranslatorPipeline


# ==========================================================
# CONFIG
# ==========================================================

st.set_page_config(
    page_title="Manga Translator",
    page_icon="📖",
    layout="wide"
)

st.title("📖 Manga Translator")

st.markdown(
    "Deteksi Bubble → OCR → Translate → Hapus Teks → Render Ulang"
)

# ==========================================================
# SIDEBAR
# ==========================================================

st.sidebar.header("Settings")

language = st.sidebar.selectbox(
    "Target Language",
    [
        "Indonesia",
        "English"
    ]
)

conf = st.sidebar.slider(
    "YOLO Confidence",
    min_value=0.10,
    max_value=1.00,
    value=0.30,
    step=0.05
)

font_path = st.sidebar.text_input(
    "Font",
    value="CANDARAI.ttf"
)

# ==========================================================
# TARGET LANGUAGE
# ==========================================================

target = "id"

if language == "English":
    target = "en"

# ==========================================================
# LOAD PIPELINE
# ==========================================================

@st.cache_resource
def load_pipeline(target_language):

    return MangaTranslatorPipeline(
        yolo_weights_path="yolo11s.pt",
        comic_font_path=font_path,
        target_language=target_language
    )

pipeline = load_pipeline(target)

# ==========================================================
# UPLOAD
# ==========================================================

uploaded_file = st.file_uploader(
    "Upload Manga (.jpg, .png, .jpeg, .zip atau .pdf)",
    type=[
        "jpg",
        "jpeg",
        "png",
        "webp",
        "pdf",
        "zip"
    ]
)

# ==========================================================
# SINGLE IMAGE
# ==========================================================

def process_single_image(uploaded_file):

    image = Image.open(uploaded_file)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Translated")
        st.info("Belum diproses")

    if not st.button(
        "🚀 Translate",
        use_container_width=True
    ):
        return

    progress = st.progress(0)

    status = st.empty()

    status.text("Menyimpan gambar...")

    temp = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".jpg"
    )

    uploaded_file.seek(0)

    temp.write(
        uploaded_file.getbuffer()
    )

    temp.close()

    progress.progress(20)

    output_dir = "outputs"

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    output_path = os.path.join(
        output_dir,
        "translated.jpg"
    )

    status.text("Menerjemahkan...")

    result = pipeline.process_manga_page(
        temp.name,
        output_path,
        conf_threshold=conf
    )

    progress.progress(100)

    status.success("Selesai")

    st.success("Translation Finished")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original")
        st.image(
            image,
            use_container_width=True
        )

    with col2:
        st.subheader("Translated")
        st.image(
            output_path,
            use_container_width=True
        )

    st.divider()

    rows = []

    for i in range(len(result["boxes"])):

        rows.append({
            "Bubble": i + 1,
            "Japanese": result["ocr"][i],
            "Translation": result["translation"][i]
        })

    df = pd.DataFrame(rows)

    st.subheader("OCR Result")

    st.dataframe(
        df,
        use_container_width=True
    )

    csv = df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "⬇ Download OCR CSV",
        csv,
        file_name="ocr_result.csv",
        mime="text/csv",
        on_click="ignore"
    )

    with open(output_path, "rb") as f:

        st.download_button(
            "⬇ Download Image",
            data=f,
            file_name="translated.jpg",
            mime="image/jpeg",
            on_click="ignore"
        )

    os.unlink(temp.name)

# ==========================================================
# PDF PROCESSING
# ==========================================================
def process_pdf(uploaded_file):

    temp_dir = tempfile.mkdtemp()

    pdf_path = os.path.join(temp_dir, uploaded_file.name)

    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    pages = convert_from_path(pdf_path)

    image_files = []

    ##########################################################
    # UBAH PDF MENJADI GAMBAR
    ##########################################################

    for i, page in enumerate(pages):

        path = os.path.join(temp_dir, f"page_{i+1}.jpg")

        page = page.convert("L").convert("RGB")

        page.save(path, "JPEG")

        image_files.append(path)

    image_files.sort()

    ##########################################################
    # VALIDASI
    ##########################################################

    if len(image_files) == 0:
        st.error("PDF tidak memiliki halaman.")
        shutil.rmtree(temp_dir)
        return

    MAX_PAGE = 35

    if len(image_files) > MAX_PAGE:
        st.error(f"Maksimal {MAX_PAGE} halaman.")
        shutil.rmtree(temp_dir)
        return

    ##########################################################
    # PREVIEW
    ##########################################################

    st.success(f"Ditemukan {len(image_files)} halaman.")

    preview = Image.open(image_files[0])

    st.image(
        preview,
        caption="Preview halaman pertama",
        use_container_width=True
    )

    ##########################################################
    # BUTTON
    ##########################################################

    if not st.button(
        "🚀 Translate PDF",
        use_container_width=True
    ):
        return

    output_folder = os.path.join(
        temp_dir,
        "translated"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    progress = st.progress(0)
    status = st.empty()

    translated_images = []

    ##########################################################
    # TRANSLATE SETIAP HALAMAN
    ##########################################################

    for i, image_path in enumerate(image_files):

        status.text(
            f"Memproses halaman {i+1} dari {len(image_files)}"
        )

        output_path = os.path.join(
            output_folder,
            os.path.basename(image_path)
        )

        pipeline.process_manga_page(
            image_path,
            output_path,
            conf_threshold=conf
        )

        translated_images.append(output_path)

        progress.progress(
            (i + 1) / len(image_files)
        )

    ##########################################################
    # GABUNGKAN MENJADI PDF
    ##########################################################

    status.text("Membuat PDF...")

    images = []

    for img_path in translated_images:
        images.append(
            Image.open(img_path).convert("RGB")
        )

    pdf_output = os.path.join(
        temp_dir,
        "translated.pdf"
    )

    images[0].save(
        pdf_output,
        save_all=True,
        append_images=images[1:]
    )

    ##########################################################
    # DOWNLOAD PDF
    ##########################################################

    status.success("Selesai")

    with open(pdf_output, "rb") as f:

        st.download_button(
            "⬇ Download PDF",
            data=f,
            file_name="translated.pdf",
            mime="application/pdf",
            on_click="ignore"
        )

    ##########################################################
    # CLEANUP
    ##########################################################

    shutil.rmtree(temp_dir)
  

# ==========================================================
# ZIP PROCESSING
# ==========================================================

def process_zip(uploaded_file):

    st.write("process_zip dijalankan")

    temp_dir = tempfile.mkdtemp()

    zip_path = os.path.join(
        temp_dir,
        uploaded_file.name
    )

    with open(zip_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    try:

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(temp_dir)
        
        st.write("Temp directory:")
        st.write(temp_dir)

        st.write("Isi folder utama:")
        st.write(os.listdir(temp_dir))

    except zipfile.BadZipFile:

        st.error("File ZIP tidak valid.")

        shutil.rmtree(temp_dir)

        return

    image_files = []

    for ext in ["jpg", "jpeg", "png", "webp"]:

        image_files.extend(

            glob.glob(

                os.path.join(
                    temp_dir,
                    "**",
                    f"*.{ext}"
                ),

                recursive=True

            )

        )

    image_files.sort()

    ##########################################################
    # VALIDASI ZIP
    ##########################################################

    if len(image_files) == 0:

        st.error(
            "ZIP tidak berisi gambar."
        )

        shutil.rmtree(temp_dir)

        return

    MAX_PAGE = 35

    if len(image_files) > MAX_PAGE:

        st.error(
            f"Maksimal {MAX_PAGE} halaman."
        )

        shutil.rmtree(temp_dir)

        return

   

    ##########################################################
    # PREVIEW
    ##########################################################

    st.success(
        f"Ditemukan {len(image_files)} halaman."
    )

    preview_df = pd.DataFrame({

        "Halaman":
            range(
                1,
                len(image_files)+1
            ),

        "Nama File":
            [
                os.path.basename(i)
                for i in image_files
            ]

    })

    st.dataframe(
        preview_df,
        use_container_width=True
    )

    preview = Image.open(
        image_files[0]
    )

    st.image(
        preview,
        caption="Preview halaman pertama",
        use_container_width=True
    )

    ##########################################################
    # TRANSLATE BUTTON
    ##########################################################

    if not st.button(
        "🚀 Translate ZIP",
        use_container_width=True
    ):
        return

    output_folder = os.path.join(
        temp_dir,
        "translated"
    )

    os.makedirs(
        output_folder,
        exist_ok=True
    )

    progress = st.progress(0)

    status = st.empty()

    ##########################################################
    # LOOP SETIAP HALAMAN
    ##########################################################

    for i, image_path in enumerate(image_files):

        status.text(

            f"Memproses halaman {i+1} dari {len(image_files)}"

        )

        output_path = os.path.join(

            output_folder,

            os.path.basename(image_path)

        )

        try:

            pipeline.process_manga_page(

                image_path,

                output_path,

                conf_threshold=conf

            )

        except Exception as e:

            st.warning(

                f"Gagal memproses {os.path.basename(image_path)}"

            )

            print(e)

        progress.progress(

            (i+1) / len(image_files)

        )

    ##########################################################
    # ZIP HASIL
    ##########################################################

    status.success("Membuat ZIP hasil...")

    zip_result = shutil.make_archive(

        os.path.join(
            temp_dir,
            "translated_manga"
        ),

        "zip",

        output_folder

    )

    status.success("Selesai")

    st.success("Seluruh halaman berhasil diterjemahkan.")

    ##########################################################
    # DOWNLOAD
    ##########################################################

    with open(zip_result, "rb") as f:

        st.download_button(

            "⬇ Download Hasil ZIP",
            data=f,
            file_name="translated_manga.zip",
            mime="application/zip",
            on_click="ignore"

        )

    ##########################################################
    # CLEANUP
    ##########################################################

    shutil.rmtree(temp_dir)

# ==========================================================
# MAIN
# ==========================================================

if uploaded_file is not None:

    st.write(uploaded_file.name)

    extension = os.path.splitext(uploaded_file.name)[1].lower()

    st.write(extension)

    ##########################################################
    # SINGLE IMAGE
    ##########################################################

    if extension in [
            ".jpg",
            ".jpeg",
            ".png",
            ".webp",
        ]:
            process_single_image(uploaded_file)

    ##########################################################
    # PDF
    ##########################################################

    elif extension == ".pdf":
            process_pdf(uploaded_file)


    ##########################################################
    # ZIP
    ##########################################################

    elif extension == ".zip":

        st.write("Masuk ke process_zip()")

        process_zip(
            uploaded_file
        )

    ##########################################################
    # FILE TIDAK DIDUKUNG
    ##########################################################

    else:

        st.error(
            "Format file tidak didukung."
        )

else:

    st.info(
        """
Silakan upload salah satu:

- JPG
- JPEG
- PNG
- ZIP (maksimal 25 halaman)
"""
    )
