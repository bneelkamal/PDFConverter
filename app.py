import streamlit as st
import os
import tempfile
from io import BytesIO
import zipfile
from PIL import Image
from pdf2image import convert_from_bytes
from pdf2docx import Converter
import base64
import time

import pytesseract
from pypdf import PdfWriter, PdfReader, __version__ as pypdf_version
import ocrmypdf # New dependency for Section 3 OCR

# --- Configuration & Page Setup ---
st.set_page_config(page_title="File Converter Hub", layout="wide", initial_sidebar_state="expanded")
st.title("📄 File Converter Hub 🔄")
st.write("Convert PDFs (Sec 1), Combine Images to PDF (Sec 2), or Merge PDFs (Sec 3 - with optional OCR).")

# --- Helper Functions (pdf_to_images_st, pdf_to_word_st, create_zip_from_images, images_to_searchable_pdf_st, images_to_plain_pdf_st are unchanged from previous version) ---
@st.cache_data(show_spinner=False)
def pdf_to_images_st(pdf_bytes, dpi, img_format, poppler_path=None):
    try: images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=img_format.lower(), poppler_path=poppler_path); return images
    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower(): st.info("PDF to Image: Ensure Poppler is set up (PATH/packages.txt).")
        raise 
        
@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file: temp_pdf_file.write(pdf_bytes); temp_pdf_path = temp_pdf_file.name
        output_docx_buffer = BytesIO()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as temp_docx_file:
            temp_docx_path = temp_docx_file.name; cv = Converter(temp_pdf_path); cv.convert(temp_docx_path); cv.close()
            with open(temp_docx_path, 'rb') as f_docx: output_docx_buffer.write(f_docx.read())
        output_docx_buffer.seek(0); return output_docx_buffer
    except Exception as e:
        if "tesseract" in str(e).lower(): st.info("PDF to Word (Scanned): Ensure Tesseract is set up for pdf2docx OCR.")
        raise
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try: os.remove(temp_pdf_path)
            except Exception: st.warning(f"Could not remove temp PDF: {temp_pdf_path}")

def create_zip_from_images(images, base_filename, img_format):
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            img_filename = f"{base_filename}_page_{str(i + 1).zfill(len(str(len(images))))}.{img_format.lower()}"
            img_byte_arr = BytesIO(); save_img = img
            if img.mode == 'RGBA' and img_format.lower() == 'jpeg': save_img = img.convert('RGB')
            elif img.mode == 'P': save_img = img.convert('RGB')
            save_img.save(img_byte_arr, format=img_format.upper()); img_byte_arr = img_byte_arr.getvalue()
            zip_file.writestr(img_filename, img_byte_arr)
    zip_buffer.seek(0); return zip_buffer

@st.cache_data(show_spinner=False)
def images_to_searchable_pdf_st(image_file_objects, ocr_language='eng'):
    writer = PdfWriter(); total = len(image_file_objects)
    if total == 0: st.warning("No images for PDF."); return None
    pb, stxt = st.empty(), st.empty(); pb.progress(0.0)
    for i, fo in enumerate(image_file_objects):
        fn, ib = fo.name, fo.getvalue(); fo.seek(0)
        stxt.text(f"OCR Img {i+1}/{total} ('{fn}', lang: {ocr_language})...")
        try:
            img = Image.open(BytesIO(ib)); 
            if img.mode in ['RGBA', 'P']: img = img.convert('RGB')
            page_bytes = pytesseract.image_to_pdf_or_hocr(img, lang=ocr_language, extension='pdf')
            reader = PdfReader(BytesIO(page_bytes))
            if reader.pages: writer.add_page(reader.pages[0])
            else: st.warning(f"OCR: No page from '{fn}'."); continue 
        except pytesseract.TesseractNotFoundError: st.error("Tesseract not found."); stxt.empty(); pb.empty(); return None
        except pytesseract.TesseractError as te: st.error(f"Tesseract error '{fn}': {te}"); stxt.empty(); pb.empty(); return None
        except Exception as e: st.error(f"Error OCRing '{fn}': {e}"); stxt.empty(); pb.empty(); return None
        pb.progress((i + 1) / total)
    stxt.text("Finalizing searchable PDF..."); 
    if not writer.pages: st.warning("No pages processed."); stxt.empty(); pb.empty(); return None
    writer.add_metadata({"/Producer": ""}); buf = BytesIO(); writer.write(buf); buf.seek(0)
    stxt.success("Searchable PDF created!"); time.sleep(1); stxt.empty(); pb.empty()
    return buf

@st.cache_data(show_spinner=False)
def images_to_plain_pdf_st(image_file_objects):
    pil_imgs, stxt = [], st.empty(); total = len(image_file_objects)
    for i, fo in enumerate(image_file_objects):
        fn, ib = fo.name, fo.getvalue(); fo.seek(0)
        stxt.text(f"Processing Img {i+1}/{total} ('{fn}')...")
        try: 
            img = Image.open(BytesIO(ib)); 
            if img.mode in ['RGBA', 'P']: img = img.convert('RGB')
            pil_imgs.append(img)
        except Exception as e: st.error(f"Error opening '{fn}': {e}"); stxt.empty(); return None
    if not pil_imgs: st.warning("No valid images."); stxt.empty(); return None
    stxt.text("Creating image-only PDF..."); buf_pil = BytesIO()
    try: pil_imgs[0].save(buf_pil, format='PDF', save_all=True, append_images=pil_imgs[1:]); buf_pil.seek(0)
    except Exception as e: st.error(f"Pillow PDF save error: {e}"); stxt.empty(); return None
    writer = PdfWriter()
    try:
        reader = PdfReader(buf_pil)
        for page in reader.pages: writer.add_page(page)
        writer.add_metadata({"/Producer": ""}); out_buf = BytesIO(); writer.write(out_buf); out_buf.seek(0)
        stxt.success("Image-only PDF created!"); time.sleep(1); stxt.empty()
        return out_buf
    except Exception as e: st.error(f"PDF metadata finalization error: {e}"); stxt.empty(); return None

@st.cache_data(show_spinner=False)
def merge_pdfs_st(ordered_pdf_file_objects): # This function JUST merges, no OCR here.
    pdf_writer = PdfWriter()
    stxt = st.empty(); total = len(ordered_pdf_file_objects)
    if total == 0: st.warning("No PDFs to merge."); return None
    pages_added_count = 0
    for i, fo in enumerate(ordered_pdf_file_objects):
        fn = fo.name; stxt.text(f"Merging PDF {i+1}/{total} ('{fn}')...")
        try:
            pdf_bytes = fo.getvalue(); fo.seek(0)
            reader = PdfReader(BytesIO(pdf_bytes))
            if reader.is_encrypted: st.warning(f"'{fn}' is encrypted. Skipping."); continue
            for page in reader.pages: pdf_writer.add_page(page); pages_added_count +=1
        except Exception as e: st.error(f"Error merging '{fn}': {e}. Skipping."); continue
    if not pdf_writer.pages: st.warning("No pages added from PDFs."); stxt.empty(); return None
    pdf_writer.add_metadata({"/Producer": ""})
    buf = BytesIO(); pdf_writer.write(buf); buf.seek(0)
    stxt.success(f"Merged {pages_added_count} pages from {total} PDF(s) (pre-OCR)."); time.sleep(1); stxt.empty()
    return buf.getvalue() # Return bytes

# --- NEW Helper Function for OCRing an existing PDF with ocrmypdf ---
@st.cache_data(show_spinner=False)
def ocr_existing_pdf_st(input_pdf_bytes, language='eng', deskew=True, force_ocr=True):
    """Performs OCR on an existing PDF's bytes using ocrmypdf."""
    if not input_pdf_bytes: return None
    stxt = st.empty()
    stxt.text(f"Performing OCR on merged PDF (lang: {language}). This may take time...")
    
    output_pdf_buffer = BytesIO()
    try:
        # ocrmypdf needs input as BytesIO or path, output as BytesIO or path
        ocrmypdf.ocr(
            BytesIO(input_pdf_bytes),
            output_pdf_buffer,
            language=language,
            deskew=deskew,
            force_ocr=force_ocr,  # Important: re-does OCR even if text layers exist
            skip_text=False,      # Ensures text layer is added
            progress_bar=False    # Streamlit handles its own progress/spinner
            # Add other ocrmypdf options as needed, e.g., optimize=0 for faster processing if quality allows
        )
        output_pdf_buffer.seek(0)
        stxt.success("OCR processing on merged PDF complete!")
        time.sleep(1)
        stxt.empty()
        return output_pdf_buffer.getvalue() # Return bytes
    except ocrmypdf.exceptions.TesseractNotFoundError:
        st.error("ocrmypdf: Tesseract OCR engine not found. Check setup (PATH/packages.txt).")
        stxt.empty(); return None
    except ocrmypdf.exceptions.MissingDependencyError as e:
        st.error(f"ocrmypdf: Missing system dependency: {e}. (e.g., ghostscript might be needed - check packages.txt for deployment)")
        stxt.empty(); return None
    except Exception as e:
        st.error(f"ocrmypdf: Error during OCR processing: {e}")
        stxt.empty(); return None


# --- Sidebar ---
st.sidebar.title("File Converter App")
st.sidebar.markdown("---")
st.sidebar.header("PDF-WORD-IMAGE-MERGE")
st.sidebar.info(
    f"""
    Convert PDFs, Combine Images, or Merge PDFs (with optional OCR).
    - OCR (Sec 2 & 3) requires Tesseract & language packs.
    - PDF to Image (Sec 1) requires Poppler.
    - If deployed (e.g. Streamlit Cloud), add `tesseract-ocr`, `tesseract-ocr-eng` (and other langs), `ghostscript`, `poppler-utils` to `packages.txt`. And `ocrmypdf` to `requirements.txt`.
    (pypdf: {pypdf_version}, ocrmypdf available if imported)
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Refreshed: {time.strftime('%Y%m%d-%H%M%S')}")

# --- Initialize Session State ---
# Section 1
if 's1_results' not in st.session_state: st.session_state.s1_results = []
# Section 2
if 'img_to_pdf_done' not in st.session_state: st.session_state.img_to_pdf_done = False
if 'img_to_pdf_result_bytes' not in st.session_state: st.session_state.img_to_pdf_result_bytes = None
if 'is_ocr_pdf_s2' not in st.session_state: st.session_state.is_ocr_pdf_s2 = False # Specific to S2
if 'ordered_image_files' not in st.session_state: st.session_state.ordered_image_files = []
if 'last_uploaded_file_ids_img_to_pdf' not in st.session_state: st.session_state.last_uploaded_file_ids_img_to_pdf = []
# Section 3 (Merge PDFs)
if 's3_ordered_pdf_files' not in st.session_state: st.session_state.s3_ordered_pdf_files = []
if 's3_last_uploaded_file_ids' not in st.session_state: st.session_state.s3_last_uploaded_file_ids = []
if 's3_final_pdf_bytes' not in st.session_state: st.session_state.s3_final_pdf_bytes = None # Stores final PDF (merged or merged+OCR'd)
if 's3_process_done' not in st.session_state: st.session_state.s3_process_done = False
if 's3_is_ocr_applied' not in st.session_state: st.session_state.s3_is_ocr_applied = False # Flag for S3 OCR


# --- UI Section 1: PDF Conversions (Unchanged) ---
st.header("1. Convert PDF(s) to Image/Word")
# ... (Section 1 code remains the same as your last version) ...
uploaded_pdf_files_s1 = st.file_uploader( "Upload one or more PDF files", type=["pdf"], key="pdf_uploader_key_s1", accept_multiple_files=True)
if uploaded_pdf_files_s1:
    st.markdown("---"); col1_pdf, col2_pdf = st.columns([1, 2])
    with col1_pdf:
        st.subheader("Conversion Options (for all uploaded PDFs)")
        pdf_conversion_type_s1 = st.radio( "Convert All Uploaded PDFs To:", ("Images", "Word Document (.docx)"), key="pdf_conversion_type_key_s1", horizontal=True, on_change=lambda: st.session_state.update(s1_results=[]))
        pdf_options_s1 = {}
        if pdf_conversion_type_s1 == "Images":
            pdf_options_s1['img_format'] = st.selectbox("Image Format:", ["PNG", "JPEG"], key="img_format_s1")
            pdf_options_s1['dpi'] = st.slider("Image Quality (DPI):", 72, 600, 200, 10, key="dpi_s1")
        if st.button("🚀 Convert All PDFs", key="convert_all_pdfs_button_s1", use_container_width=True):
            st.session_state.s1_results = []
            with st.spinner(f"Processing {len(uploaded_pdf_files_s1)} PDF(s)..."):
                for file_obj in uploaded_pdf_files_s1:
                    result_entry = {'input_filename': file_obj.name, 'conversion_type': pdf_conversion_type_s1, 'status': 'failure', 'output': None, 'messages': []}
                    try:
                        pdf_bytes_in = file_obj.getvalue(); file_obj.seek(0)
                        if pdf_conversion_type_s1 == "Images":
                            result_entry['img_format_options'] = pdf_options_s1.copy()
                            images = pdf_to_images_st(pdf_bytes_in, pdf_options_s1['dpi'], pdf_options_s1['img_format'])
                            if images: result_entry.update({'output': images, 'status': 'success', 'messages': [f"{len(images)} images generated."]})
                            else: result_entry['messages'].append("Image conversion returned no images.")
                        elif pdf_conversion_type_s1 == "Word Document (.docx)":
                            docx_bytes_io = pdf_to_word_st(pdf_bytes_in)
                            if docx_bytes_io: result_entry.update({'output': docx_bytes_io.getvalue(), 'status': 'success', 'messages': ["Word document generated."]})
                            else: result_entry['messages'].append("Word conversion returned no data.")
                    except Exception as e: result_entry['messages'].append(f"Error: {str(e)}")
                    st.session_state.s1_results.append(result_entry)
            if st.session_state.s1_results: st.success(f"Batch conversion attempted. See results.")
            else: st.warning("No PDFs processed.")
    with col2_pdf:
        if st.session_state.s1_results:
            st.subheader("Batch Conversion Results")
            for idx, result in enumerate(st.session_state.s1_results):
                with st.expander(f"Results for: {result['input_filename']} (Status: {result['status']})", expanded=(result['status']=='failure')):
                    if result['status'] == 'success':
                        if result['conversion_type'] == "Images":
                            images_output, img_opts, base_fn = result['output'], result['img_format_options'], os.path.splitext(result['input_filename'])[0]
                            st.write(result['messages'][0] if result['messages'] else f"{len(images_output)} image(s).")
                            zip_buffer = create_zip_from_images(images_output, base_fn, img_opts['img_format'])
                            st.download_button(f"⬇️ Download Images (.zip)", zip_buffer, f"{base_fn}_images.zip", "application/zip", key=f"dl_zip_s1_{idx}_{base_fn}", use_container_width=True)
                            for i, img_res in enumerate(images_output):
                                if i < 2: st.image(img_res, caption=f"Page {i+1}", width=200)
                                elif i == 2: st.write(f"(+ {len(images_output) - 2} more)"); break
                        elif result['conversion_type'] == "Word Document (.docx)":
                            st.write(result['messages'][0] if result['messages'] else "Word doc ready.")
                            st.download_button("⬇️ Download Word (.docx)", result['output'], f"{os.path.splitext(result['input_filename'])[0]}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"dl_docx_s1_{idx}", use_container_width=True)
                    else: 
                        for msg in result['messages']: st.error(msg)
        elif uploaded_pdf_files_s1: st.info("Select options & click 'Convert All PDFs'.")


# --- UI Section 2: Image to PDF Conversion (Unchanged) ---
st.divider()
st.header("2. Combine Images to PDF with OCR (optional)")
# ... (Section 2 code remains the same as your last version) ...
uploaded_image_files_s2 = st.file_uploader("Upload image(s) (PNG, JPG, etc.)", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=True, key="uploader_s2")
if uploaded_image_files_s2:
    current_file_ids_s2 = sorted([f.file_id for f in uploaded_image_files_s2])
    if st.session_state.last_uploaded_file_ids_img_to_pdf != current_file_ids_s2:
        st.session_state.ordered_image_files = list(uploaded_image_files_s2)
        st.session_state.last_uploaded_file_ids_img_to_pdf = current_file_ids_s2
        st.session_state.img_to_pdf_done, st.session_state.img_to_pdf_result_bytes = False, None
else:
    if st.session_state.ordered_image_files or st.session_state.last_uploaded_file_ids_img_to_pdf:
        st.session_state.ordered_image_files, st.session_state.last_uploaded_file_ids_img_to_pdf = [], []
        st.session_state.img_to_pdf_done, st.session_state.img_to_pdf_result_bytes = False, None
if st.session_state.ordered_image_files:
    st.subheader("Order Pages (Images)"); st.caption("Use buttons to reorder images for the PDF.")
    for i, file_obj in enumerate(st.session_state.ordered_image_files):
        cols_s2_order = st.columns([0.1, 0.1, 0.6, 0.1, 0.1])
        with cols_s2_order[0]: st.write(f"{i+1}.")
        with cols_s2_order[1]: st.image(file_obj.getvalue(), width=50); file_obj.seek(0)
        with cols_s2_order[2]: st.write(file_obj.name)
        with cols_s2_order[3]:
            if i > 0: 
                if st.button("🔼", key=f"s2_up_{file_obj.id}_{i}", help="Move Up"):
                    st.session_state.ordered_image_files.insert(i-1, st.session_state.ordered_image_files.pop(i)); st.rerun()
        with cols_s2_order[4]:
            if i < len(st.session_state.ordered_image_files) - 1:
                if st.button("🔽", key=f"s2_down_{file_obj.id}_{i}", help="Move Down"):
                    st.session_state.ordered_image_files.insert(i+1, st.session_state.ordered_image_files.pop(i)); st.rerun()
    st.markdown("---")
    perform_ocr_s2 = st.checkbox("Make PDF searchable (OCR)?", value=True, key="ocr_check_s2", help="Slower, makes text searchable.")
    ocr_lang_code_s2 = "eng" # Default for S2
    if perform_ocr_s2:
        ocr_lang_opts_s2 = {"English":"eng", "Spanish":"spa", "French":"fra", "German":"deu", "Other":"manual"}
        sel_lang_name_s2 = st.selectbox("OCR Language (Images):", list(ocr_lang_opts_s2.keys()), 0, key="ocr_lang_s2")
        ocr_lang_code_s2 = ocr_lang_opts_s2[sel_lang_name_s2]
        if ocr_lang_code_s2 == "manual": ocr_lang_code_s2 = st.text_input("Tesseract lang code (Images):", "eng", key="ocr_manual_s2").lower().strip()
        st.caption(f"Using OCR lang (Images): `{ocr_lang_code_s2}`.")
    if st.button("🖼️ Combine Images to PDF", key="combine_img_button_s2", use_container_width=True):
        st.session_state.img_to_pdf_done, st.session_state.img_to_pdf_result_bytes = False, None
        current_ord_files_s2 = st.session_state.ordered_image_files
        pdf_buffer_s2, st.session_state.is_ocr_pdf_s2 = None, perform_ocr_s2 # Use s2 specific flag
        if perform_ocr_s2:
            if not ocr_lang_code_s2: st.error("Manual OCR lang code empty.")
            else: pdf_buffer_s2 = images_to_searchable_pdf_st(current_ord_files_s2, ocr_language=ocr_lang_code_s2)
        else: pdf_buffer_s2 = images_to_plain_pdf_st(current_ord_files_s2)
        if pdf_buffer_s2: st.session_state.img_to_pdf_result_bytes, st.session_state.img_to_pdf_done = pdf_buffer_s2.getvalue(), True # getvalue for bytes
        else: st.error(f"Image to {'OCR ' if perform_ocr_s2 else ''}PDF failed.")
    if st.session_state.get('img_to_pdf_done') and st.session_state.get('img_to_pdf_result_bytes'):
        st.subheader("Combined PDF Result (from Images)")
        time_str = time.strftime('%Y%m%d-%H%M%S')
        # Use s2 specific OCR flag
        out_pdf_fn_s2 = f"{'ocr_img_' + ocr_lang_code_s2 if st.session_state.is_ocr_pdf_s2 else 'img'}_{time_str}.pdf"
        dl_label_s2 = f"⬇️ Download {'Searchable ' if st.session_state.is_ocr_pdf_s2 else ''}PDF (from Images)"
        st.download_button(dl_label_s2, st.session_state.img_to_pdf_result_bytes, out_pdf_fn_s2, "application/pdf", key="dl_combined_pdf_s2", use_container_width=True)


# --- UI Section 3: Merge Multiple PDFs (with optional OCR) ---
st.divider()
st.header("3. Merge  PDF's with OCR (optional)")
uploaded_pdf_files_s3 = st.file_uploader(
    "Upload two or more PDF files to merge",
    type=["pdf"],
    accept_multiple_files=True,
    key="pdf_uploader_s3"
)

# Manage the ordered list of PDFs for merging in session state
if uploaded_pdf_files_s3:
    current_file_ids_s3 = sorted([f.file_id for f in uploaded_pdf_files_s3])
    if st.session_state.s3_last_uploaded_file_ids != current_file_ids_s3:
        st.session_state.s3_ordered_pdf_files = list(uploaded_pdf_files_s3)
        st.session_state.s3_last_uploaded_file_ids = current_file_ids_s3
        st.session_state.s3_process_done = False
        st.session_state.s3_final_pdf_bytes = None
else:
    if st.session_state.s3_ordered_pdf_files or st.session_state.s3_last_uploaded_file_ids:
        st.session_state.s3_ordered_pdf_files = []
        st.session_state.s3_last_uploaded_file_ids = []
        st.session_state.s3_process_done = False
        st.session_state.s3_final_pdf_bytes = None

if st.session_state.s3_ordered_pdf_files:
    if len(st.session_state.s3_ordered_pdf_files) < 1: # Allow single PDF for OCR now
        st.info("Upload at least one PDF to process (or two to merge).")
    
    st.subheader("Order PDFs for Processing")
    st.caption("Current order of PDFs. Use buttons to reorder before merging/OCR.")
    for i, file_obj in enumerate(st.session_state.s3_ordered_pdf_files):
        cols_s3_order = st.columns([0.1, 0.7, 0.1, 0.1])
        with cols_s3_order[0]: st.write(f"{i+1}.")
        with cols_s3_order[1]: st.write(file_obj.name)
        with cols_s3_order[2]:
            if i > 0: 
                if st.button("🔼", key=f"s3_up_{file_obj.id}_{i}", help="Move Up"):
                    st.session_state.s3_ordered_pdf_files.insert(i-1, st.session_state.s3_ordered_pdf_files.pop(i)); st.rerun()
        with cols_s3_order[3]:
            if i < len(st.session_state.s3_ordered_pdf_files) - 1:
                if st.button("🔽", key=f"s3_down_{file_obj.id}_{i}", help="Move Down"):
                    st.session_state.s3_ordered_pdf_files.insert(i+1, st.session_state.s3_ordered_pdf_files.pop(i)); st.rerun()
    st.markdown("---")

    perform_ocr_s3 = st.checkbox("Make final PDF searchable (perform OCR)?", value=False, key="ocr_check_s3",
                                 help="If checked, OCRmyPDF will process the merged PDF. Can be slow. Tesseract & Ghostscript needed.")
    ocr_lang_code_s3 = "eng" # Default for S3
    if perform_ocr_s3:
        ocr_lang_opts_s3 = {"English":"eng", "Spanish":"spa", "French":"fra", "German":"deu", "Other":"manual"}
        sel_lang_name_s3 = st.selectbox("OCR Language (Merged PDF):", list(ocr_lang_opts_s3.keys()), 0, key="ocr_lang_s3")
        ocr_lang_code_s3 = ocr_lang_opts_s3[sel_lang_name_s3]
        if ocr_lang_code_s3 == "manual": ocr_lang_code_s3 = st.text_input("Tesseract lang code (Merged PDF):", "eng", key="ocr_manual_s3").lower().strip()
        st.caption(f"Using OCR lang (Merged PDF): `{ocr_lang_code_s3}`.")

    button_label_s3 = "🔗 Merge PDFs"
    if perform_ocr_s3:
        button_label_s3 = "🔗 Merge & Make Searchable (OCR)"
    
    process_disabled_s3 = False
    if len(st.session_state.s3_ordered_pdf_files) < 2 and not perform_ocr_s3: # Need >=2 to merge if not OCRing a single file
        process_disabled_s3 = True
        st.caption("Upload at least two PDFs to merge if not performing OCR on a single PDF.")
    elif len(st.session_state.s3_ordered_pdf_files) < 1: # Need >=1 if OCRing
         process_disabled_s3 = True


    if st.button(button_label_s3, key="process_pdfs_button_s3", use_container_width=True, disabled=process_disabled_s3):
        st.session_state.s3_process_done = False
        st.session_state.s3_final_pdf_bytes = None
        st.session_state.s3_is_ocr_applied = perform_ocr_s3 # Store if OCR was intended

        current_ordered_pdfs_s3 = st.session_state.s3_ordered_pdf_files
        
        # Step 1: Merge the PDFs (if more than one, or just take the first if only one and OCRing)
        merged_pdf_bytes = None
        if len(current_ordered_pdfs_s3) == 1 and perform_ocr_s3: # Single PDF for OCR
            st.write(f"Processing single PDF for OCR: {current_ordered_pdfs_s3[0].name}")
            merged_pdf_bytes = current_ordered_pdfs_s3[0].getvalue()
            current_ordered_pdfs_s3[0].seek(0)
        elif len(current_ordered_pdfs_s3) >= 1: # Also covers >1 for merge, or 1 for no-OCR path (though button might be disabled)
            merged_pdf_bytes = merge_pdfs_st(current_ordered_pdfs_s3) # Returns bytes
        
        if not merged_pdf_bytes:
            st.error("Failed to merge PDFs or load single PDF for OCR.")
        else:
            final_processed_pdf_bytes = merged_pdf_bytes
            if perform_ocr_s3:
                if not ocr_lang_code_s3 and ocr_lang_opts_s3[sel_lang_name_s3] == "manual": # Check if manual input is empty
                     st.error("Manual OCR language code cannot be empty for merged PDF OCR.")
                     final_processed_pdf_bytes = None # Prevent download
                else:
                    with st.spinner(f"Applying OCR (lang: {ocr_lang_code_s3}) to the PDF... This can take a significant amount of time."):
                        final_processed_pdf_bytes = ocr_existing_pdf_st(merged_pdf_bytes, language=ocr_lang_code_s3)
            
            if final_processed_pdf_bytes:
                st.session_state.s3_final_pdf_bytes = final_processed_pdf_bytes
                st.session_state.s3_process_done = True
            else:
                st.error("Final PDF processing (possibly OCR step) failed.")


    if st.session_state.get('s3_process_done') and st.session_state.get('s3_final_pdf_bytes'):
        st.subheader("Processed PDF Result")
        time_str = time.strftime('%Y%m%d-%H%M%S')
        
        # Use s3_is_ocr_applied to determine filename and label
        is_ocr_applied_s3 = st.session_state.get('s3_is_ocr_applied', False)
        ocr_tag = f"_ocr_{ocr_lang_code_s3}" if is_ocr_applied_s3 else ""
        out_pdf_filename_s3 = f"processed_document{ocr_tag}_{time_str}.pdf"
        download_label_s3 = f"⬇️ Download {'Searchable ' if is_ocr_applied_s3 else ''}Processed PDF"
        
        st.download_button(
            label=download_label_s3,
            data=st.session_state.s3_final_pdf_bytes,
            file_name=out_pdf_filename_s3,
            mime="application/pdf",
            key="download_processed_pdf_s3",
            use_container_width=True
        )

# Final info message
if not uploaded_pdf_files_s1 and not st.session_state.ordered_image_files and not st.session_state.s3_ordered_pdf_files:
    st.info("☝️ Upload files in any section to get started.")
