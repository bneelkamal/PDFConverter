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

import pytesseract # For image to single-page OCR'd PDF (if ever needed directly, less so now)
from pypdf import PdfWriter, PdfReader, __version__ as pypdf_version
import ocrmypdf # For final PDF OCR

# --- Configuration & Page Setup ---
st.set_page_config(page_title="File Converter Hub", layout="wide", initial_sidebar_state="expanded")
st.title("📄 File Converter Hub 🔄")
st.write("Convert PDFs to other formats (Sec 1), or Combine various files into a single PDF (Sec 2).")

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def pdf_to_images_st(pdf_bytes, dpi, img_format, poppler_path=None):
    """Converts PDF bytes to a list of PIL Image objects."""
    try: images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=img_format.lower(), poppler_path=poppler_path); return images
    except Exception as e:
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower(): st.info("PDF to Image: Ensure Poppler is set up (PATH/packages.txt).")
        raise 
        
@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    """Converts PDF bytes to Word (.docx) bytes."""
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
    """Creates a ZIP archive containing image files in memory."""
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
def ocr_existing_pdf_st(input_pdf_bytes, language='eng', deskew=True, force_ocr=True, context_section="Combined"):
    """Performs OCR on an existing PDF's bytes using ocrmypdf."""
    if not input_pdf_bytes: return None
    stxt = st.empty()
    stxt.text(f"{context_section}-Performing OCR on PDF (lang: {language}). This may take considerable time...")
    output_pdf_buffer = BytesIO()
    try:
        ocrmypdf.ocr( BytesIO(input_pdf_bytes), output_pdf_buffer, language=language, deskew=deskew, force_ocr=force_ocr, skip_text=False, progress_bar=False)
        output_pdf_buffer.seek(0)
        stxt.success(f"{context_section}-OCR processing on PDF complete!"); time.sleep(2); stxt.empty()
        return output_pdf_buffer.getvalue()
    except ocrmypdf.exceptions.TesseractNotFoundError: st.error(f"{context_section}-ocrmypdf: Tesseract OCR not found. Check setup."); stxt.empty(); return None
    except ocrmypdf.exceptions.MissingDependencyError as e: st.error(f"{context_section}-ocrmypdf: Missing dependency: {e}. (e.g., ghostscript)."); stxt.empty(); return None
    except Exception as e: st.error(f"{context_section}-ocrmypdf: Error during OCR: {e}"); stxt.empty(); return None

@st.cache_data(show_spinner=False)
def image_to_single_page_pdf_bytes(image_file_object, filename_for_error="image"):
    """Converts an image UploadedFile object to bytes of a single-page PDF."""
    try:
        img_pil = Image.open(image_file_object) 
        if img_pil.mode == 'RGBA' or img_pil.mode == 'P':
            img_pil = img_pil.convert('RGB')
        pdf_buffer = BytesIO()
        img_pil.save(pdf_buffer, format='PDF', resolution=150.0) # Good default resolution
        pdf_buffer.seek(0)
        return pdf_buffer.getvalue()
    except Exception as e:
        st.error(f"Error converting image '{filename_for_error}' to PDF page: {e}")
        return None

# --- Sidebar ---
st.sidebar.title("File Converter App")
st.sidebar.markdown("---")
st.sidebar.header("All-in-One PDF & Image Hub")
st.sidebar.info(
    f"""
    - **Sec 1:** PDF to Image/Word
    - **Sec 2:** Combine PDF/Images (opt. OCR)

    **Setup Notes (for deployed apps like Streamlit Cloud):**
    - **`requirements.txt`:** `streamlit`, `Pillow`, `pdf2image`, `pdf2docx`, `pytesseract`, `pypdf`, `ocrmypdf`
    - **`packages.txt`:** `tesseract-ocr`, `tesseract-ocr-eng` (and other languages e.g., `tesseract-ocr-deu`), `ghostscript`, `poppler-utils`
    (pypdf: {pypdf_version})
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Refreshed: {time.strftime('%Y%m%d-%H%M%S')}")

# --- Initialize Session State ---
# Section 1 (PDF Conversions)
if 's1_results' not in st.session_state: 
    st.session_state.s1_results = []

# Section 2 (Consolidated Combine Files)
if 's2_ordered_items' not in st.session_state: 
    st.session_state.s2_ordered_items = [] # List of dicts: {'file': UploadedFile, 'type': 'pdf'/'image', 'id': file.id, 'name': file.name}
if 's2_last_uploaded_file_ids' not in st.session_state: 
    st.session_state.s2_last_uploaded_file_ids = []
if 's2_final_pdf_bytes' not in st.session_state: 
    st.session_state.s2_final_pdf_bytes = None
if 's2_process_done' not in st.session_state: 
    st.session_state.s2_process_done = False
if 's2_is_ocr_applied' not in st.session_state: 
    st.session_state.s2_is_ocr_applied = False


# --- UI Section 1: PDF Conversions (Largely Unchanged) ---
st.header("1. Convert PDF(s) to Other Formats")
uploaded_pdf_files_s1 = st.file_uploader(
    "Upload one or more PDF files", 
    type=["pdf"], 
    key="pdf_uploader_s1", # Unique key
    accept_multiple_files=True
)
if uploaded_pdf_files_s1:
    st.markdown("---"); col1_s1, col2_s1 = st.columns([1, 2])
    with col1_s1:
        st.subheader("Conversion Options")
        pdf_conversion_type_s1 = st.radio( 
            "Convert All Uploaded PDFs To:", 
            ("Images", "Word Document (.docx)"), 
            key="conversion_type_s1", 
            horizontal=True,
            on_change=lambda: st.session_state.update(s1_results=[])
        )
        pdf_options_s1 = {}
        if pdf_conversion_type_s1 == "Images":
            pdf_options_s1['img_format'] = st.selectbox("Image Format:", ["PNG", "JPEG"], key="img_format_s1")
            pdf_options_s1['dpi'] = st.slider("Image Quality (DPI):", 72, 600, 200, 10, key="dpi_s1")

        if st.button("🚀 Convert All PDFs", key="convert_button_s1", use_container_width=True):
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
                    except Exception as e: result_entry['messages'].append(f"Error processing '{file_obj.name}': {str(e)}")
                    st.session_state.s1_results.append(result_entry)
            if st.session_state.s1_results: st.success(f"Batch conversion of {len(uploaded_pdf_files_s1)} PDF(s) attempted.")
            else: st.warning("No PDFs were processed in this batch.")
    with col2_s1:
        if st.session_state.s1_results:
            st.subheader("Batch Conversion Results")
            for idx, result in enumerate(st.session_state.s1_results):
                with st.expander(f"Results for: {result['input_filename']} (Status: {result['status']})", expanded=(result['status']=='failure')):
                    if result['status'] == 'success':
                        if result['conversion_type'] == "Images":
                            images_output, img_opts, base_fn = result['output'], result['img_format_options'], os.path.splitext(result['input_filename'])[0]
                            st.write(result['messages'][0] if result['messages'] else f"{len(images_output)} image(s).")
                            zip_buffer = create_zip_from_images(images_output, base_fn, img_opts['img_format'])
                            st.download_button(f"⬇️ Download Images (.zip)", zip_buffer, f"{base_fn}_images.zip", "application/zip", key=f"s1_dl_zip_{idx}_{base_fn}", use_container_width=True)
                            for i, img_res in enumerate(images_output):
                                if i < 2: st.image(img_res, caption=f"Page {i+1}", width=200)
                                elif i == 2: st.write(f"(+ {len(images_output) - 2} more in ZIP)"); break
                        elif result['conversion_type'] == "Word Document (.docx)":
                            st.write(result['messages'][0] if result['messages'] else "Word doc ready.")
                            st.download_button("⬇️ Download Word (.docx)", result['output'], f"{os.path.splitext(result['input_filename'])[0]}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"s1_dl_docx_{idx}", use_container_width=True)
                    else: 
                        for msg in result['messages']: st.error(msg)
        elif uploaded_pdf_files_s1: st.info("Select options & click 'Convert All PDFs'.")


# --- UI Section 2: Combine PDF and/or Image Files (Consolidated) ---
st.divider()
st.header("2. Combine PDF and/or Image Files into a Single PDF")
uploaded_mixed_files_s2 = st.file_uploader(
    "Upload PDF and/or Image files (PNG, JPG, TIFF, etc.)",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    key="mixed_uploader_s2" # Unique key for this uploader
)

if uploaded_mixed_files_s2:
    current_file_ids_s2 = sorted([f.file_id for f in uploaded_mixed_files_s2]) # Use file_id for better uniqueness
    if st.session_state.s2_last_uploaded_file_ids != current_file_ids_s2:
        st.session_state.s2_ordered_items = []
        for f_obj in uploaded_mixed_files_s2:
            file_type = 'pdf' if f_obj.type == "application/pdf" else 'image'
            st.session_state.s2_ordered_items.append({'file': f_obj, 'type': file_type, 'id': f_obj.file_id, 'name': f_obj.name})
        st.session_state.s2_last_uploaded_file_ids = current_file_ids_s2
        st.session_state.s2_process_done = False # Reset results if files change
        st.session_state.s2_final_pdf_bytes = None
else: # Clear if no files are uploaded
    if st.session_state.s2_ordered_items or st.session_state.s2_last_uploaded_file_ids:
        st.session_state.s2_ordered_items = []
        st.session_state.s2_last_uploaded_file_ids = []
        st.session_state.s2_process_done = False
        st.session_state.s2_final_pdf_bytes = None

if st.session_state.s2_ordered_items:
    st.subheader("Order Files for Combined PDF")
    st.caption("Current order of all uploaded items. Use buttons to reorder.")

    for i, item_info in enumerate(st.session_state.s2_ordered_items):
        file_obj = item_info['file']
        item_type = item_info['type']
        item_id = item_info['id'] 
        item_name = item_info['name']

        cols_s2_order = st.columns([0.08, 0.12, 0.6, 0.1, 0.1]) 
        with cols_s2_order[0]: st.write(f"{i+1}.")
        with cols_s2_order[1]:
            if item_type == 'image':
                st.image(file_obj.getvalue(), width=50, caption="Img")
                file_obj.seek(0) 
            else: st.markdown(f"📄 **PDF**", help=item_name)
        with cols_s2_order[2]: st.write(item_name)
        with cols_s2_order[3]: 
            if i > 0: 
                if st.button("🔼", key=f"s2_up_{item_id}_{i}", help="Move Up"):
                    st.session_state.s2_ordered_items.insert(i-1, st.session_state.s2_ordered_items.pop(i))
                    st.rerun()
        with cols_s2_order[4]: 
            if i < len(st.session_state.s2_ordered_items) - 1: 
                if st.button("🔽", key=f"s2_down_{item_id}_{i}", help="Move Down"):
                    st.session_state.s2_ordered_items.insert(i+1, st.session_state.s2_ordered_items.pop(i))
                    st.rerun()
    st.markdown("---")

    perform_ocr_s2 = st.checkbox("Make final PDF searchable (perform OCR)?", value=False, key="ocr_check_s2",
                                 help="Processes the final combined PDF. Can be very slow. Needs Tesseract & Ghostscript.")
    ocr_lang_code_s2 = "eng" 
    if perform_ocr_s2:
        ocr_lang_opts_s2 = {"English":"eng", "Spanish":"spa", "French":"fra", "German":"deu", "Other (Manual)":"manual"} # Added manual option
        sel_lang_name_s2 = st.selectbox("OCR Language (Combined PDF):", list(ocr_lang_opts_s2.keys()), 0, key="ocr_lang_s2")
        ocr_lang_code_s2 = ocr_lang_opts_s2[sel_lang_name_s2]
        if ocr_lang_code_s2 == "manual": 
            ocr_lang_code_s2 = st.text_input("Tesseract lang code (e.g., 'pol'):", "eng", key="ocr_manual_s2").lower().strip()
        st.caption(f"Using OCR language: `{ocr_lang_code_s2}` for the combined PDF.")

    button_label_s2 = "🧩 Combine All Files to PDF"
    if perform_ocr_s2:
        button_label_s2 = "🧩 Combine & Make Searchable (OCR)"

    if st.button(button_label_s2, key="process_mixed_files_s2", use_container_width=True, disabled=(not st.session_state.s2_ordered_items)):
        st.session_state.s2_process_done = False
        st.session_state.s2_final_pdf_bytes = None
        st.session_state.s2_is_ocr_applied = perform_ocr_s2

        with st.spinner("Processing and combining files... This may take a while, especially with OCR."):
            pdf_merger_s2 = PdfWriter()
            any_page_added_s2 = False
            status_area_s2 = st.empty() 

            for idx, item_info_s2 in enumerate(st.session_state.s2_ordered_items):
                file_obj_s2 = item_info_s2['file']
                item_type_s2 = item_info_s2['type']
                item_name_s2 = item_info_s2['name']
                status_area_s2.text(f"Step 1: Processing item {idx+1}/{len(st.session_state.s2_ordered_items)}: '{item_name_s2}' ({item_type_s2})...")

                pdf_item_bytes_s2 = None
                if item_type_s2 == 'image':
                    pdf_item_bytes_s2 = image_to_single_page_pdf_bytes(file_obj_s2, filename_for_error=item_name_s2)
                elif item_type_s2 == 'pdf':
                    pdf_item_bytes_s2 = file_obj_s2.getvalue()
                    file_obj_s2.seek(0)
                
                if pdf_item_bytes_s2:
                    try:
                        reader_s2 = PdfReader(BytesIO(pdf_item_bytes_s2))
                        if reader_s2.is_encrypted:
                            st.warning(f"Skipping encrypted PDF: {item_name_s2}")
                            continue
                        for page_s2 in reader_s2.pages:
                            pdf_merger_s2.add_page(page_s2)
                            any_page_added_s2 = True
                    except Exception as e_read:
                        st.error(f"Error reading/adding pages from '{item_name_s2}': {e_read}")
                else:
                    st.warning(f"Could not convert '{item_name_s2}' ({item_type_s2}) to PDF page for merging.")
            
            status_area_s2.text("Step 2: Finalizing base combined PDF...")
            
            if not any_page_added_s2:
                st.error("No pages could be prepared for the final combined document.")
                final_combined_pdf_bytes_s2 = None
            else:
                merged_base_pdf_buffer_s2 = BytesIO()
                pdf_merger_s2.add_metadata({"/Producer": ""}) # Blank out producer
                pdf_merger_s2.write(merged_base_pdf_buffer_s2)
                final_combined_pdf_bytes_s2 = merged_base_pdf_buffer_s2.getvalue()

            if final_combined_pdf_bytes_s2 and perform_ocr_s2:
                status_area_s2.text(f"Step 3: Applying OCR (lang: {ocr_lang_code_s2})... This can be very slow.")
                if not ocr_lang_code_s2 and ocr_lang_opts_s2.get(sel_lang_name_s2) == "manual": # Check if manual input is empty
                     st.error("Manual OCR language code cannot be empty.")
                     final_combined_pdf_bytes_s2 = None 
                else:
                    final_combined_pdf_bytes_s2 = ocr_existing_pdf_st(final_combined_pdf_bytes_s2, language=ocr_lang_code_s2, context_section="S2-Combined")
            
            status_area_s2.empty() # Clear status messages

            if final_combined_pdf_bytes_s2:
                st.session_state.s2_final_pdf_bytes = final_combined_pdf_bytes_s2
                st.session_state.s2_process_done = True
                st.success("File combination and processing complete!")
            else:
                st.error("Final combined PDF generation failed.")

    if st.session_state.get('s2_process_done') and st.session_state.get('s2_final_pdf_bytes'):
        st.subheader("Final Combined PDF Result")
        time_str = time.strftime('%Y%m%d-%H%M%S')
        is_ocr_applied_s2 = st.session_state.get('s2_is_ocr_applied', False)
        ocr_tag_s2_final = f"_ocr_{ocr_lang_code_s2}" if is_ocr_applied_s2 else ""
        out_pdf_filename_s2 = f"combined_document{ocr_tag_s2_final}_{time_str}.pdf"
        download_label_s2 = f"⬇️ Download {'Searchable ' if is_ocr_applied_s2 else ''}Combined PDF"
        
        st.download_button(
            label=download_label_s2,
            data=st.session_state.s2_final_pdf_bytes,
            file_name=out_pdf_filename_s2,
            mime="application/pdf",
            key="download_combined_mixed_s2", # Reusing key from old S4 as it's now the primary combined download
            use_container_width=True
        )

# Final info message
if not uploaded_pdf_files_s1 and not st.session_state.s2_ordered_items:
    st.info("☝️ Upload files in Section 1 (PDF Conversion) or Section 2 (Combine Files) to get started.")

