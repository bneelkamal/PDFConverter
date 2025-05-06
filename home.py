import streamlit as st
import os
import tempfile
from io import BytesIO
import zipfile
from PIL import Image # Pillow is used for Image-to-PDF
from pdf2image import convert_from_bytes # Used for PDF-to-Image
from pdf2docx import Converter # Used for PDF-to-Word
import base64 # Included from your original code
import time # For unique filenames if needed

import pytesseract # For OCR
from pypdf import PdfWriter, PdfReader # For merging OCR'd PDF pages

# --- Configuration & Page Setup ---
st.set_page_config(page_title="File Converter Hub", layout="wide", initial_sidebar_state="expanded")
st.title("📄 File Converter Hub 🔄")
st.write("Convert PDFs to Images/Word, or combine Images into a searchable PDF.")

# --- Helper Functions ---

@st.cache_data(show_spinner=False)
def pdf_to_images_st(pdf_bytes, dpi, img_format, poppler_path=None):
    """Converts PDF bytes to a list of PIL Image objects."""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=img_format.lower(), poppler_path=poppler_path)
        return images
    except Exception as e:
        st.error(f"Error during PDF to Image conversion: {e}")
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            st.info("Ensure Poppler is installed and its 'bin' directory is in your system's PATH. Check sidebar for details.")
        return None

@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    """Converts PDF bytes to Word (.docx) bytes."""
    temp_pdf_file = None
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(pdf_bytes)
            temp_pdf_path = temp_pdf_file.name

        output_docx_buffer = BytesIO()
        # pdf2docx needs a file path for both input and output for conversion
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as temp_docx_file: # Output path
            temp_docx_path = temp_docx_file.name
            cv = Converter(temp_pdf_path) # Input PDF path
            cv.convert(temp_docx_path, start=0, end=None) # Convert to output DOCX path
            cv.close()
            with open(temp_docx_path, 'rb') as f_docx:
                output_docx_buffer.write(f_docx.read())
        
        output_docx_buffer.seek(0)
        return output_docx_buffer
    except Exception as e:
        st.error(f"Error during PDF to Word conversion: {e}")
        if "tesseract" in str(e).lower() and "image-based" in str(e).lower(): # Example check
             st.info("The PDF might be image-based. For best results with scanned PDFs in PDF-to-Word, ensure Tesseract OCR is installed and in PATH, as pdf2docx might attempt to use it.")
        return None
    finally:
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception as clean_e:
                st.warning(f"Could not remove temporary PDF file: {clean_e}")

def create_zip_from_images(images, base_filename, img_format):
    """Creates a ZIP archive containing image files in memory."""
    zip_buffer = BytesIO()
    num_digits = len(str(len(images)))
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            page_num_str = str(i + 1).zfill(num_digits)
            img_filename = f"{base_filename}_page_{page_num_str}.{img_format.lower()}"
            img_byte_arr = BytesIO()
            
            save_img = img
            # Ensure image is in a compatible format for saving, especially for JPEG
            if img.mode == 'RGBA' and img_format.lower() == 'jpeg':
                save_img = img.convert('RGB')
            elif img.mode == 'P': # Handle palette mode by converting to RGB
                 save_img = img.convert('RGB')
            
            save_img.save(img_byte_arr, format=img_format.upper())
            img_byte_arr = img_byte_arr.getvalue()
            zip_file.writestr(img_filename, img_byte_arr)
    zip_buffer.seek(0)
    return zip_buffer

@st.cache_data(show_spinner=False)
def images_to_searchable_pdf_st(image_bytes_list, filenames, ocr_language='eng'):
    """Converts a list of image bytes into a single searchable PDF bytes buffer using OCR."""
    output_pdf_writer = PdfWriter()
    
    total_images = len(image_bytes_list)
    if total_images == 0:
        st.warning("No images provided for PDF conversion.")
        return None

    progress_bar_placeholder = st.empty()
    status_text_placeholder = st.empty()
    
    # Initialize progress bar
    progress_bar_placeholder.progress(0.0)

    for i, (img_bytes, filename) in enumerate(zip(image_bytes_list, filenames)):
        status_text_placeholder.text(f"Processing image {i+1}/{total_images} ('{filename}') with OCR (lang: {ocr_language})...")
        try:
            img_pil = Image.open(BytesIO(img_bytes))
            
            if img_pil.mode == 'RGBA' or img_pil.mode == 'P':
                img_pil = img_pil.convert('RGB')

            pdf_page_bytes = pytesseract.image_to_pdf_or_hocr(img_pil, lang=ocr_language, extension='pdf')
            
            ocr_page_pdf_reader = PdfReader(BytesIO(pdf_page_bytes))
            if len(ocr_page_pdf_reader.pages) > 0:
                output_pdf_writer.add_page(ocr_page_pdf_reader.pages[0])
            else:
                st.warning(f"OCR did not produce a readable PDF page for image '{filename}'. Skipping.")
                continue 

        except pytesseract.TesseractNotFoundError:
            st.error("Tesseract OCR engine not found. Please install Tesseract and ensure it's in your PATH.")
            st.info("See sidebar for Tesseract installation guidance and links.")
            status_text_placeholder.empty()
            progress_bar_placeholder.empty()
            return None
        except pytesseract.TesseractError as te:
            st.error(f"Tesseract OCR error on '{filename}': {str(te)}. "
                     f"Ensure language data for '{ocr_language}' (e.g., '{ocr_language}.traineddata') is installed in Tesseract's 'tessdata' directory.")
            st.info("See sidebar for Tesseract language data links.")
            status_text_placeholder.empty()
            progress_bar_placeholder.empty()
            return None
        except Exception as e:
            st.error(f"An unexpected error occurred while processing image '{filename}' with OCR: {e}")
            status_text_placeholder.empty()
            progress_bar_placeholder.empty()
            return None
        
        progress_bar_placeholder.progress(float(i + 1) / total_images)

    status_text_placeholder.text("Finalizing PDF...")
    if not output_pdf_writer.pages:
        st.warning("No images were successfully processed into PDF pages.")
        status_text_placeholder.empty()
        progress_bar_placeholder.empty()
        return None

    final_pdf_buffer = BytesIO()
    output_pdf_writer.write(final_pdf_buffer)
    final_pdf_buffer.seek(0)
    
    status_text_placeholder.success("Searchable PDF created successfully!")
    time.sleep(2) # Keep success message visible for a moment
    status_text_placeholder.empty()
    progress_bar_placeholder.empty()
    return final_pdf_buffer

# --- Sidebar for Dependency Info ---
st.sidebar.title("⚠️ Important Notes & Setup")
st.sidebar.markdown("---")
st.sidebar.header("Poppler (for PDF to Image)")
st.sidebar.info(
    """
    PDF to Image conversion requires Poppler.
    - **Local Machine:** Ensure Poppler is installed and its `bin` directory is added to your system's PATH.
      - Windows: [Poppler Windows Releases by oSchwartz](https://github.com/oschwartz10612/poppler-windows/releases/) (Extract and add the `poppler-ver-xxx/bin` to PATH).
      - Linux: `sudo apt-get install poppler-utils`
      - macOS: `brew install poppler`
    - **Deployment (e.g., Streamlit Cloud):** Include Poppler in your environment (e.g., add `poppler-utils` to `packages.txt`).
    """
)
st.sidebar.markdown("---")
st.sidebar.header("Tesseract OCR (for Image to Searchable PDF)")
st.sidebar.info(
    """
    Image to Searchable PDF conversion uses Tesseract OCR.
    - **Installation:** Ensure Tesseract OCR engine (v4.0 or higher recommended) is installed.
      - Official Guide: [Tesseract Installation](https://tesseract-ocr.github.io/tessdoc/Installation.html)
    - **PATH:** Add Tesseract's installation directory to your system's PATH environment variable.
    - **Language Data:** Download and place language data files (e.g., `eng.traineddata`) into Tesseract's `tessdata` sub-directory (usually in the Tesseract installation folder).
      - Language Data: [tessdata_fast (recommended)](https://github.com/tesseract-ocr/tessdata_fast) or [tessdata_best](https://github.com/tesseract-ocr/tessdata_best).
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Last Refreshed: {time.strftime('%Y-%m-%d %H:%M:%S')}")


# --- Initialize Session State ---
if 'pdf_to_img_done' not in st.session_state: st.session_state['pdf_to_img_done'] = False
if 'pdf_to_word_done' not in st.session_state: st.session_state['pdf_to_word_done'] = False
if 'img_to_pdf_done' not in st.session_state: st.session_state['img_to_pdf_done'] = False

if 'image_results' not in st.session_state: st.session_state['image_results'] = None
if 'word_result_bytes' not in st.session_state: st.session_state['word_result_bytes'] = None
if 'pdf_result_bytes' not in st.session_state: st.session_state['pdf_result_bytes'] = None

# --- UI Section 1: PDF Conversions ---
st.header("1. Convert PDF")
uploaded_pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader_key")

if uploaded_pdf_file is not None:
    pdf_bytes_in = uploaded_pdf_file.getvalue()
    pdf_basename = os.path.splitext(uploaded_pdf_file.name)[0]

    st.markdown("---")
    col1_pdf, col2_pdf = st.columns([1, 2]) # Adjusted column ratio for better layout

    with col1_pdf:
        st.subheader("Conversion Options")
        pdf_conversion_type = st.radio(
            "Convert PDF To:",
            ("Images", "Word Document (.docx)"),
            key="pdf_conversion_type_key",
            horizontal=True,
            on_change=lambda: st.session_state.update(pdf_to_img_done=False, pdf_to_word_done=False, image_results=None, word_result_bytes=None)
        )

        pdf_options_dict = {}
        if pdf_conversion_type == "Images":
            pdf_options_dict['img_format'] = st.selectbox("Image Format:", ["PNG", "JPEG"], key="img_format_pdf_to_img_key")
            pdf_options_dict['dpi'] = st.slider("Image Quality (DPI):", min_value=72, max_value=600, value=200, step=10, key="dpi_pdf_to_img_key")

        if st.button("🚀 Convert PDF", key="convert_pdf_button_key", use_container_width=True):
            st.session_state.update(pdf_to_img_done=False, pdf_to_word_done=False, image_results=None, word_result_bytes=None) # Reset specific states
            with st.spinner("Processing PDF... Please wait."):
                if pdf_conversion_type == "Images":
                    images = pdf_to_images_st(pdf_bytes_in, pdf_options_dict['dpi'], pdf_options_dict['img_format'])
                    if images:
                        st.session_state['image_results'] = images
                        st.session_state['pdf_to_img_done'] = True
                        st.success("PDF successfully converted to images!")
                    else: st.error("PDF to Image conversion failed. Check Poppler setup in sidebar.")
                elif pdf_conversion_type == "Word Document (.docx)":
                    docx_bytes_io = pdf_to_word_st(pdf_bytes_in)
                    if docx_bytes_io:
                        st.session_state['word_result_bytes'] = docx_bytes_io.getvalue()
                        st.session_state['pdf_to_word_done'] = True
                        st.success("PDF successfully converted to Word!")
                    else: st.error("PDF to Word conversion failed.")
    with col2_pdf:
        st.subheader("Results")
        if st.session_state.get('pdf_to_img_done') and st.session_state.get('image_results'):
            st.write(f"Generated {len(st.session_state['image_results'])} image(s):")
            img_format_used = pdf_options_dict.get('img_format', 'PNG') 
            zip_buffer = create_zip_from_images(st.session_state['image_results'], pdf_basename, img_format_used)
            st.download_button(
                label=f"⬇️ Download All Images (.zip)", data=zip_buffer,
                file_name=f"{pdf_basename}_images.zip", mime="application/zip", key="download_zip_images_key", use_container_width=True
            )
            st.markdown("---")
            # Display up to 3 image previews
            for i, img_res in enumerate(st.session_state['image_results']):
                if i < 3: 
                    st.image(img_res, caption=f"Page {i+1} Preview", use_column_width=True)
                elif i == 3:
                    st.write(f"(Plus {len(st.session_state['image_results']) - 3} more images in the ZIP file)")
                    break
        elif st.session_state.get('pdf_to_word_done') and st.session_state.get('word_result_bytes'):
            st.write("Your Word document is ready:")
            st.download_button(
                label="⬇️ Download Word Document (.docx)", data=st.session_state['word_result_bytes'],
                file_name=f"{pdf_basename}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", 
                key="download_docx_file_key", use_container_width=True
            )
        elif uploaded_pdf_file: # If file is uploaded but no action taken yet
             st.info("Select conversion options from the left and click 'Convert PDF'.")

# --- UI Section 2: Image to PDF Conversion ---
st.divider()
st.header("2. Combine Images to Searchable PDF (with OCR)")
uploaded_image_files = st.file_uploader(
    "Upload one or more image files (PNG, JPG, JPEG, BMP, TIFF)",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    key="image_uploader_for_pdf_key"
)

if uploaded_image_files:
    uploaded_image_files.sort(key=lambda f: f.name)

    st.write(f"{len(uploaded_image_files)} image(s) selected:")
    # Preview a few images
    cols_img_preview = st.columns(min(len(uploaded_image_files), 5)) # Show up to 5 previews
    for i, img_file_preview in enumerate(uploaded_image_files):
        if i < 5:
            cols_img_preview[i].image(img_file_preview, caption=img_file_preview.name, width=100)
        elif i == 5:
            st.write(f"(Plus {len(uploaded_image_files)-5} more files not previewed...)")
            break
    
    st.markdown("---")
    ocr_lang_options = {
        "English": "eng", "Spanish": "spa", "French": "fra", "German": "deu", 
        "Italian": "ita", "Portuguese": "por", "Dutch": "nld",
        "Chinese (Simplified)": "chi_sim", "Chinese (Traditional)": "chi_tra",
        "Japanese": "jpn", "Korean": "kor", "Hindi": "hin", "Arabic": "ara",
        "Russian": "rus", "Other (Manual Input)": "manual"
    }
    selected_lang_name = st.selectbox(
        "Select OCR Language:", 
        options=list(ocr_lang_options.keys()), 
        index=0, 
        key="ocr_language_select_key",
        help="Ensure you have the corresponding Tesseract language data file (e.g., 'eng.traineddata') installed."
    )
    
    ocr_lang_code = ocr_lang_options[selected_lang_name]
    if ocr_lang_code == "manual":
        ocr_lang_code = st.text_input("Enter Tesseract language code (e.g., 'pol' for Polish):", value="eng", key="ocr_manual_lang_key").lower().strip()
    
    st.caption(f"Using Tesseract language code: `{ocr_lang_code}`. Please verify language data installation (see sidebar).")

    if st.button("🖼️ Combine Images to Searchable PDF", key="convert_images_to_pdf_button_key", use_container_width=True):
        # Reset relevant session state for this section
        st.session_state['img_to_pdf_done'] = False
        st.session_state['pdf_result_bytes'] = None # Reset previous combined PDF result

        image_bytes_list = [img_f.getvalue() for img_f in uploaded_image_files]
        filenames = [img_f.name for img_f in uploaded_image_files]
        
        if not ocr_lang_code:
            st.error("OCR language code cannot be empty if 'Other (Manual Input)' is selected.")
        else:
            # The function images_to_searchable_pdf_st now handles its own spinner/progress.
            pdf_buffer = images_to_searchable_pdf_st(image_bytes_list, filenames, ocr_language=ocr_lang_code)
            
            if pdf_buffer:
                st.session_state['pdf_result_bytes'] = pdf_buffer.getvalue()
                st.session_state['img_to_pdf_done'] = True
                # Success message is now part of the function images_to_searchable_pdf_st
            else:
                # Error messages are also handled inside the function, this is a fallback.
                st.error("Image to Searchable PDF conversion failed. Check messages above and Tesseract setup in sidebar.")

    if st.session_state.get('img_to_pdf_done') and st.session_state.get('pdf_result_bytes'):
        st.subheader("Searchable PDF Result")
        out_pdf_filename = f"ocr_combined_images_{ocr_lang_code}_{time.strftime('%Y%m%d-%H%M%S')}.pdf"
        st.download_button(
            label="⬇️ Download Searchable PDF",
            data=st.session_state['pdf_result_bytes'],
            file_name=out_pdf_filename,
            mime="application/pdf",
            key="download_searchable_pdf_button_key",
            use_container_width=True
        )

# Initial message if no files are uploaded in any section
if not uploaded_pdf_file and not uploaded_image_files:
    st.info("☝️ Upload a PDF in Section 1 OR images in Section 2 to get started.")
