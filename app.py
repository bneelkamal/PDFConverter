import streamlit as st
import os
import tempfile
from io import BytesIO
import zipfile
from PIL import Image, UnidentifiedImageError
from pdf2image import convert_from_bytes
from pdf2image.exceptions import PDFPageCountError, PDFSyntaxError
from pdf2docx import Converter
import base64
import time
import traceback # Import traceback for detailed error logging

import pytesseract
from pypdf import PdfWriter, PdfReader, __version__ as pypdf_version
from pypdf.errors import PdfReadError, DependencyError as PyPDFDependencyError # Import specific pypdf errors
import ocrmypdf
from ocrmypdf.exceptions import (
    MissingDependencyError as OCRmyPDFMissingDependencyError,
    EncryptedPdfError as OCRmyPDFEncryptedPdfError,
    PriorOcrFoundError as OCRmyPDFPriorOcrFoundError,
    # Removed OcrmypdfError and TesseractError as they are not directly importable from exceptions
)

# --- Configuration & Page Setup ---
st.set_page_config(page_title="File Converter Hub", layout="wide", initial_sidebar_state="expanded")
st.title("📄 File Converter Hub 🔄")
st.write("Convert PDFs to other formats (Sec 1), or Combine various files into a single PDF (Sec 2 - images OCR'd by default, with optional password protection).")

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def pdf_to_images_st(pdf_bytes, dpi, img_format, poppler_path=None):
    """Converts PDF bytes to a list of PIL Image objects."""
    try:
        # Ensure poppler_path is None if not explicitly set, relying on PATH
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=img_format.lower(), poppler_path=poppler_path)
        return images
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        st.error(f"Image too large: {e}. The PDF page dimensions at the selected DPI are too big. Try reducing the DPI setting.")
        raise
    except PDFPageCountError:
        st.error("Could not get page count from PDF. It might be corrupted or password-protected without user password.")
        raise
    except PDFSyntaxError:
        st.error("PDF syntax error. The PDF might be corrupted or not a valid PDF.")
        raise
    except Exception as e:
        # Catch specific errors related to poppler executable not found
        if "poppler" in str(e).lower() or "pdftoppm" in str(e).lower():
            st.info("PDF to Image: Ensure Poppler is set up (PATH/packages.txt).")
        # Log other exceptions for debugging
        traceback.print_exc()
        raise

@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    """Converts PDF bytes to Word (.docx) bytes."""
    temp_pdf_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(pdf_bytes)
            temp_pdf_path = temp_pdf_file.name
        output_docx_buffer = BytesIO()
        # pdf2docx requires file paths, so we use temp files
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as temp_docx_file:
            temp_docx_path = temp_docx_file.name
            cv = Converter(temp_pdf_path)
            cv.convert(temp_docx_path)
            cv.close()
            with open(temp_docx_path, 'rb') as f_docx:
                output_docx_buffer.write(f_docx.read())
        output_docx_buffer.seek(0)
        return output_docx_buffer
    except Exception as e:
        # Catch specific errors related to tesseract executable not found by pdf2docx
        if "tesseract" in str(e).lower() or "No such file or directory" in str(e):
             st.info("PDF to Word (Scanned): Ensure Tesseract is set up for pdf2docx OCR (PATH/packages.txt).")
        # Log other exceptions for debugging
        traceback.print_exc()
        raise
    finally:
        # Clean up the temporary PDF file
        if temp_pdf_path and os.path.exists(temp_pdf_path):
            try:
                os.remove(temp_pdf_path)
            except Exception:
                # Log or handle cleanup error if necessary
                st.warning(f"Could not remove temp PDF: {temp_pdf_path}")


def create_zip_from_images(images, base_filename, img_format):
    """Creates a ZIP archive containing image files in memory, for a single PDF source."""
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, img in enumerate(images):
            # Create image filename with zero-padding for sorting
            img_filename = f"{base_filename}_page_{str(i + 1).zfill(len(str(len(images))))}.{img_format.lower()}"
            img_byte_arr = BytesIO()
            save_img = img
            # Convert image mode if necessary for the target format
            if img.mode == 'RGBA' and img_format.lower() == 'jpeg':
                save_img = img.convert('RGB')
            elif img.mode == 'P': # Convert palette images
                save_img = img.convert('RGB')
            try:
                save_img.save(img_byte_arr, format=img_format.upper())
                img_byte_arr = img_byte_arr.getvalue()
                zip_file.writestr(img_filename, img_byte_arr)
            except Exception as e:
                st.warning(f"Could not save image page {i+1} to ZIP: {e}")
                traceback.print_exc() # Log error
                # Continue with the next image
    zip_buffer.seek(0)
    return zip_buffer

def create_master_zip_from_s1_results(s1_results_list, img_format):
    """Creates a single ZIP archive from all successfully converted images in s1_results."""
    master_zip_buffer = BytesIO()
    with zipfile.ZipFile(master_zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file_master:
        total_images_added = 0
        for result_entry in s1_results_list:
            # Only process successful image conversions
            if result_entry['status'] == 'success' and result_entry['conversion_type'] == 'Images':
                images = result_entry['output']
                original_pdf_basename = os.path.splitext(result_entry['input_filename'])[0]
                num_digits = len(str(len(images))) # For zero-padding

                for i, img in enumerate(images):
                    page_num_str = str(i + 1).zfill(num_digits)
                    # Ensure unique filenames in the master zip if multiple PDFs have similar names
                    img_filename_in_zip = f"{original_pdf_basename}_page_{page_num_str}.{img_format.lower()}"

                    img_byte_arr = BytesIO()
                    save_img = img
                    # Convert image mode if necessary
                    if img.mode == 'RGBA' and img_format.lower() == 'jpeg':
                        save_img = img.convert('RGB')
                    elif img.mode == 'P':
                        save_img = img.convert('RGB')

                    try:
                        save_img.save(img_byte_arr, format=img_format.upper())
                        img_data_bytes = img_byte_arr.getvalue()
                        zip_file_master.writestr(img_filename_in_zip, img_data_bytes)
                        total_images_added +=1
                    except Exception as e:
                        st.warning(f"Could not add image '{img_filename_in_zip}' to master ZIP: {e}")
                        traceback.print_exc() # Log error
                        # Continue with the next image

        # Check if any images were added before returning the buffer
        if total_images_added == 0:
            return None
    master_zip_buffer.seek(0)
    return master_zip_buffer

@st.cache_data(show_spinner=False)
def ocr_existing_pdf_st(input_pdf_bytes, language='eng', deskew=True, force_ocr=True, context_section="Combined"):
    """Performs OCR on an existing PDF's bytes using ocrmypdf."""
    if not input_pdf_bytes: return None
    stxt = st.empty() # Use st.empty() for dynamic status updates
    stxt.text(f"{context_section}-Performing OCR on PDF (lang: {language}). This may take considerable time...")
    output_pdf_buffer = BytesIO()
    try:
        # ocrmypdf writes directly to the output buffer
        ocrmypdf.ocr( BytesIO(input_pdf_bytes), output_pdf_buffer, language=language, deskew=deskew, force_ocr=force_ocr, skip_text=False, progress_bar=False)
        output_pdf_buffer.seek(0) # Rewind buffer to the beginning
        stxt.success(f"{context_section}-OCR processing on PDF complete!"); time.sleep(2); stxt.empty() # Clear status after success
        return output_pdf_buffer.getvalue() # Return bytes
    except OCRmyPDFMissingDependencyError as e:
        st.error(f"{context_section}-ocrmypdf: Missing system dependency: {e}. Ensure Tesseract and Ghostscript are installed and in PATH (or packages.txt for deployment)."); stxt.empty(); traceback.print_exc(); return None
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        st.error(f"{context_section}-ocrmypdf: Image too large within PDF: {e}. The PDF contains an image that exceeds pixel limits.");
        stxt.empty(); traceback.print_exc(); return None
    except OCRmyPDFEncryptedPdfError:
        st.error(f"{context_section}-ocrmypdf: The PDF is encrypted and cannot be processed without decryption first."); stxt.empty(); traceback.print_exc(); return None
    except OCRmyPDFPriorOcrFoundError: # This can happen if force_ocr=False and text is found
        st.warning(f"{context_section}-ocrmypdf: PDF already has OCR. 'force_ocr=True' (default) re-processes. If 'force_ocr=False' was used, existing text layer is kept.");
        output_pdf_buffer.seek(0)
        stxt.success(f"{context_section}-OCR processing (prior OCR found)."); time.sleep(2); stxt.empty()
        # Return the content of the buffer if anything was written, otherwise return original bytes
        return output_pdf_buffer.getvalue() if output_pdf_buffer.getbuffer().nbytes > 0 else input_pdf_bytes
    # Removed explicit catch for OcrmypdfError and TesseractError
    except Exception as e: # General fallback for other unexpected issues, including OcrmypdfError and TesseractError
        st.error(f"{context_section}-ocrmypdf: An unexpected error occurred during OCR: {e}"); stxt.empty(); traceback.print_exc(); return None

# MODIFIED Function: Image to OCR'd single-page PDF using Pytesseract
@st.cache_data(show_spinner=False)
def image_to_ocr_pdf_page_bytes(image_file_object, ocr_language='eng', filename_for_error="image"):
    """Converts an image UploadedFile object to bytes of a single-page OCR'd PDF using Pytesseract."""
    try:
        image_file_object.seek(0) # Ensure file pointer is at the beginning
        img_pil = Image.open(image_file_object)
        # Convert image mode if necessary for Pytesseract
        if img_pil.mode == 'RGBA' or img_pil.mode == 'P':
            img_pil = img_pil.convert('RGB')

        # Use Pytesseract to get PDF bytes with OCR layer
        # Pytesseract's image_to_pdf_or_hocr with extension='pdf' returns bytes directly
        pdf_bytes = pytesseract.image_to_pdf_or_hocr(img_pil, lang=ocr_language, extension='pdf')
        return pdf_bytes # This is already bytes
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        st.error(f"Image too large ('{filename_for_error}'): {e}. The image dimensions exceed pixel limits. Please use a smaller image.")
        traceback.print_exc() # Log error
        return None
    except UnidentifiedImageError:
        st.error(f"Cannot identify image file ('{filename_for_error}'). It might be corrupted or an unsupported format.")
        traceback.print_exc() # Log error
        return None
    except pytesseract.TesseractNotFoundError:
        st.error(f"Pytesseract: Tesseract OCR engine not found for image '{filename_for_error}'. Check Tesseract installation and PATH (or packages.txt).")
        traceback.print_exc() # Log error
        return None
    except pytesseract.TesseractError as te: # Catch other Tesseract errors
        st.error(f"Pytesseract: Tesseract error processing image '{filename_for_error}': {te}")
        traceback.print_exc() # Log error
        return None
    except Exception as e:
        st.error(f"Error converting image '{filename_for_error}' to OCR'd PDF page: {e}")
        traceback.print_exc() # Log error
        return None

# --- Sidebar ---
st.sidebar.title("File Converter App")
st.sidebar.markdown("---")
st.sidebar.header("All-in-One PDF & Image Hub")
st.sidebar.info(
    f"""
    - **Sec 1:** PDF to Image/Word
    - **Sec 2:** Combine PDF/Images. Images are OCR'd by default during their initial conversion to PDF pages. Optionally, the final combined PDF can be further processed with OCR and password protected.

    **Important Setup for Deployed Apps (e.g., Streamlit Cloud):**
    - **`requirements.txt` should include:** `streamlit`, `Pillow`, `pdf2image`, `pdf2docx`, `pytesseract`, `pypdf`, `ocrmypdf`
    - **`packages.txt` should include:** `tesseract-ocr`, `tesseract-ocr-eng` (and other language packs like `tesseract-ocr-deu`), `ghostscript`, `poppler-utils`

    Failure to include these system packages in `packages.txt` will lead to errors.
    (pypdf: {pypdf_version})
    """
)
st.sidebar.markdown("---")
st.sidebar.caption(f"Refreshed: {time.strftime('%Y%m%d-%H%M%S')}")

# --- Initialize Session State ---
# Section 1 (PDF Conversions)
if 's1_results' not in st.session_state:
    st.session_state.s1_results = []
if 's1_conversion_type' not in st.session_state:
    st.session_state.s1_conversion_type = "Images"
if 's1_img_format' not in st.session_state:
    st.session_state.s1_img_format = "PNG"
if 's1_master_zip_bytes' not in st.session_state: # Initialize master zip bytes state
    st.session_state.s1_master_zip_bytes = None


# Section 2 (Consolidated Combine Files)
if 's2_ordered_items' not in st.session_state:
    st.session_state.s2_ordered_items = []
if 's2_last_uploaded_file_ids' not in st.session_state:
    st.session_state.s2_last_uploaded_file_ids = []
if 's2_final_pdf_bytes' not in st.session_state:
    st.session_state.s2_final_pdf_bytes = None
if 's2_process_done' not in st.session_state:
    st.session_state.s2_process_done = False
if 's2_is_final_ocr_applied' not in st.session_state: # Track if final pass OCR was applied
    st.session_state.s2_is_final_ocr_applied = False
if 's2_ocr_lang_code' not in st.session_state: # To store selected OCR lang for S2
    st.session_state.s2_ocr_lang_code = "eng"
if 's2_apply_password' not in st.session_state: # State for password protection checkbox
    st.session_state.s2_apply_password = False
if 's2_password' not in st.session_state: # State for password input
    st.session_state.s2_password = ""
if 's2_confirm_password' not in st.session_state: # State for confirm password input
    st.session_state.s2_confirm_password = ""


# --- UI Section 1: PDF Conversions ---
st.header("1. Convert PDF(s) to Other Formats")
uploaded_pdf_files_s1 = st.file_uploader(
    "Upload one or more PDF files",
    type=["pdf"],
    key="pdf_uploader_s1",
    accept_multiple_files=True
)
if uploaded_pdf_files_s1:
    st.markdown("---"); col1_s1, col2_s1 = st.columns([1, 2])
    with col1_s1:
        st.subheader("Conversion Options")
        # Radio button for conversion type, resets results on change
        st.session_state.s1_conversion_type = st.radio(
            "Convert All Uploaded PDFs To:",
            ("Images", "Word Document (.docx)"),
            key="conversion_type_s1_radio",
            index=0 if st.session_state.s1_conversion_type == "Images" else 1,
            horizontal=True,
            on_change=lambda: st.session_state.update(s1_results=[], s1_master_zip_bytes=None) # Also clear master zip state
        )

        pdf_options_s1 = {}
        if st.session_state.s1_conversion_type == "Images":
            # Image format selection
            st.session_state.s1_img_format = st.selectbox(
                "Image Format:",
                ["PNG", "JPEG"],
                key="img_format_s1_select",
                index=0 if st.session_state.s1_img_format == "PNG" else 1
                )
            pdf_options_s1['img_format'] = st.session_state.s1_img_format
            # DPI slider for image quality - MAX VALUE CHANGED TO 300
            pdf_options_s1['dpi'] = st.slider("Image Quality (DPI):", 72, 300, 200, 10, key="dpi_s1", help="Lower DPI for very large PDFs if you encounter 'Image too large' errors.")

        # Convert button for Section 1
        if st.button("🚀 Convert All PDFs", key="convert_button_s1", use_container_width=True):
            st.session_state.s1_results = [] # Clear previous results
            st.session_state.s1_master_zip_bytes = None # Clear previous master zip
            with st.spinner(f"Processing {len(uploaded_pdf_files_s1)} PDF(s)..."):
                for file_obj in uploaded_pdf_files_s1:
                    # Prepare a result entry for each file
                    result_entry = {
                        'input_filename': file_obj.name,
                        'conversion_type': st.session_state.s1_conversion_type,
                        'status': 'failure', # Default status
                        'output': None, # Store output data (images list or docx bytes)
                        'messages': [] # Collect messages (success or error)
                        }
                    try:
                        pdf_bytes_in = file_obj.getvalue(); file_obj.seek(0) # Read bytes and reset pointer
                        if st.session_state.s1_conversion_type == "Images":
                            result_entry['img_format_options'] = pdf_options_s1.copy() # Store options used for this conversion
                            images = pdf_to_images_st(pdf_bytes_in, pdf_options_s1['dpi'], pdf_options_s1['img_format'])
                            if images: result_entry.update({'output': images, 'status': 'success', 'messages': [f"{len(images)} images generated."]})
                            else: result_entry['messages'].append("Image conversion returned no images.")
                        elif st.session_state.s1_conversion_type == "Word Document (.docx)":
                            docx_bytes_io = pdf_to_word_st(pdf_bytes_in)
                            # pdf_to_word_st returns BytesIO object, get bytes value
                            if docx_bytes_io: result_entry.update({'output': docx_bytes_io.getvalue(), 'status': 'success', 'messages': ["Word document generated."]})
                            else: result_entry['messages'].append("Word conversion returned no data.")
                    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as e_bomb:
                        result_entry['messages'].append(f"Error for '{file_obj.name}': Image too large (Decompression Bomb). Try lower DPI. Details: {e_bomb}")
                        traceback.print_exc() # Log error
                    except Exception as e:
                        # Catch any other unexpected errors during conversion
                        result_entry['messages'].append(f"Error processing '{file_obj.name}': {str(e)}")
                        traceback.print_exc() # Log error
                    st.session_state.s1_results.append(result_entry) # Add result to session state list
            # Provide a summary message after the batch
            if st.session_state.s1_results: st.success(f"Batch conversion of {len(uploaded_pdf_files_s1)} PDF(s) attempted.")
            else: st.warning("No PDFs were processed in this batch.")

    with col2_s1:
        # Display results for each file
        if st.session_state.s1_results:
            st.subheader("Batch Conversion Results")
            successful_image_conversions = 0 # Counter for master zip availability
            for idx, result in enumerate(st.session_state.s1_results):
                # Use an expander to show results for each file, expanded if there was a failure
                with st.expander(f"Results for: {result['input_filename']} (Status: {result['status']})", expanded=(result['status']=='failure')):
                    if result['status'] == 'success':
                        if result['conversion_type'] == "Images":
                            successful_image_conversions += 1
                            images_output = result['output']
                            img_opts_exp = result['img_format_options']
                            base_fn_exp = os.path.splitext(result['input_filename'])[0]
                            st.write(result['messages'][0] if result['messages'] else f"{len(images_output)} image(s).")

                            # --- Modified logic for single vs multiple image downloads ---
                            if len(uploaded_pdf_files_s1) == 1:
                                # If only one file was uploaded, offer individual image downloads and a ZIP
                                st.info("Single PDF converted to images. Download individually or as a ZIP.")
                                for i, img_res in enumerate(images_output):
                                     img_filename_ind = f"{base_fn_exp}_page_{str(i + 1).zfill(len(str(len(images_output))))}.{img_opts_exp['img_format'].lower()}"
                                     img_byte_arr_ind = BytesIO()
                                     # Ensure image is in a suitable mode for saving
                                     save_img_ind = img_res
                                     if img_res.mode == 'RGBA' and img_opts_exp['img_format'].lower() == 'jpeg':
                                         save_img_ind = img_res.convert('RGB')
                                     elif img_res.mode == 'P':
                                         save_img_ind = img_res.convert('RGB')
                                     save_img_ind.save(img_byte_arr_ind, format=img_opts_exp['img_format'].upper())
                                     st.download_button(f"⬇️ Download Page {i+1} ({img_opts_exp['img_format']})", img_byte_arr_ind.getvalue(), img_filename_ind, f"image/{img_opts_exp['img_format'].lower()}", key=f"s1_dl_img_ind_{idx}_{i}", use_container_width=True)
                                     if i < 2: st.image(img_res, caption=f"Page {i+1}", width=200) # Show previews
                                     elif i == 2: st.write(f"(+ {len(images_output) - 2} more pages)"); break # Limit previews

                                st.markdown("---") # Separator before the ZIP option for single file
                                # Still offer the ZIP download for the single file as an alternative
                                zip_buffer_individual = create_zip_from_images(images_output, base_fn_exp, img_opts_exp['img_format'])
                                st.download_button(f"⬇️ Download All Images from This PDF (.zip)", zip_buffer_individual, f"{base_fn_exp}_images.zip", "application/zip", key=f"s1_dl_zip_ind_{idx}_{base_fn_exp}_single", use_container_width=True)

                            else:
                                # If multiple files were uploaded, offer a ZIP download for images from this specific PDF
                                st.write("Multiple PDFs converted. Download images from this PDF as a ZIP.")
                                zip_buffer_individual = create_zip_from_images(images_output, base_fn_exp, img_opts_exp['img_format'])
                                st.download_button(f"⬇️ Download These Images (.zip)", zip_buffer_individual, f"{base_fn_exp}_images.zip", "application/zip", key=f"s1_dl_zip_ind_{idx}_{base_fn_exp}", use_container_width=True)
                                # Show previews of the first few images
                                for i, img_res in enumerate(images_output):
                                    if i < 2: st.image(img_res, caption=f"Page {i+1}", width=200)
                                    elif i == 2: st.write(f"(+ {len(images_output) - 2} more in this ZIP)"); break # Limit previews
                            # --- End of modified logic ---

                        elif result['conversion_type'] == "Word Document (.docx)":
                            st.write(result['messages'][0] if result['messages'] else "Word doc ready.")
                            # Offer download for the Word document
                            st.download_button("⬇️ Download Word (.docx)", result['output'], f"{os.path.splitext(result['input_filename'])[0]}.docx", "application/vnd.openxmlformats-officedocument.wordprocessingml.document", key=f"s1_dl_docx_{idx}", use_container_width=True)
                    else:
                        # Display error messages for failed conversions
                        for msg in result['messages']: st.error(msg)

            # Offer a master ZIP download for all successfully converted images (still applies to multiple files)
            if st.session_state.s1_conversion_type == "Images" and successful_image_conversions > 0 and len(uploaded_pdf_files_s1) > 1:
                st.markdown("---")
                st.subheader("Download All Images Together")
                # Button to trigger preparation of the master zip
                if st.button("📦 Prepare Master ZIP of All Images", key="prepare_master_zip_s1", use_container_width=True):
                    with st.spinner("Creating master ZIP file..."):
                        master_zip_buffer = create_master_zip_from_s1_results(
                            st.session_state.s1_results,
                            st.session_state.s1_img_format
                            )
                        if master_zip_buffer:
                            st.session_state.s1_master_zip_bytes = master_zip_buffer.getvalue() # Store bytes in session state
                        else:
                            st.session_state.s1_master_zip_bytes = None
                            st.warning("No images were successfully converted to include in a master ZIP.")

                # Display the download button for the master zip if prepared
                if 's1_master_zip_bytes' in st.session_state and st.session_state.s1_master_zip_bytes:
                    time_str_zip = time.strftime('%Y%m%d-%H%M%S')
                    st.download_button(
                        label="⬇️ Download All Images from All PDFs (.zip)",
                        data=st.session_state.s1_master_zip_bytes,
                        file_name=f"all_converted_images_{time_str_zip}.zip",
                        mime="application/zip",
                        key="download_master_zip_s1_final",
                        use_container_width=True
                    )

        # Initial message when files are uploaded but not yet converted
        elif uploaded_pdf_files_s1: st.info("Select options & click 'Convert All PDFs'.")


# --- UI Section 2: Combine PDF and/or Image Files (Consolidated) ---
st.divider()
st.header("2. Combine PDF and/or Image Files into a Single PDF")
st.caption("Images uploaded in this section will be OCR'd by default during their initial conversion to PDF pages. Optionally, the final combined PDF can be further processed with OCR and password protected.")

# File uploader for mixed file types
uploaded_mixed_files_s2 = st.file_uploader(
    "Upload PDF and/or Image files (PNG, JPG, TIFF, etc.)",
    type=["pdf", "png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    key="mixed_uploader_s2"
)

# Logic to initialize or clear s2_ordered_items based on uploaded files
if uploaded_mixed_files_s2:
    # Create a unique identifier for the current set of uploaded files
    current_file_ids_s2 = sorted([f.file_id for f in uploaded_mixed_files_s2])
    # Check if the uploaded files have changed since the last rerun
    if st.session_state.s2_last_uploaded_file_ids != current_file_ids_s2:
        # If files changed, re-initialize the ordered items list
        st.session_state.s2_ordered_items = []
        for f_obj in uploaded_mixed_files_s2:
            file_type = 'pdf' if f_obj.type == "application/pdf" else 'image'
            st.session_state.s2_ordered_items.append({'file': f_obj, 'type': file_type, 'id': f_obj.file_id, 'name': f_obj.name})
        # Update the last uploaded file IDs
        st.session_state.s2_last_uploaded_file_ids = current_file_ids_s2
        # Reset process status and final PDF bytes
        st.session_state.s2_process_done = False
        st.session_state.s2_final_pdf_bytes = None
        st.session_state.s2_is_final_ocr_applied = False # Reset final OCR status
        st.session_state.s2_apply_password = False # Reset password option
        st.session_state.s2_password = "" # Clear password fields
        st.session_state.s2_confirm_password = ""
else:
    # If no files are uploaded, clear related session state
    if st.session_state.s2_ordered_items or st.session_state.s2_last_uploaded_file_ids:
        st.session_state.s2_ordered_items = []
        st.session_state.s2_last_uploaded_file_ids = []
        st.session_state.s2_process_done = False
        st.session_state.s2_final_pdf_bytes = None
        st.session_state.s2_is_final_ocr_applied = False # Reset final OCR status
        st.session_state.s2_apply_password = False # Reset password option
        st.session_state.s2_password = "" # Clear password fields
        st.session_state.s2_confirm_password = ""


# Display and reorder files if any are uploaded
if st.session_state.s2_ordered_items:
    st.subheader("Order Files for Combined PDF")
    st.caption("Current order of all uploaded items. Use buttons to reorder.")

    # Loop through the ordered items and display them with reorder buttons
    for i, item_info in enumerate(st.session_state.s2_ordered_items):
        file_obj = item_info['file']
        item_type = item_info['type']
        item_id = item_info['id']
        item_name = item_info['name']

        # Use columns for layout: index, preview/icon, filename, up button, down button
        cols_s2_order = st.columns([0.08, 0.12, 0.6, 0.1, 0.1])
        with cols_s2_order[0]: st.write(f"{i+1}.")
        with cols_s2_order[1]:
            if item_type == 'image':
                # Display a small image preview
                current_pos = file_obj.tell() # Remember current position
                st.image(file_obj.getvalue(), width=50, caption="Img")
                file_obj.seek(current_pos) # Reset file pointer after reading
            else: st.markdown(f"📄 **PDF**", help=item_name) # Display PDF icon
        with cols_s2_order[2]: st.write(item_name)
        with cols_s2_order[3]:
            # Up button (disabled for the first item)
            if i > 0:
                if st.button("🔼", key=f"s2_up_{item_id}_{i}", help="Move Up"):
                    # Swap item with the one above it and rerun
                    st.session_state.s2_ordered_items.insert(i-1, st.session_state.s2_ordered_items.pop(i))
                    st.rerun()
        with cols_s2_order[4]:
            # Down button (disabled for the last item)
            if i < len(st.session_state.s2_ordered_items) - 1:
                if st.button("🔽", key=f"s2_down_{item_id}_{i}", help="Move Down"):
                    # Swap item with the one below it and rerun
                    st.session_state.s2_ordered_items.insert(i+1, st.session_state.s2_ordered_items.pop(i))
                    st.rerun()
    st.markdown("---")

    # OCR Language selection - now always visible if there are images, for the initial Pytesseract OCR
    # And also used if final OCR pass with OCRmyPDF is selected.
    has_images_s2 = any(item['type'] == 'image' for item in st.session_state.s2_ordered_items)
    # Dictionary of common OCR language options and their Tesseract codes
    ocr_lang_opts_s2 = {"English":"eng", "Spanish":"spa", "French":"fra", "German":"deu", "Other (Manual)":"manual"}

    # Determine the currently selected language key for the selectbox
    current_lang_key = "English" # Default
    if st.session_state.s2_ocr_lang_code in ocr_lang_opts_s2.values():
         # Find the key for the stored code
         current_lang_key = list(ocr_lang_opts_s2.keys())[list(ocr_lang_opts_s2.values()).index(st.session_state.s2_ocr_lang_code)]
    elif st.session_state.s2_ocr_lang_code and st.session_state.s2_ocr_lang_code not in ocr_lang_opts_s2.values():
         # If it's a manual code not in the standard list, select "Other (Manual)"
         current_lang_key = "Other (Manual)"


    # Show language selection if images are present for their default OCR OR if final OCR is selected
    if has_images_s2 or st.session_state.get('ocr_check_s2_final_pass', False):
        st.session_state.s2_ocr_lang_code_select = st.selectbox(
            "OCR Language (for images & optional final pass):",
            list(ocr_lang_opts_s2.keys()),
            index=list(ocr_lang_opts_s2.keys()).index(current_lang_key), # Set index based on current_lang_key
            key="ocr_lang_s2_select"
            )

        # Update the stored language code based on selectbox value
        selected_ocr_lang_value_s2 = ocr_lang_opts_s2.get(st.session_state.s2_ocr_lang_code_select)

        if selected_ocr_lang_value_s2 == "manual":
            # Show text input for manual code if "Other (Manual)" is selected
            # Use the stored code as default if it exists and wasn't a standard one
            manual_default = st.session_state.s2_ocr_lang_code if st.session_state.s2_ocr_lang_code not in ocr_lang_opts_s2.values() else "eng"
            st.session_state.s2_ocr_lang_code = st.text_input("Tesseract lang code (e.g., 'pol'):", value=manual_default, key="ocr_manual_s2").lower().strip()
            if not st.session_state.s2_ocr_lang_code: # Prevent empty manual code
                 st.warning("Manual OCR language code cannot be empty. Using 'eng' as fallback.")
                 st.session_state.s2_ocr_lang_code = "eng"
        else:
            # Store the selected standard language code
            st.session_state.s2_ocr_lang_code = selected_ocr_lang_value_s2

        st.caption(f"Using OCR language: `{st.session_state.s2_ocr_lang_code}` for images. This language will also be used if final OCR pass is selected.")
    else:
         # If no images and final OCR not selected, ensure a default is set if needed later
         if 's2_ocr_lang_code' not in st.session_state: st.session_state.s2_ocr_lang_code = "eng"


    # Checkbox for the FINAL OCR pass on the combined document
    perform_final_ocr_s2 = st.checkbox("Make final combined PDF searchable (perform OCR with OCRmyPDF)?", value=st.session_state.get('ocr_check_s2_final_pass', False), key="ocr_check_s2_final_pass",
                                     help="Processes the entire combined PDF. Can be very slow. Needs Tesseract & Ghostscript.")

    st.markdown("---") # Separator before password options

    # --- Password Protection Options ---
    st.session_state.s2_apply_password = st.checkbox("Password protect the combined PDF?", value=st.session_state.s2_apply_password, key="apply_password_s2")

    if st.session_state.s2_apply_password:
        col_pass1, col_pass2 = st.columns(2)
        with col_pass1:
            st.session_state.s2_password = st.text_input("Enter Password:", type="password", key="password_s2")
        with col_pass2:
             st.session_state.s2_confirm_password = st.text_input("Confirm Password:", type="password", key="confirm_password_s2")

        # Basic password validation
        if st.session_state.s2_password != st.session_state.s2_confirm_password and st.session_state.s2_confirm_password != "":
            st.error("Passwords do not match!")
            password_match = False
        elif st.session_state.s2_apply_password and st.session_state.s2_password == "":
             st.warning("Password protection is checked, but no password is set.")
             password_match = False # Treat as not matching if empty when required
        else:
            password_match = True
    else:
        password_match = True # No password required, so it "matches"

    st.markdown("---") # Separator after password options


    button_label_s2 = "🧩 Combine All Files to PDF"
    if perform_final_ocr_s2: # Button label reflects the final pass
        button_label_s2 = "🧩 Combine & Make Searchable (OCRmyPDF)"
    if st.session_state.s2_apply_password: # Add password info to button label
         button_label_s2 += " & Add Password"


    # Process button - disabled if passwords don't match and password protection is on
    if st.button(button_label_s2, key="process_mixed_files_s2", use_container_width=True, disabled=(not st.session_state.s2_ordered_items or (st.session_state.s2_apply_password and not password_match))):
        st.session_state.s2_process_done = False # Reset process status
        st.session_state.s2_final_pdf_bytes = None # Clear previous result
        st.session_state.s2_is_final_ocr_applied = perform_final_ocr_s2 # Track if final pass was chosen

        # Re-validate password just before processing
        if st.session_state.s2_apply_password:
            if st.session_state.s2_password == "" or st.session_state.s2_password != st.session_state.s2_confirm_password:
                 st.error("Password error: Passwords do not match or are empty.")
                 # Stop processing
                 st.session_state.s2_process_done = False
                 st.session_state.s2_final_pdf_bytes = None
                 st.session_state.s2_is_final_ocr_applied = False
                 st.rerun() # Rerun to show error and stop spinner

        with st.spinner("Processing and combining files... This may take a while, especially if final OCR is selected."):
            pdf_merger_s2 = PdfWriter()
            any_page_added_s2 = False # Flag to check if any pages were successfully added
            status_area_s2 = st.empty() # Placeholder for status messages

            # Step 1: Convert images to OCR'd PDF pages (default) and collect all PDF bytes
            intermediate_pdf_pages_bytes = []
            for idx, item_info_s2 in enumerate(st.session_state.s2_ordered_items):
                file_obj_s2 = item_info_s2['file']
                item_type_s2 = item_info_s2['type']
                item_name_s2 = item_info_s2['name']
                status_area_s2.text(f"Preparing item {idx+1}/{len(st.session_state.s2_ordered_items)}: '{item_name_s2}' ({item_type_s2})...")

                pdf_item_bytes_s2 = None
                if item_type_s2 == 'image':
                    file_obj_s2.seek(0) # Ensure file pointer is at the start
                    # Use the selected language for initial image OCR (Pytesseract)
                    current_ocr_lang = st.session_state.s2_ocr_lang_code
                    if not current_ocr_lang: # Check for empty language code
                         st.error(f"OCR language code for image '{item_name_s2}' is empty. Please select or enter a language.")
                         continue # Skip this item if language is missing
                    pdf_item_bytes_s2 = image_to_ocr_pdf_page_bytes(file_obj_s2, ocr_language=current_ocr_lang, filename_for_error=item_name_s2)
                elif item_type_s2 == 'pdf':
                    file_obj_s2.seek(0)
                    pdf_item_bytes_s2 = file_obj_s2.getvalue()
                    file_obj_s2.seek(0) # Reset pointer after reading

                if pdf_item_bytes_s2:
                    intermediate_pdf_pages_bytes.append({'name': item_name_s2, 'bytes': pdf_item_bytes_s2})
                else:
                    st.warning(f"Could not process '{item_name_s2}' ({item_type_s2}) for merging.")

            # Step 2: Merge all prepared PDF pages
            status_area_s2.text("Merging all processed items...")
            if not intermediate_pdf_pages_bytes:
                st.error("No items could be prepared for merging.")
                final_combined_pdf_bytes_s2 = None
            else:
                for item_data in intermediate_pdf_pages_bytes:
                    try:
                        reader_s2 = PdfReader(BytesIO(item_data['bytes']))
                        if reader_s2.is_encrypted:
                            st.warning(f"Skipping encrypted PDF content from: {item_data['name']}")
                            continue
                        # Add all pages from the current item to the merger
                        for page_s2 in reader_s2.pages:
                            pdf_merger_s2.add_page(page_s2)
                            any_page_added_s2 = True # Mark that at least one page was added
                    except PdfReadError as e_read_pdf:
                        st.error(f"Error reading PDF content from '{item_data['name']}': {e_read_pdf}. The file might be corrupted.")
                        traceback.print_exc() # Log error
                    except Exception as e_read:
                        st.error(f"Error reading/adding pages from '{item_data['name']}': {e_read}")
                        traceback.print_exc() # Log error


                if not any_page_added_s2:
                    st.error("No pages could be added to the final combined document.")
                    final_combined_pdf_bytes_s2 = None
                else:
                    # Write the merged PDF to a BytesIO buffer
                    merged_base_pdf_buffer_s2 = BytesIO()
                    # Optional: Remove default producer metadata
                    pdf_merger_s2.add_metadata({"/Producer": ""})
                    pdf_merger_s2.write(merged_base_pdf_buffer_s2)
                    final_combined_pdf_bytes_s2 = merged_base_pdf_buffer_s2.getvalue() # Get bytes

            # Step 3: Optional final OCR pass on the entire merged document using OCRmyPDF
            if final_combined_pdf_bytes_s2 and perform_final_ocr_s2:
                status_area_s2.text(f"Applying final OCR pass (lang: {st.session_state.s2_ocr_lang_code})... This can be very slow.")
                # Use the selected language code for the final OCR pass
                final_ocr_lang = st.session_state.s2_ocr_lang_code
                if not final_ocr_lang: # Check for empty language code
                     st.error("OCR language for final pass is not set or is empty.")
                     final_combined_pdf_bytes_s2 = None # Prevent proceeding if language is missing
                else:
                    # Call the ocr_existing_pdf_st function
                    final_combined_pdf_bytes_s2 = ocr_existing_pdf_st(final_combined_pdf_bytes_s2, language=final_ocr_lang, context_section="S2-FinalPass")

            # Step 4: Apply Password Protection if selected
            if final_combined_pdf_bytes_s2 and st.session_state.s2_apply_password and st.session_state.s2_password:
                 status_area_s2.text("Applying password protection...")
                 try:
                     # Create a new PdfReader from the processed bytes
                     reader_to_encrypt = PdfReader(BytesIO(final_combined_pdf_bytes_s2))
                     writer_encrypted = PdfWriter()

                     # Add all pages from the reader to the new writer
                     for page in reader_to_encrypt.pages:
                         writer_encrypted.add_page(page)

                     # Apply encryption
                     writer_encrypted.encrypt(st.session_state.s2_password)

                     # Write the encrypted PDF to a new buffer
                     encrypted_pdf_buffer = BytesIO()
                     writer_encrypted.write(encrypted_pdf_buffer)
                     final_combined_pdf_bytes_s2 = encrypted_pdf_buffer.getvalue() # Update the final bytes

                     status_area_s2.success("Password protection applied!"); time.sleep(2); status_area_s2.empty()
                 except Exception as e:
                     st.error(f"Error applying password protection: {e}")
                     traceback.print_exc() # Log error
                     final_combined_pdf_bytes_s2 = None # Indicate failure


            status_area_s2.empty() # Clear the final status text area

            # Store the final result and update process status
            if final_combined_pdf_bytes_s2:
                st.session_state.s2_final_pdf_bytes = final_combined_pdf_bytes_s2
                st.session_state.s2_process_done = True
                st.success("File combination and processing complete!")
            else:
                st.error("Final combined PDF generation failed.")

    # Display download button if the process is done and result is available
    if st.session_state.get('s2_process_done') and st.session_state.get('s2_final_pdf_bytes'):
        st.subheader("Final Combined PDF Result")
        time_str = time.strftime('%Y%m%d-%H%M%S')
        is_final_ocr_applied_s2 = st.session_state.get('s2_is_final_ocr_applied', False)
        is_password_applied_s2 = st.session_state.get('s2_apply_password', False)

        # Determine the appropriate tags for the filename based on OCR and password status
        ocr_tag = f"_ocr_{st.session_state.s2_ocr_lang_code}" if is_final_ocr_applied_s2 else "_img_ocr_default"
        password_tag = "_password" if is_password_applied_s2 else ""

        out_pdf_filename_s2 = f"combined_document{ocr_tag}{password_tag}_{time_str}.pdf"

        # Construct the download button label based on applied options
        download_label_parts = ["⬇️ Download"]
        if is_final_ocr_applied_s2:
            download_label_parts.append("Searchable (Final Pass OCR)")
        else:
             download_label_parts.append("Combined (Images OCRd)")
        if is_password_applied_s2:
            download_label_parts.append("Password Protected")
        download_label_parts.append("PDF")

        download_label_s2 = " ".join(download_label_parts)


        st.download_button(
            label=download_label_s2,
            data=st.session_state.s2_final_pdf_bytes,
            file_name=out_pdf_filename_s2,
            mime="application/pdf",
            key="download_combined_mixed_s2",
            use_container_width=True
        )

# Final info message when no files are uploaded in either section
if not uploaded_pdf_files_s1 and not st.session_state.s2_ordered_items:
    st.info("☝️ Upload files in Section 1 (PDF Conversion) or Section 2 (Combine Files) to get started.")
