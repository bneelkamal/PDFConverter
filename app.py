import streamlit as st
import os
import tempfile
from io import BytesIO
import zipfile
from PIL import Image # Pillow is used for Image-to-PDF
from pdf2image import convert_from_bytes # Used for PDF-to-Image
from pdf2docx import Converter # Used for PDF-to-Word
import base64
import time # For unique filenames if needed

# --- Configuration & Page Setup ---
st.set_page_config(page_title="File Converter Hub", layout="wide")
st.title("📄 File Converter Hub 🔄")
st.write("Convert PDFs to Images/Word, or combine Images into a PDF.")

# --- Helper Functions (Adapted for Streamlit) ---

# PDF-to-Image (Cached)
@st.cache_data(show_spinner=False)
def pdf_to_images_st(pdf_bytes, dpi, img_format, poppler_path=None):
    """Converts PDF bytes to a list of PIL Image objects."""
    try:
        images = convert_from_bytes(pdf_bytes, dpi=dpi, fmt=img_format.lower(), poppler_path=poppler_path)
        return images
    except Exception as e:
        st.error(f"Error during PDF to Image conversion: {e}")
        if "poppler" in str(e).lower():
            st.info("Ensure Poppler is installed and accessible. Check the sidebar for details.")
        return None

# PDF-to-Word (Cached)
@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    """Converts PDF bytes to Word (.docx) bytes."""
    temp_pdf_file = None
    temp_pdf_path = None # Initialize path variable
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(pdf_bytes)
            temp_pdf_path = temp_pdf_file.name

        output_docx_buffer = BytesIO()
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as temp_docx_file:
            temp_docx_path = temp_docx_file.name
            cv = Converter(temp_pdf_path) # Use path from the first tempfile
            cv.convert(temp_docx_path, start=0, end=None)
            cv.close()
            with open(temp_docx_path, 'rb') as f_docx:
                output_docx_buffer.write(f_docx.read())

        output_docx_buffer.seek(0)
        return output_docx_buffer

    except Exception as e:
        st.error(f"Error during PDF to Word conversion: {e}")
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
            # Ensure image is in a compatible format for saving
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'P':
                 img = img.convert('RGB') # Or handle palette properly if needed
            img.save(img_byte_arr, format=img_format.upper())
            img_byte_arr = img_byte_arr.getvalue()
            zip_file.writestr(img_filename, img_byte_arr)
    zip_buffer.seek(0)
    return zip_buffer

# NEW: Images-to-PDF (Cached)
@st.cache_data(show_spinner=False)
def images_to_pdf_st(image_bytes_list, filenames):
    """Converts a list of image bytes into a single PDF bytes buffer."""
    pil_images = []
    for img_bytes, filename in zip(image_bytes_list, filenames):
        try:
            img = Image.open(BytesIO(img_bytes))
            # Convert to RGB if necessary (common requirement for PDF saving)
            if img.mode == 'RGBA':
                img = img.convert('RGB')
            elif img.mode == 'P': # Handle palette mode
                 img = img.convert('RGB')

            pil_images.append(img)
        except Exception as e:
            st.error(f"Error opening or converting image '{filename}': {e}")
            return None # Abort if any image fails

    if not pil_images:
        st.warning("No valid images found to convert.")
        return None

    # Save images to a PDF in memory
    pdf_buffer = BytesIO()
    try:
        pil_images[0].save(
            pdf_buffer,
            format='PDF',
            save_all=True, # Important: Tells Pillow to save all images
            append_images=pil_images[1:] # Append the rest of the images
        )
        pdf_buffer.seek(0)
        return pdf_buffer
    except Exception as e:
        st.error(f"Error saving images to PDF: {e}")
        return None

# --- Sidebar for Poppler Info ---
st.sidebar.title("⚠️ Important Note")
st.sidebar.info(
    """
    **PDF to Image conversion requires Poppler.**

    - **If running locally:** Ensure Poppler is installed and added to your PATH.
      Download for Windows: [Poppler Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases/)
    - **If deploying:** Include Poppler in your environment (e.g., add `poppler-utils` to `packages.txt` for Streamlit Cloud).
    """
)

# --- Initialize Session State ---
# Use unique keys for results from different conversion types
if 'pdf_to_img_done' not in st.session_state: st.session_state['pdf_to_img_done'] = False
if 'pdf_to_word_done' not in st.session_state: st.session_state['pdf_to_word_done'] = False
if 'img_to_pdf_done' not in st.session_state: st.session_state['img_to_pdf_done'] = False

if 'image_results' not in st.session_state: st.session_state['image_results'] = None
if 'word_result_bytes' not in st.session_state: st.session_state['word_result_bytes'] = None
if 'pdf_result_bytes' not in st.session_state: st.session_state['pdf_result_bytes'] = None

# --- UI Section 1: PDF Conversions ---
st.header("1. Convert PDF")
uploaded_pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"], key="pdf_uploader")

if uploaded_pdf_file is not None:
    pdf_bytes_in = uploaded_pdf_file.getvalue()
    pdf_basename = os.path.splitext(uploaded_pdf_file.name)[0]

    st.markdown("---")
    col1_pdf, col2_pdf = st.columns([1, 2])

    with col1_pdf:
        st.subheader("PDF Conversion Options")
        pdf_conversion_type = st.radio(
            "Convert PDF To:",
            ("Images", "Word Document"),
            key="pdf_conversion_type",
            horizontal=True,
            on_change=lambda: st.session_state.update(pdf_to_img_done=False, pdf_to_word_done=False, image_results=None, word_result_bytes=None)
        )

        pdf_options_dict = {}
        if pdf_conversion_type == "Images":
            pdf_options_dict['img_format'] = st.selectbox("Image Format:", ["JPEG", "PNG"], key="img_format")
            pdf_options_dict['dpi'] = st.slider("Image Quality (DPI):", min_value=72, max_value=600, value=300, step=10, key="dpi")

        if st.button("🚀 Convert PDF", key="convert_pdf_button"):
            # Clear previous PDF conversion results
            st.session_state['pdf_to_img_done'] = False
            st.session_state['pdf_to_word_done'] = False
            st.session_state['image_results'] = None
            st.session_state['word_result_bytes'] = None

            with st.spinner("Processing PDF... Please wait."):
                if pdf_conversion_type == "Images":
                    images = pdf_to_images_st(pdf_bytes_in, pdf_options_dict['dpi'], pdf_options_dict['img_format'])
                    if images:
                        st.session_state['image_results'] = images
                        st.session_state['pdf_to_img_done'] = True
                        st.success("PDF successfully converted to images!")
                    else: st.error("PDF to Image conversion failed.")
                elif pdf_conversion_type == "Word Document":
                    docx_bytes_io = pdf_to_word_st(pdf_bytes_in)
                    if docx_bytes_io:
                        st.session_state['word_result_bytes'] = docx_bytes_io.getvalue()
                        st.session_state['pdf_to_word_done'] = True
                        st.success("PDF successfully converted to Word!")
                    else: st.error("PDF to Word conversion failed.")

    with col2_pdf:
        st.subheader("PDF Conversion Results")
        # Display PDF-to-Image Results
        if st.session_state.get('pdf_to_img_done') and st.session_state.get('image_results'):
            st.write(f"Generated {len(st.session_state['image_results'])} image(s):")
            img_format_used = pdf_options_dict.get('img_format', 'JPEG')
            zip_buffer = create_zip_from_images(st.session_state['image_results'], pdf_basename, img_format_used)
            st.download_button(
                label=f"⬇️ Download All Images (.zip)", data=zip_buffer,
                file_name=f"{pdf_basename}_images.zip", mime="application/zip", key="download_zip"
            )
            st.markdown("---")
            for i, img in enumerate(st.session_state['image_results']):
                if i < 3: st.image(img, caption=f"Page {i+1}", use_column_width=True)
                else:
                    st.write(f"(Plus {len(st.session_state['image_results']) - 3} more images in the ZIP file)")
                    break

        # Display PDF-to-Word Results
        elif st.session_state.get('pdf_to_word_done') and st.session_state.get('word_result_bytes'):
            st.write("Your Word document is ready:")
            st.download_button(
                label="⬇️ Download Word Document (.docx)", data=st.session_state['word_result_bytes'],
                file_name=f"{pdf_basename}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document", key="download_docx"
            )
        elif uploaded_pdf_file:
             st.info("Select conversion options and click 'Convert PDF'.")

# --- UI Section 2: Image to PDF Conversion ---
st.divider()
st.header("2. Convert Images to PDF")
uploaded_image_files = st.file_uploader(
    "Upload one or more image files (PNG, JPG, BMP, TIFF)",
    type=["png", "jpg", "jpeg", "bmp", "tiff"],
    accept_multiple_files=True,
    key="image_uploader"
)

if uploaded_image_files:
    # Sort images by filename for predictable order in PDF
    uploaded_image_files.sort(key=lambda f: f.name)

    st.write(f"{len(uploaded_image_files)} image(s) selected:")
    # Display small previews or just list names
    cols = st.columns(5)
    for i, img_file in enumerate(uploaded_image_files):
         if i < 10: # Show previews for first 10 images
              cols[i % 5].image(img_file, caption=img_file.name, width=100)
         elif i == 10:
              st.write(f"(Plus {len(uploaded_image_files)-10} more files...)")


    if st.button("🖼️ Convert Images to PDF", key="convert_images_button"):
        # Clear previous image-to-pdf results
        st.session_state['img_to_pdf_done'] = False
        st.session_state['pdf_result_bytes'] = None

        image_bytes_list = [img.getvalue() for img in uploaded_image_files]
        filenames = [img.name for img in uploaded_image_files]

        with st.spinner("Combining images into PDF..."):
            pdf_buffer = images_to_pdf_st(image_bytes_list, filenames)
            if pdf_buffer:
                st.session_state['pdf_result_bytes'] = pdf_buffer.getvalue()
                st.session_state['img_to_pdf_done'] = True
                st.success("Images successfully combined into a PDF!")
            else:
                st.error("Image to PDF conversion failed.")

    # Display Image-to-PDF results
    if st.session_state.get('img_to_pdf_done') and st.session_state.get('pdf_result_bytes'):
         st.subheader("Image Conversion Result")
         # Create a default filename
         out_pdf_filename = f"combined_images_{time.strftime('%Y%m%d_%H%M%S')}.pdf"
         st.download_button(
              label="⬇️ Download Combined PDF",
              data=st.session_state['pdf_result_bytes'],
              file_name=out_pdf_filename,
              mime="application/pdf",
              key="download_combined_pdf"
         )

elif not uploaded_pdf_file: # Only show this if no PDF is uploaded either
     st.info("☝️ Upload a PDF above OR images here to get started.")
