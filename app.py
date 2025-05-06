import streamlit as st
import os
import tempfile
from io import BytesIO
import zipfile
from PIL import Image
from pdf2image import convert_from_bytes, pdfinfo_from_bytes
from pdf2docx import Converter
import base64 # To embed images for display if needed, though st.image works directly

# --- Configuration & Page Setup ---
st.set_page_config(page_title="PDF Converter", layout="wide")
st.title("📄 PDF Converter App")
st.write("Upload your PDF and choose to convert it to images or a Word document.")

# --- Helper Functions (Adapted for Streamlit) ---

# Cache the image conversion to avoid recomputing if the same file and settings are used
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

# Cache the Word conversion
@st.cache_data(show_spinner=False)
def pdf_to_word_st(pdf_bytes):
    """Converts PDF bytes to Word (.docx) bytes."""
    temp_pdf_file = None # Initialize
    try:
        # pdf2docx requires a file path, so save bytes temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf_file:
            temp_pdf_file.write(pdf_bytes)
            temp_pdf_path = temp_pdf_file.name # Get the path

        output_docx_buffer = BytesIO() # Create an in-memory buffer for the output

        # Perform conversion
        cv = Converter(temp_pdf_path)
        # Convert directly into the buffer (pdf2docx doesn't support this directly)
        # So, we save to a temp docx file and read it back. Less ideal but necessary.
        with tempfile.NamedTemporaryFile(delete=True, suffix=".docx") as temp_docx_file:
             temp_docx_path = temp_docx_file.name
             cv.convert(temp_docx_path, start=0, end=None)
             cv.close()
             # Read the converted docx back into our buffer
             with open(temp_docx_path, 'rb') as f_docx:
                  output_docx_buffer.write(f_docx.read())

        output_docx_buffer.seek(0) # Reset buffer position to the beginning
        return output_docx_buffer

    except Exception as e:
        st.error(f"Error during PDF to Word conversion: {e}")
        return None
    finally:
        # Clean up the temporary PDF file explicitly if it was created
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

            # Save image to an in-memory buffer
            img_byte_arr = BytesIO()
            img.save(img_byte_arr, format=img_format.upper())
            img_byte_arr = img_byte_arr.getvalue() # Get bytes

            # Write image bytes to zip file
            zip_file.writestr(img_filename, img_byte_arr)

    zip_buffer.seek(0)
    return zip_buffer

# --- Sidebar for Poppler Info ---
st.sidebar.title("⚠️ Important Note")
st.sidebar.info(
    """
    **PDF to Image conversion requires Poppler.**

    - **If running locally:** Ensure Poppler is installed on your system and added to your PATH environment variable.
      Download for Windows: [Poppler Windows Releases](https://github.com/oschwartz10612/poppler-windows/releases/)
    - **If deploying (e.g., Streamlit Cloud):** You need to include Poppler in your deployment environment. For Streamlit Cloud, add `poppler-utils` to your `packages.txt` file.
    """
)
# Optional: Add Poppler Path input if needed for local testing without PATH setup
# poppler_path_input = st.sidebar.text_input("Optional: Poppler 'bin' Path (if not in PATH)", key="poppler_path_ui")


# --- Main Application Area ---
uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])

# Initialize session state keys
if 'conversion_done' not in st.session_state:
    st.session_state['conversion_done'] = False
if 'image_results' not in st.session_state:
    st.session_state['image_results'] = None
if 'word_result_bytes' not in st.session_state:
    st.session_state['word_result_bytes'] = None


if uploaded_file is not None:
    # Read PDF bytes once
    pdf_bytes = uploaded_file.getvalue()
    base_filename = os.path.splitext(uploaded_file.name)[0]

    st.markdown("---")
    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Conversion Options")
        conversion_type = st.radio(
            "Select Conversion Type:",
            ("PDF to Images", "PDF to Word"),
            key="conversion_type",
            on_change=lambda: st.session_state.update(conversion_done=False, image_results=None, word_result_bytes=None) # Reset on type change
        )

        options_dict = {}
        if conversion_type == "PDF to Images":
            options_dict['img_format'] = st.selectbox("Image Format:", ["JPEG", "PNG"], key="img_format")
            options_dict['dpi'] = st.slider("Image Quality (DPI):", min_value=72, max_value=600, value=300, step=10, key="dpi")
            # Use Poppler path from sidebar input if provided
            # options_dict['poppler_path'] = poppler_path_input if poppler_path_input else None

        # Convert Button - place it after options
        if st.button("🚀 Convert PDF", key="convert_button"):
            # Clear previous results before starting new conversion
            st.session_state['conversion_done'] = False
            st.session_state['image_results'] = None
            st.session_state['word_result_bytes'] = None

            with st.spinner("Processing... Please wait."):
                if conversion_type == "PDF to Images":
                    images = pdf_to_images_st(pdf_bytes, options_dict['dpi'], options_dict['img_format']) # Pass poppler_path if using input
                    if images:
                        st.session_state['image_results'] = images
                        st.session_state['conversion_done'] = True
                        st.success("PDF successfully converted to images!")
                    else:
                         st.error("Image conversion failed.")

                elif conversion_type == "PDF to Word":
                    docx_bytes_io = pdf_to_word_st(pdf_bytes)
                    if docx_bytes_io:
                        st.session_state['word_result_bytes'] = docx_bytes_io.getvalue() # Store raw bytes
                        st.session_state['conversion_done'] = True
                        st.success("PDF successfully converted to Word!")
                    else:
                         st.error("Word conversion failed.")


    with col2:
        st.subheader("Results")
        if st.session_state['conversion_done']:
            # --- Display Image Results ---
            if st.session_state.get('image_results'):
                st.write(f"Generated {len(st.session_state['image_results'])} image(s):")

                # Prepare ZIP download for images
                zip_buffer = create_zip_from_images(
                    st.session_state['image_results'],
                    base_filename,
                    options_dict.get('img_format', 'JPEG') # Get format used
                 )
                st.download_button(
                    label="⬇️ Download All Images (.zip)",
                    data=zip_buffer,
                    file_name=f"{base_filename}_images.zip",
                    mime="application/zip",
                    key="download_zip"
                )
                st.markdown("---")
                # Display images (optional: limit display count for many pages)
                for i, img in enumerate(st.session_state['image_results']):
                     if i < 5: # Display first 5 images as preview
                         st.image(img, caption=f"Page {i+1}", use_column_width=True)
                     else:
                          st.write(f"(Plus {len(st.session_state['image_results']) - 5} more images in the ZIP file)")
                          break # Stop displaying more images

            # --- Display Word Result ---
            if st.session_state.get('word_result_bytes'):
                st.write("Your Word document is ready:")
                st.download_button(
                    label="⬇️ Download Word Document (.docx)",
                    data=st.session_state['word_result_bytes'],
                    file_name=f"{base_filename}.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="download_docx"
                )
        elif uploaded_file: # Show only if file is uploaded but conversion not done/failed
             st.info("Click the 'Convert PDF' button after selecting options.")

else:
    st.info("☝️ Upload a PDF file to get started.")