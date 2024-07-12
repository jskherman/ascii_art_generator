import streamlit as st
from ascii_art_generator import image_to_ascii

st.set_page_config(
    page_title="ASCII Art Generator",
    page_icon="🎨",
    layout="wide",
)

# Load CSS styles
if "css" not in st.session_state:
    st.session_state["css"] = open("style.css").read()
st.markdown(f"<style>{st.session_state['css']}\n</style>", unsafe_allow_html=True)

st.header("ASCII Art Generator")

st.subheader("Step 1: Upload an Image to Convert to ASCII Art")

acol1, acol2 = st.columns([4, 5])

with acol1:
    image_file = st.file_uploader("Choose image file:")

with acol2:
    if image_file is not None:
        # To read file as bytes:
        bytes_data = image_file.read()

        # Write image to disk
        with open("input_image.jpg", "wb") as f:
            f.write(bytes_data)

        with st.expander("Preview Input Image", expanded=True):
            st.image(
                bytes_data,
                caption=f'Uploaded Image:  "{image_file.name}"',
                use_column_width=True,
            )

st.write("")

st.subheader("Step 2: Define Generator Parameters")

bcol1, bcol2 = st.columns(2)

with bcol1:
    max_width = st.slider(
        "Choose the maximum width of the ASCII art:",
        help="The higher the value, the more detailed and faithful the ASCII art will be to the original image.",
        min_value=32,
        max_value=1024,
        value=512,
        key="max_width",
        step=8,
    )

    color = st.color_picker(
        label="Choose the color of the text:",
        value="#F2613F",
        key="color",
        help="This will be the accent color of the characters in the generated ASCII art.",
    )

with bcol2:
    edge_threshold = st.slider(
        "Choose the edge threshold, any arbitrary value:",
        help="As the threshold increases, hard edges in the ASCII art will become less pronounced.",
        min_value=0,
        max_value=255,
        value=100,
        key="edge_threshold",
        step=5,
    )

    image_type = st.selectbox(
        "Choose whether to use PNG or SVG for the output image:",
        options=["PNG", "SVG"],
        index=0,
        key="image_type",
        help="PNG is a raster image format, while SVG is a vector image format.",
    )

st.write("")
generate_button = st.button(
    "Generate ASCII Art!", type="primary", disabled=image_file is None
)

st.divider()

st.subheader("Output")

if image_file is not None:
    if generate_button:
        image_to_ascii(
            image_file,
            "output.txt",
            edge_threshold=edge_threshold,
            max_width=max_width,
            color_output=f"output.{image_type.lower()}",
            color_overlay=color,
        )

        ocol1, ocol2, ocol3 = st.columns([1, 1, 4], vertical_alignment="center")

        with ocol3:
            st.success("ASCII art generated successfully! 🎉")

        with open("output.txt", "r", encoding="utf-8") as f:
            ascii_art = f.read()

            with ocol1:
                download_button = st.download_button(
                    label="Download ASCII Art",
                    data=ascii_art,
                    file_name="output.txt",
                    mime="text/plain",
                )

            with ocol2:
                download_image_button = st.download_button(
                    label="Download Image",
                    data=open(f"output.{image_type.lower()}", "rb").read(),
                    file_name=f"output.{image_type.lower()}",
                    mime=("image/png" if image_type == "PNG" else "image/svg+xml"),
                )

            with st.expander("Preview ASCII Art Image", expanded=True):
                st.image(
                    "output.png",
                    caption=f'[{image_type}] ASCII Art Image of "{image_file.name}"',
                    use_column_width=True,
                )

            with st.expander("Preview ASCII Art Text", expanded=False):
                st.code(ascii_art, language="")
    else:
        st.warning("Click the button above to generate ASCII art.")
else:
    st.warning("Please upload an image first to generate ASCII art.")
