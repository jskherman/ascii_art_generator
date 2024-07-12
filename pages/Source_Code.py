import streamlit as st

# Load CSS styles
if "css" not in st.session_state:
    st.session_state["css"] = open("style.css").read()
st.markdown(f"<style>{st.session_state['css']}\n</style>", unsafe_allow_html=True)

with st.expander("ASCII Art Generator Code", expanded=True):
    with open("ascii_art_generator.py", "r", encoding="utf-8") as f:
        # code_file = st.text_area("Generator Code Editor", height=1024, value=f.read())
        st.code(f.read(), language="python")

download_file = st.download_button(
    label="Download ASCII Art Generator Script",
    data=open("ascii_art_generator.py", "rb").read(),
    file_name="ascii_art_generator.py",
)


# write_file = st.button("Save to File")

# if write_file:
#     with open("generator.py", "w", encoding="utf-8") as f:
#         f.write(code_file)
#     st.toast("File saved successfully!", icon="🎉")
