import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader


# To hide the sidebar
st.set_page_config(initial_sidebar_state="collapsed")

st.markdown(
    """
<style>
    [data-testid="collapsedControl"] {
        display: none
    }
</style>
""",
    unsafe_allow_html=True,
)

# Set the background image
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://d2yjegym0lbr1w.cloudfront.net/thumbs/ctscan_1280.jpg?v=20161004"); # https://images.unsplash.com/photo-1542281286-9e0a16bb7366
    background-size: 100vw 100vh;  # This sets the size to cover 100% of the viewport width and height
    background-position: center;  
    background-repeat: no-repeat;
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)

# Reduce padding
st.markdown("""
        <style>
               .block-container {
                    padding-top: 1rem;
                    padding-bottom: 1rem;
                    padding-left: 2rem;
                    padding-right: 5rem;
                }
        </style>
        """, unsafe_allow_html=True)

st.title('CT Analysis')

with open('./credentials.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days'],
    config['preauthorized']
)

name, authentication_status, username = authenticator.login('main', fields={'Form name': 'Login'})


if authentication_status:
    st.session_state.authentication_status = True
    st.switch_page('pages/main.py')

elif authentication_status is False:
    st.session_state.authentication_status = False
    st.markdown('<p style="background-color:rgba(255, 43, 43, 0.3);'
                'color:#000;'
                'font-size:16px;'
                'border-top-left-radius:0.5rem;'
                'border-top-right-radius:0.5rem;'
                'border-bottom-right-radius:0.5rem;'
                'border-bottom-left-radius:0.5rem;'
                'padding:2%;">'
                'Username/password is incorrect</p>',
        unsafe_allow_html=True)
    # st.error('Username/password is incorrect')

elif authentication_status is None:
    st.session_state.authentication_status = None
    st.markdown('<p style="background-color:rgba(255, 227, 18, 0.4);'
                'color:#000;'
                'font-size:16px;'
                'border-top-left-radius:0.5rem;'
                'border-top-right-radius:0.5rem;'
                'border-bottom-right-radius:0.5rem;'
                'border-bottom-left-radius:0.5rem;'
                'padding:2%;">'
                'Please enter your username and password</p>',
                unsafe_allow_html=True)
    #st.warning('Please enter your username and password')

# Register button
if st.button("Register"):
    st.switch_page("pages/register.py")

