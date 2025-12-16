"""
Main Streamlit application integrating Marimo and Anywidget
"""
import streamlit as st
import numpy as np
from widgets.animated_widget import AnimatedCounter, AnimatedChart

# Page configuration
st.set_page_config(
    page_title="Streamlit + Marimo + Anywidget",
    page_icon="🚀",
    layout="wide"
)

# Title
st.title("🚀 Streamlit + Marimo + Anywidget Demo")
st.markdown("A comprehensive integration example demonstrating reactive Python notebooks and custom interactive widgets")

# Sidebar controls
st.sidebar.header("⚙️ Controls")
show_counter = st.sidebar.checkbox("Show Counter Widget", value=True)
show_chart = st.sidebar.checkbox("Show Chart Widget", value=True)
chart_bars = st.sidebar.slider("Number of Chart Bars", 5, 20, 10)

# Marimo Integration Section
st.header("📊 Marimo Integration")
st.markdown("Reactive Python computations with Marimo-style reactive cells")

# Simulate Marimo reactive computation
@st.cache_data
def compute_statistics():
    """Simulate a Marimo reactive cell computation"""
    data = np.random.randn(1000)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data)),
        'data': data.tolist()
    }

stats = compute_statistics()

# Display statistics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{stats['mean']:.2f}")
col2.metric("Std Dev", f"{stats['std']:.2f}")
col3.metric("Min", f"{stats['min']:.2f}")
col4.metric("Max", f"{stats['max']:.2f}")

# Display data visualization
st.subheader("Data Distribution")
st.line_chart(stats['data'])

# Anywidget Section
st.header("🎨 Anywidget Animations")
st.markdown("Custom interactive widgets with JavaScript animations")

if show_counter:
    st.subheader("Interactive Counter Widget")
    st.markdown("Click the button to increment the counter with animation effects")
    
    # Initialize counter value in session state
    if 'counter_value' not in st.session_state:
        st.session_state.counter_value = 0
    
    counter = AnimatedCounter(value=st.session_state.counter_value)
    
    # Display Anywidget in Streamlit
    # Note: Anywidget is primarily designed for Jupyter, so we use a workaround
    try:
        # Get the widget's HTML representation
        html_content = counter._repr_html_()
        if html_content:
            st.components.v1.html(html_content, height=200, scrolling=False)
        else:
            raise AttributeError("No HTML representation available")
    except (AttributeError, Exception) as e:
        # Fallback: Show widget info and use Streamlit's native components
        st.info("💡 **Note:** Anywidget works best in Jupyter notebooks. Showing widget state:")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Counter Value", st.session_state.counter_value)
        with col2:
            if st.button("➕ Increment Counter"):
                st.session_state.counter_value += 1
                st.rerun()

if show_chart:
    st.subheader("Animated Bar Chart Widget")
    st.markdown("Watch the bars animate as they appear")
    chart_data = np.random.rand(chart_bars).tolist()
    chart = AnimatedChart(data=chart_data, width=700, height=400)
    
    # Display Anywidget in Streamlit
    try:
        # Get the widget's HTML representation
        html_content = chart._repr_html_()
        if html_content:
            st.components.v1.html(html_content, height=450, scrolling=False)
        else:
            raise AttributeError("No HTML representation available")
    except (AttributeError, Exception) as e:
        # Fallback: Use Streamlit's native chart
        st.info("💡 **Note:** Anywidget works best in Jupyter notebooks. Showing data with Streamlit's native chart:")
        st.bar_chart(chart_data)
        st.write("**Chart Data:**", chart_data)

# Additional Information
with st.expander("ℹ️ About This Demo"):
    st.markdown("""
    This application demonstrates:
    - **Streamlit**: Web framework for Python applications
    - **Marimo**: Reactive Python notebook framework (simulated here)
    - **Anywidget**: Custom interactive widgets with JavaScript
    
    The widgets use traitlets for bidirectional communication between Python and JavaScript.
    """)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Marimo, and Anywidget | Powered by UV")

