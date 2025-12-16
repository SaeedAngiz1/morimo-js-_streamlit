"""
Helper functions for displaying Anywidget in Streamlit
"""
import streamlit as st
from IPython.display import HTML


def display_anywidget(widget, height=400):
    """
    Display an Anywidget in Streamlit.
    
    Args:
        widget: Anywidget instance
        height: Height of the widget container in pixels
    
    Returns:
        None (displays the widget)
    """
    try:
        # Try to get HTML representation using IPython's display system
        from IPython.display import display
        
        # Get the widget's HTML representation
        if hasattr(widget, '_repr_html_'):
            html_content = widget._repr_html_()
        elif hasattr(widget, '_repr_mimebundle_'):
            bundle = widget._repr_mimebundle_()
            html_content = bundle.get('text/html', [''])[0]
        else:
            # Create a basic HTML wrapper
            html_content = f"""
            <div style="padding: 20px; border: 1px solid #ddd; border-radius: 5px;">
                <p>Anywidget: {type(widget).__name__}</p>
                <p style="color: #666;">Widget attributes: {dir(widget)}</p>
            </div>
            """
        
        # Display in Streamlit
        if html_content:
            st.components.v1.html(html_content, height=height)
        else:
            st.warning("Widget HTML representation is empty")
            
    except ImportError:
        st.error("IPython is required for Anywidget display. Install it with: pip install ipython")
    except Exception as e:
        st.error(f"Error displaying widget: {str(e)}")
        st.info("Note: Anywidget works best in Jupyter notebooks. For Streamlit, consider using native Streamlit components or creating custom Streamlit components.")


def display_anywidget_simple(widget, height=400):
    """
    Simple fallback method to display widget information.
    """
    st.write(f"**Widget Type:** {type(widget).__name__}")
    
    # Display widget attributes
    if hasattr(widget, 'value'):
        st.write(f"**Value:** {widget.value}")
    if hasattr(widget, 'data'):
        st.write(f"**Data:** {widget.data}")
    
    # Try to render as HTML
    try:
        html = widget._repr_html_() if hasattr(widget, '_repr_html_') else None
        if html:
            st.components.v1.html(html, height=height)
    except:
        pass

