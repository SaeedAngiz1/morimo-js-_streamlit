# Streamlit + Marimo + Anywidget Integration Guide

A comprehensive guide to creating a Streamlit web app that integrates Marimo (reactive Python notebooks) and Anywidget (custom interactive widgets), using UV as the package manager.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Project Setup with UV](#project-setup-with-uv)
3. [Project Structure](#project-structure)
4. [Creating the Streamlit App](#creating-the-streamlit-app)
5. [Integrating Marimo](#integrating-marimo)
6. [Adding Anywidget Animations](#adding-anywidget-animations)
7. [Python-JavaScript Communication](#python-javascript-communication)
8. [Deployment to GitHub](#deployment-to-github)
9. [Running the Application](#running-the-application)

---

## Prerequisites

- Python 3.8 or higher
- Git installed on your system
- A GitHub account (for deployment)

---

## Project Setup with UV

### Step 1: Install UV

UV is a fast Python package manager written in Rust. Install it using one of these methods:

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative (using pip):**
```bash
pip install uv
```

### Step 2: Initialize the Project

```bash
# Create and navigate to project directory
mkdir streamlit-marimo-anywidget
cd streamlit-marimo-anywidget

# Initialize UV project
uv init
```

This creates:
- `pyproject.toml` - Project configuration and dependencies
- `.venv/` - Virtual environment (automatically managed by UV)
- Basic project structure

### Step 3: Add Dependencies

```bash
uv add streamlit marimo anywidget
```

This installs:
- **Streamlit**: Web framework for Python apps
- **Marimo**: Reactive Python notebooks
- **Anywidget**: Custom interactive widgets with JavaScript

---

## Project Structure

After setup, your project should look like this:

```
streamlit-marimo-anywidget/
├── .venv/                 # Virtual environment (auto-generated)
├── .gitignore            # Git ignore file
├── pyproject.toml        # UV project configuration
├── requirements.txt      # For compatibility (optional)
├── README.md            # This file
├── app.py               # Main Streamlit application
├── widgets/
│   ├── __init__.py
│   └── animated_widget.py  # Custom Anywidget example
└── marimo_modules/
    └── simple_marimo.py    # Marimo example module
```

---

## Creating the Streamlit App

### Basic Streamlit App Structure

Create `app.py`:

```python
import streamlit as st

st.set_page_config(
    page_title="Streamlit + Marimo + Anywidget",
    page_icon="🚀",
    layout="wide"
)

st.title("🚀 Streamlit + Marimo + Anywidget Demo")
st.markdown("A demonstration of integrating Marimo and Anywidget in Streamlit")
```

---

## Integrating Marimo

Marimo is a reactive Python notebook framework. Here's how to integrate it:

### Option 1: Using Marimo as a Component

Create `marimo_modules/simple_marimo.py`:

```python
"""
Simple Marimo module example
"""
import marimo

__all__ = ["create_marimo_app"]

def create_marimo_app():
    """Create a simple Marimo app"""
    app = marimo.App()
    
    @app.cell
    def __():
        import pandas as pd
        import numpy as np
        return pd, np
    
    @app.cell
    def __(pd, np):
        # Generate sample data
        data = pd.DataFrame({
            'x': np.random.randn(100),
            'y': np.random.randn(100)
        })
        return data
    
    @app.cell
    def __(data):
        # Simple computation
        result = f"Data shape: {data.shape}, Mean X: {data['x'].mean():.2f}"
        return result
    
    return app
```

### Option 2: Direct Integration in Streamlit

Add to `app.py`:

```python
import streamlit as st
import marimo

# Initialize Marimo
marimo.init()

# Create a reactive Marimo cell
@marimo.cell
def marimo_computation():
    import numpy as np
    data = np.random.randn(50)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'data': data.tolist()
    }

# Use in Streamlit
st.header("📊 Marimo Integration")
result = marimo_computation()
st.write(f"**Mean:** {result['mean']:.2f}")
st.write(f"**Std:** {result['std']:.2f}")
st.line_chart(result['data'])
```

---

## Adding Anywidget Animations

Anywidget allows you to create custom interactive widgets with JavaScript.

### Creating a Custom Animated Widget

Create `widgets/animated_widget.py`:

```python
"""
Custom Anywidget with animation
"""
import traitlets
from anywidget import AnyWidget


class AnimatedCounter(AnyWidget):
    """An animated counter widget"""
    
    _esm = """
    export function render({ model, el }) {
        // Get initial value
        let count = model.get("value");
        
        // Create container
        const container = document.createElement("div");
        container.style.cssText = `
            padding: 20px;
            text-align: center;
            font-family: Arial, sans-serif;
        `;
        
        // Create counter display
        const display = document.createElement("div");
        display.id = "counter-display";
        display.style.cssText = `
            font-size: 48px;
            font-weight: bold;
            color: #1f77b4;
            margin: 20px 0;
            transition: transform 0.2s;
        `;
        display.textContent = count;
        
        // Create button
        const button = document.createElement("button");
        button.textContent = "Increment";
        button.style.cssText = `
            padding: 10px 20px;
            font-size: 16px;
            background-color: #1f77b4;
            color: white;
            border: none;
            border-radius: 5px;
            cursor: pointer;
        `;
        
        // Button click handler
        button.onclick = () => {
            count = model.get("value") + 1;
            model.set("value", count);
            model.save_changes();
            
            // Animation effect
            display.style.transform = "scale(1.2)";
            setTimeout(() => {
                display.style.transform = "scale(1)";
            }, 200);
        };
        
        // Listen for value changes from Python
        model.on("change:value", () => {
            count = model.get("value");
            display.textContent = count;
        });
        
        container.appendChild(display);
        container.appendChild(button);
        el.appendChild(container);
    }
    """
    
    value = traitlets.Int(0).tag(sync=True)


class AnimatedChart(AnyWidget):
    """An animated chart widget"""
    
    _esm = """
    export function render({ model, el }) {
        const data = model.get("data") || [];
        const width = model.get("width") || 400;
        const height = model.get("height") || 300;
        
        // Create SVG canvas
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", width);
        svg.setAttribute("height", height);
        svg.style.border = "1px solid #ccc";
        
        // Draw animated bars
        const maxValue = Math.max(...data, 1);
        const barWidth = width / data.length;
        
        data.forEach((value, index) => {
            const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            const barHeight = (value / maxValue) * height * 0.8;
            
            bar.setAttribute("x", index * barWidth + 5);
            bar.setAttribute("y", height - barHeight);
            bar.setAttribute("width", barWidth - 10);
            bar.setAttribute("height", 0); // Start at 0 for animation
            bar.setAttribute("fill", `hsl(${index * 360 / data.length}, 70%, 50%)`);
            
            // Animate height
            setTimeout(() => {
                bar.setAttribute("height", barHeight);
                bar.style.transition = "height 0.5s ease-out";
            }, index * 100);
            
            svg.appendChild(bar);
        });
        
        el.appendChild(svg);
        
        // Update on data change
        model.on("change:data", () => {
            const newData = model.get("data");
            // Re-render logic here
        });
    }
    """
    
    data = traitlets.List(traitlets.Float()).tag(sync=True)
    width = traitlets.Int(400).tag(sync=True)
    height = traitlets.Int(300).tag(sync=True)
```

### Using Anywidget in Streamlit

Add to `app.py`:

```python
import streamlit as st
from widgets.animated_widget import AnimatedCounter, AnimatedChart
import numpy as np

st.header("🎨 Anywidget Animations")

# Counter widget
st.subheader("Interactive Counter")
counter = AnimatedCounter(value=0)
st.components.v1.html(counter._repr_html_(), height=200)

# Chart widget
st.subheader("Animated Chart")
chart_data = np.random.rand(10).tolist()
chart = AnimatedChart(data=chart_data, width=600, height=300)
st.components.v1.html(chart._repr_html_(), height=350)
```

---

## Python-JavaScript Communication

### Key Concepts

1. **Traitlets**: Python traits that sync with JavaScript
   - Use `.tag(sync=True)` to enable bidirectional sync
   - Changes in Python → JavaScript automatically
   - Changes in JavaScript → Python automatically

2. **Model API in JavaScript**:
   - `model.get("property")` - Get value from Python
   - `model.set("property", value)` - Set value (syncs to Python)
   - `model.save_changes()` - Commit changes
   - `model.on("change:property", callback)` - Listen for changes

### Example: Two-Way Communication

```python
class InteractiveWidget(AnyWidget):
    _esm = """
    export function render({ model, el }) {
        const input = document.createElement("input");
        input.type = "text";
        input.value = model.get("text") || "";
        
        // JavaScript → Python
        input.oninput = (e) => {
            model.set("text", e.target.value);
            model.save_changes();
        };
        
        // Python → JavaScript
        model.on("change:text", () => {
            input.value = model.get("text");
        });
        
        el.appendChild(input);
    }
    """
    
    text = traitlets.Unicode("").tag(sync=True)
```

---

## Deployment to GitHub

### Step 1: Create Configuration Files

#### `.gitignore`

Create `.gitignore`:

```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python

# Virtual Environment
.venv/
venv/
ENV/
env/

# UV
.uv/
uv.lock

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Streamlit
.streamlit/secrets.toml

# Jupyter
.ipynb_checkpoints/
*.ipynb

# Distribution
dist/
build/
*.egg-info/
```

#### `requirements.txt` (Optional, for compatibility)

Generate from UV:

```bash
uv pip compile pyproject.toml -o requirements.txt
```

Or manually create:

```
streamlit>=1.28.0
marimo>=0.1.0
anywidget>=0.9.0
```

### Step 2: Initialize Git Repository

```bash
# Initialize git
git init

# Add all files
git add .

# Initial commit
git commit -m "Initial commit: Streamlit + Marimo + Anywidget integration"
```

### Step 3: Create GitHub Repository

1. Go to [GitHub](https://github.com) and create a new repository
2. **Do NOT** initialize with README, .gitignore, or license (we already have these)

### Step 4: Push to GitHub

```bash
# Add remote (replace with your repository URL)
git remote add origin https://github.com/yourusername/streamlit-marimo-anywidget.git

# Rename branch to main (if needed)
git branch -M main

# Push to GitHub
git push -u origin main
```

### Step 5: Add Repository Badges (Optional)

Add to README.md:

```markdown
![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![UV](https://img.shields.io/badge/uv-latest-orange.svg)
```

---

## Running the Application

### Local Development

```bash
# Activate UV environment (if needed)
uv sync

# Run Streamlit app
uv run streamlit run app.py
```

Or:

```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`

### Deployment Options

1. **Streamlit Community Cloud**:
   - Connect your GitHub repository
   - Streamlit automatically detects `requirements.txt` or `pyproject.toml`
   - Deploys automatically on push

2. **Docker** (for custom deployment):
   ```dockerfile
   FROM python:3.11-slim
   WORKDIR /app
   COPY . .
   RUN pip install uv && uv sync
   EXPOSE 8501
   CMD ["streamlit", "run", "app.py", "--server.port=8501"]
   ```

---

## Complete Example: `app.py`

Here's a complete example combining everything:

```python
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
st.markdown("A comprehensive integration example")

# Sidebar
st.sidebar.header("Controls")
show_counter = st.sidebar.checkbox("Show Counter", value=True)
show_chart = st.sidebar.checkbox("Show Chart", value=True)

# Marimo Section
st.header("📊 Marimo Integration")
st.markdown("Reactive Python computations with Marimo")

# Simple Marimo-style computation
@st.cache_data
def compute_statistics():
    data = np.random.randn(1000)
    return {
        'mean': float(np.mean(data)),
        'std': float(np.std(data)),
        'min': float(np.min(data)),
        'max': float(np.max(data))
    }

stats = compute_statistics()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Mean", f"{stats['mean']:.2f}")
col2.metric("Std Dev", f"{stats['std']:.2f}")
col3.metric("Min", f"{stats['min']:.2f}")
col4.metric("Max", f"{stats['max']:.2f}")

# Anywidget Section
st.header("🎨 Anywidget Animations")

if show_counter:
    st.subheader("Interactive Counter")
    counter = AnimatedCounter(value=0)
    st.components.v1.html(counter._repr_html_(), height=200)

if show_chart:
    st.subheader("Animated Chart")
    num_bars = st.slider("Number of bars", 5, 20, 10)
    chart_data = np.random.rand(num_bars).tolist()
    chart = AnimatedChart(data=chart_data, width=700, height=400)
    st.components.v1.html(chart._repr_html_(), height=450)

# Footer
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit, Marimo, and Anywidget")
```

---

## Troubleshooting

### Common Issues

1. **UV not found**: Add UV to PATH or use full path
2. **Import errors**: Run `uv sync` to ensure dependencies are installed
3. **Widget not rendering**: Check browser console for JavaScript errors
4. **Marimo not working**: Ensure Marimo is properly initialized

### Getting Help

- [UV Documentation](https://docs.astral.sh/uv/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Marimo Documentation](https://marimo.io/docs)
- [Anywidget Documentation](https://anywidget.dev/)

---

## License

This project is open source and available under the MIT License.

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Happy Coding! 🚀**

