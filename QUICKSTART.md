# Quick Start Guide

Get up and running in 5 minutes!

## Prerequisites

- Python 3.8+
- Git (optional, for cloning)

## Installation Steps

### 1. Install UV

**Windows (PowerShell):**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Alternative:**
```bash
pip install uv
```

### 2. Install Dependencies

```bash
# Install all dependencies
uv sync
```

Or manually:
```bash
uv add streamlit marimo anywidget numpy
```

### 3. Run the Application

```bash
# Using UV
uv run streamlit run app.py

# Or directly
streamlit run app.py
```

The app will open at `http://localhost:8501`

## What You'll See

1. **Marimo Integration Section**: Shows reactive Python computations with statistics
2. **Anywidget Animations**: Interactive counter and animated bar chart
3. **Sidebar Controls**: Toggle widgets and adjust settings

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Check [DEPLOYMENT.md](DEPLOYMENT.md) for deployment options
- Customize the widgets in `widgets/animated_widget.py`
- Modify the app in `app.py`

## Troubleshooting

**UV not found?**
- Add UV to your PATH, or use full path to UV executable
- On Windows, restart PowerShell after installation

**Import errors?**
- Run `uv sync` to ensure all dependencies are installed
- Check that you're in the project directory

**Widgets not showing?**
- Check browser console for JavaScript errors
- Ensure JavaScript is enabled in your browser

## Need Help?

- Check the main [README.md](README.md)
- Review the code comments in `app.py` and `widgets/animated_widget.py`
- Open an issue on GitHub

Happy coding! 🚀

