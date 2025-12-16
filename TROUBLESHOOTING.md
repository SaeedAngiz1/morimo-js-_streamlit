# Troubleshooting Guide

Common issues and solutions for the Streamlit + Marimo + Anywidget project.

## Common Errors

### 1. "ModuleNotFoundError: No module named 'anywidget'"

**Solution:**
```bash
# Install dependencies using UV
uv sync

# Or manually
uv add anywidget
```

### 2. "AttributeError: 'AnimatedCounter' object has no attribute '_repr_html_'"

**Cause:** Anywidget is primarily designed for Jupyter notebooks and may not have HTML representation in all environments.

**Solutions:**

**Option A: Install IPython (Recommended)**
```bash
uv add ipython
```

**Option B: Use the fallback display**
The app includes fallback displays that work even if HTML representation isn't available. The widgets will show their state using Streamlit's native components.

**Option C: Run in Jupyter Notebook**
Anywidget works best in Jupyter. Consider using Jupyter for widget development:
```bash
uv add jupyter
uv run jupyter notebook
```

### 3. "ImportError: cannot import name 'AnyWidget' from 'anywidget'"

**Solution:**
```bash
# Update anywidget
uv add "anywidget>=0.9.0" --upgrade
```

### 4. Widgets not displaying in Streamlit

**Cause:** Anywidget is optimized for Jupyter notebooks, not Streamlit.

**Solutions:**

1. **Install IPython:**
   ```bash
   uv add ipython
   ```

2. **Use the fallback mode:** The app automatically falls back to Streamlit's native components if widgets can't render.

3. **Check browser console:** Open browser developer tools (F12) and check for JavaScript errors.

4. **Alternative:** Consider creating custom Streamlit components instead of using Anywidget directly in Streamlit.

### 5. "UV command not found"

**Solution:**

**Windows:**
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

After installation, restart your terminal.

### 6. Streamlit app won't start

**Check:**
- Python version: `python --version` (should be 3.8+)
- Dependencies installed: `uv sync`
- Port 8501 available: Try `streamlit run app.py --server.port=8502`

### 7. "Marimo not found" or Marimo errors

**Note:** The current implementation simulates Marimo functionality. For full Marimo support:

```bash
uv add marimo
```

Then check the [Marimo documentation](https://marimo.io/docs) for proper integration.

### 8. Widgets show but don't interact

**Possible causes:**
- JavaScript errors in browser console
- Widget state not syncing properly
- Browser compatibility issues

**Solutions:**
1. Check browser console (F12) for errors
2. Try a different browser
3. Clear browser cache
4. Use the fallback interactive mode in the app

## Getting Help

1. **Check the error message** - Read it carefully for clues
2. **Check browser console** - Open F12 and look for JavaScript errors
3. **Verify dependencies** - Run `uv sync` to ensure everything is installed
4. **Check Python version** - Ensure you're using Python 3.8+
5. **Review logs** - Check Streamlit's error messages in the terminal

## Environment Setup Verification

Run this to verify your setup:

```bash
# Check Python version
python --version

# Check UV installation
uv --version

# Verify dependencies
uv sync

# Test imports
python -c "import streamlit; import anywidget; import numpy; print('All imports successful!')"
```

## Alternative Approaches

If Anywidget continues to cause issues in Streamlit:

1. **Use Streamlit's native components** - The app includes fallback modes
2. **Create custom Streamlit components** - More work but better integration
3. **Use Plotly or Bokeh** - For interactive visualizations
4. **Run widgets in Jupyter** - Use Jupyter for widget development, Streamlit for deployment

## Still Having Issues?

1. Check the [main README.md](README.md) for setup instructions
2. Review the [QUICKSTART.md](QUICKSTART.md) for basic setup
3. Open an issue on GitHub with:
   - Error message
   - Python version
   - Operating system
   - Steps to reproduce

