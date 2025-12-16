# Deployment Guide

This guide covers deploying your Streamlit + Marimo + Anywidget application to various platforms.

## Prerequisites

- GitHub repository set up (see main README.md)
- All dependencies listed in `pyproject.toml` or `requirements.txt`

## Deployment Options

### 1. Streamlit Community Cloud (Recommended)

Streamlit Community Cloud is the easiest way to deploy Streamlit apps.

#### Steps:

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Ready for deployment"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - Streamlit Cloud automatically detects `requirements.txt` or `pyproject.toml`
   - For UV projects, ensure `requirements.txt` exists (we've included it)

#### Custom Configuration

Create `.streamlit/config.toml` in your repo:

```toml
[server]
port = 8501
enableCORS = false
enableXsrfProtection = true

[browser]
gatherUsageStats = false
```

### 2. Docker Deployment

#### Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install UV
RUN pip install uv

# Copy project files
COPY pyproject.toml ./
COPY requirements.txt ./
COPY . .

# Install dependencies
RUN uv sync

# Expose port
EXPOSE 8501

# Health check
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# Run Streamlit
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

#### Build and Run

```bash
# Build image
docker build -t streamlit-marimo-anywidget .

# Run container
docker run -p 8501:8501 streamlit-marimo-anywidget
```

### 3. Heroku Deployment

#### Create Procfile

```
web: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

#### Deploy

```bash
# Login to Heroku
heroku login

# Create app
heroku create your-app-name

# Deploy
git push heroku main
```

### 4. AWS EC2 / VPS

#### Setup Steps

1. **SSH into server**
   ```bash
   ssh user@your-server-ip
   ```

2. **Install dependencies**
   ```bash
   # Install Python and UV
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Clone repository
   git clone https://github.com/yourusername/streamlit-marimo-anywidget.git
   cd streamlit-marimo-anywidget
   
   # Install dependencies
   uv sync
   ```

3. **Run with systemd (create service)**

   Create `/etc/systemd/system/streamlit-app.service`:
   ```ini
   [Unit]
   Description=Streamlit App
   After=network.target

   [Service]
   Type=simple
   User=your-user
   WorkingDirectory=/path/to/streamlit-marimo-anywidget
   Environment="PATH=/path/to/venv/bin"
   ExecStart=/path/to/venv/bin/streamlit run app.py --server.port=8501
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

4. **Start service**
   ```bash
   sudo systemctl enable streamlit-app
   sudo systemctl start streamlit-app
   ```

5. **Configure Nginx (reverse proxy)**

   Create `/etc/nginx/sites-available/streamlit`:
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8501;
           proxy_http_version 1.1;
           proxy_set_header Upgrade $http_upgrade;
           proxy_set_header Connection "upgrade";
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }
   }
   ```

   Enable site:
   ```bash
   sudo ln -s /etc/nginx/sites-available/streamlit /etc/nginx/sites-enabled/
   sudo nginx -t
   sudo systemctl reload nginx
   ```

## Environment Variables

If your app needs environment variables:

### Streamlit Cloud
- Go to app settings
- Add secrets in "Secrets" section

### Docker
```bash
docker run -p 8501:8501 -e ENV_VAR=value streamlit-marimo-anywidget
```

### Heroku
```bash
heroku config:set ENV_VAR=value
```

## Troubleshooting

### Common Issues

1. **Port already in use**
   - Change port in Streamlit config or use different port

2. **Dependencies not found**
   - Ensure `requirements.txt` is up to date
   - Run `uv pip compile pyproject.toml -o requirements.txt`

3. **Widgets not rendering**
   - Check browser console for errors
   - Ensure JavaScript is enabled
   - Verify Anywidget is properly installed

4. **Marimo not working**
   - Check Marimo initialization
   - Verify dependencies are installed

## Monitoring

### Streamlit Cloud
- Built-in analytics and monitoring

### Custom Deployment
- Use tools like:
  - **Prometheus** for metrics
  - **Grafana** for visualization
  - **Sentry** for error tracking

## Security Considerations

1. **Secrets Management**
   - Never commit secrets to Git
   - Use environment variables or secret management services

2. **HTTPS**
   - Always use HTTPS in production
   - Use Let's Encrypt for free SSL certificates

3. **Authentication**
   - Consider adding authentication for production apps
   - Streamlit supports custom authentication

## Performance Optimization

1. **Caching**
   - Use `@st.cache_data` for expensive computations
   - Cache widget instances when possible

2. **Resource Limits**
   - Set appropriate memory limits
   - Monitor CPU usage

3. **CDN**
   - Use CDN for static assets
   - Optimize JavaScript bundles

## Support

For deployment issues:
- Check Streamlit documentation
- Review platform-specific documentation
- Open an issue on GitHub

