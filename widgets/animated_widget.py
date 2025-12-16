"""
Custom Anywidget widgets with animations
"""
import traitlets
from anywidget import AnyWidget


class AnimatedCounter(AnyWidget):
    """
    An animated counter widget that increments on button click.
    Demonstrates basic Anywidget functionality with animations.
    """
    
    _esm = """
    export function render({ model, el }) {
        // Get initial value from Python
        let count = model.get("value");
        
        // Create container with styling
        const container = document.createElement("div");
        container.style.cssText = `
            padding: 30px;
            text-align: center;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        `;
        
        // Create counter display
        const display = document.createElement("div");
        display.id = "counter-display";
        display.style.cssText = `
            font-size: 64px;
            font-weight: bold;
            color: white;
            margin: 20px 0;
            transition: transform 0.3s ease, color 0.3s ease;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        `;
        display.textContent = count;
        
        // Create increment button
        const button = document.createElement("button");
        button.textContent = "➕ Increment";
        button.style.cssText = `
            padding: 12px 24px;
            font-size: 18px;
            font-weight: bold;
            background-color: white;
            color: #667eea;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.2);
        `;
        
        // Button hover effects
        button.onmouseenter = () => {
            button.style.transform = "translateY(-2px)";
            button.style.boxShadow = "0 4px 8px rgba(0, 0, 0, 0.3)";
        };
        
        button.onmouseleave = () => {
            button.style.transform = "translateY(0)";
            button.style.boxShadow = "0 2px 4px rgba(0, 0, 0, 0.2)";
        };
        
        // Button click handler
        button.onclick = () => {
            count = model.get("value") + 1;
            model.set("value", count);
            model.save_changes();
            
            // Animation effect on increment
            display.style.transform = "scale(1.3) rotate(5deg)";
            display.style.color = "#ffd700";
            
            setTimeout(() => {
                display.style.transform = "scale(1) rotate(0deg)";
                display.style.color = "white";
            }, 300);
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
    """
    An animated bar chart widget that animates bars on render.
    Demonstrates data visualization with Anywidget.
    """
    
    _esm = """
    export function render({ model, el }) {
        const data = model.get("data") || [];
        const width = model.get("width") || 400;
        const height = model.get("height") || 300;
        
        // Clear previous content
        el.innerHTML = "";
        
        // Create SVG canvas
        const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
        svg.setAttribute("width", width);
        svg.setAttribute("height", height);
        svg.style.cssText = `
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            background: #f9f9f9;
        `;
        
        if (data.length === 0) {
            const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
            text.setAttribute("x", width / 2);
            text.setAttribute("y", height / 2);
            text.setAttribute("text-anchor", "middle");
            text.setAttribute("font-size", "18");
            text.setAttribute("fill", "#666");
            text.textContent = "No data to display";
            svg.appendChild(text);
            el.appendChild(svg);
            return;
        }
        
        // Calculate dimensions
        const maxValue = Math.max(...data, 1);
        const barWidth = (width - 40) / data.length;
        const chartHeight = height - 60;
        const padding = 20;
        
        // Draw bars with animation
        data.forEach((value, index) => {
            const barHeight = (value / maxValue) * chartHeight;
            const x = padding + index * barWidth + 5;
            const y = height - padding - barHeight;
            
            // Create bar
            const bar = document.createElementNS("http://www.w3.org/2000/svg", "rect");
            bar.setAttribute("x", x);
            bar.setAttribute("y", height - padding);
            bar.setAttribute("width", barWidth - 10);
            bar.setAttribute("height", 0); // Start at 0 for animation
            bar.setAttribute("fill", `hsl(${index * 360 / data.length}, 70%, 50%)`);
            bar.setAttribute("rx", "4");
            bar.setAttribute("ry", "4");
            bar.style.transition = "height 0.6s ease-out, y 0.6s ease-out";
            
            // Create value label
            const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
            label.setAttribute("x", x + barWidth / 2 - 5);
            label.setAttribute("y", y - 5);
            label.setAttribute("text-anchor", "middle");
            label.setAttribute("font-size", "12");
            label.setAttribute("fill", "#333");
            label.setAttribute("font-weight", "bold");
            label.textContent = value.toFixed(2);
            label.style.opacity = "0";
            label.style.transition = "opacity 0.3s ease-out";
            
            // Animate bar appearance
            setTimeout(() => {
                bar.setAttribute("height", barHeight);
                bar.setAttribute("y", y);
                label.style.opacity = "1";
            }, index * 100);
            
            // Hover effect
            bar.onmouseenter = () => {
                bar.style.opacity = "0.8";
                bar.style.transform = "scaleY(1.05)";
            };
            
            bar.onmouseleave = () => {
                bar.style.opacity = "1";
                bar.style.transform = "scaleY(1)";
            };
            
            svg.appendChild(bar);
            svg.appendChild(label);
        });
        
        // Draw axis
        const axis = document.createElementNS("http://www.w3.org/2000/svg", "line");
        axis.setAttribute("x1", padding);
        axis.setAttribute("y1", height - padding);
        axis.setAttribute("x2", width - padding);
        axis.setAttribute("y2", height - padding);
        axis.setAttribute("stroke", "#333");
        axis.setAttribute("stroke-width", "2");
        svg.insertBefore(axis, svg.firstChild);
        
        el.appendChild(svg);
        
        // Update on data change
        model.on("change:data", () => {
            // Re-render on data change
            render({ model, el });
        });
    }
    """
    
    data = traitlets.List(traitlets.Float()).tag(sync=True)
    width = traitlets.Int(400).tag(sync=True)
    height = traitlets.Int(300).tag(sync=True)

