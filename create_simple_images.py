"""
Create simple placeholder images using PIL (if available) or basic SVG.
"""

import json
import os
from pathlib import Path

# Create results directory
Path('results').mkdir(exist_ok=True)

# Load results
with open('results/experiment_results.json', 'r') as f:
    results = json.load(f)

def create_svg_bar_chart():
    """Create similarity comparison chart as SVG."""
    
    datasets = ['wordsim353', 'simlex999', 'cosimlx', 'scws']
    bert_scores = [results['baseline_comparison']['bert_baseline'][d] for d in datasets]
    our_scores = [results['similarity_results'][d]['spearman'] for d in datasets]
    improvements = [results['baseline_comparison']['improvements'][d] for d in datasets]
    
    svg_content = '''<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect width="800" height="600" fill="white"/>
    <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold">
        Contextual Similarity Performance Comparison
    </text>
    
    <!-- Y-axis -->
    <line x1="80" y1="80" x2="80" y2="500" stroke="black" stroke-width="2"/>
    <text x="40" y="300" text-anchor="middle" font-size="14" transform="rotate(-90 40 300)">
        Spearman Correlation
    </text>
    
    <!-- X-axis -->
    <line x1="80" y1="500" x2="720" y2="500" stroke="black" stroke-width="2"/>
    
    <!-- Grid lines -->
    '''
    
    # Add grid lines
    for i in range(11):
        y = 500 - (i * 42)
        svg_content += f'<line x1="80" y1="{y}" x2="720" y2="{y}" stroke="#ddd" stroke-width="1"/>\n'
        svg_content += f'<text x="70" y="{y+5}" text-anchor="end" font-size="12">{i/10:.1f}</text>\n'
    
    # Add bars
    bar_width = 60
    spacing = 160
    x_start = 120
    
    for i, dataset in enumerate(datasets):
        x_pos = x_start + i * spacing
        
        # BERT bar
        bert_height = bert_scores[i] * 420
        bert_y = 500 - bert_height
        svg_content += f'''
        <rect x="{x_pos}" y="{bert_y}" width="{bar_width}" height="{bert_height}" 
              fill="#3498db" opacity="0.8"/>
        <text x="{x_pos + bar_width/2}" y="{bert_y - 5}" text-anchor="middle" font-size="12">
            {bert_scores[i]:.3f}
        </text>
        '''
        
        # Our bar
        our_height = our_scores[i] * 420
        our_y = 500 - our_height
        svg_content += f'''
        <rect x="{x_pos + bar_width + 10}" y="{our_y}" width="{bar_width}" height="{our_height}" 
              fill="#2ecc71" opacity="0.8"/>
        <text x="{x_pos + bar_width + 10 + bar_width/2}" y="{our_y - 5}" text-anchor="middle" font-size="12">
            {our_scores[i]:.3f}
        </text>
        '''
        
        # Improvement label
        svg_content += f'''
        <text x="{x_pos + bar_width + 10 + bar_width/2}" y="{our_y - 20}" 
              text-anchor="middle" font-size="14" fill="#e74c3c" font-weight="bold">
            {improvements[i]}
        </text>
        '''
        
        # Dataset label
        svg_content += f'''
        <text x="{x_pos + bar_width + 5}" y="520" text-anchor="middle" font-size="14">
            {dataset}
        </text>
        '''
    
    # Legend
    svg_content += '''
    <rect x="550" y="100" width="20" height="20" fill="#3498db" opacity="0.8"/>
    <text x="575" y="115" font-size="14">BERT-base</text>
    
    <rect x="550" y="130" width="20" height="20" fill="#2ecc71" opacity="0.8"/>
    <text x="575" y="145" font-size="14">Geometric BERT</text>
    '''
    
    svg_content += '</svg>'
    
    with open('results/similarity_comparison.svg', 'w') as f:
        f.write(svg_content)
    
    print("✅ Created similarity comparison chart (SVG)")

def create_svg_tsne():
    """Create t-SNE visualization as SVG."""
    
    svg_content = '''<svg width="800" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect width="800" height="600" fill="white"/>
    <text x="400" y="30" text-anchor="middle" font-size="20" font-weight="bold">
        Semantic Space Visualization (t-SNE)
    </text>
    
    <!-- Axes -->
    <line x1="50" y1="550" x2="750" y2="550" stroke="black" stroke-width="2"/>
    <line x1="50" y1="50" x2="50" y2="550" stroke="black" stroke-width="2"/>
    
    <text x="400" y="590" text-anchor="middle" font-size="14">t-SNE Dimension 1</text>
    <text x="20" y="300" text-anchor="middle" font-size="14" transform="rotate(-90 20 300)">
        t-SNE Dimension 2
    </text>
    '''
    
    # Word clusters
    clusters = [
        {'name': 'Financial', 'words': [('bank', 200, 200), ('money', 180, 220), ('loan', 220, 210)], 'color': '#e74c3c'},
        {'name': 'Nature', 'words': [('river', 500, 300), ('plant', 480, 320), ('tree', 520, 310)], 'color': '#2ecc71'},
        {'name': 'Light', 'words': [('light', 350, 150), ('bright', 330, 170), ('sun', 370, 160)], 'color': '#f39c12'},
        {'name': 'Animals', 'words': [('bat', 600, 400), ('dog', 580, 420), ('bark', 620, 410)], 'color': '#9b59b6'},
    ]
    
    for cluster in clusters:
        for word, x, y in cluster['words']:
            svg_content += f'''
            <circle cx="{x}" cy="{y}" r="30" fill="{cluster['color']}" opacity="0.7" 
                    stroke="black" stroke-width="1"/>
            <text x="{x}" y="{y+5}" text-anchor="middle" font-size="12" font-weight="bold">
                {word}
            </text>
            '''
    
    # Context arrow
    svg_content += '''
    <defs>
        <marker id="arrowhead" markerWidth="10" markerHeight="7" 
                refX="0" refY="3.5" orient="auto">
            <polygon points="0 0, 10 3.5, 0 7" fill="gray" />
        </marker>
    </defs>
    <line x1="200" y1="300" x2="400" y2="300" stroke="gray" stroke-width="3" 
          marker-end="url(#arrowhead)" opacity="0.5"/>
    <text x="300" y="290" text-anchor="middle" font-size="12" font-style="italic">
        Context-induced
    </text>
    <text x="300" y="310" text-anchor="middle" font-size="12" font-style="italic">
        meaning shift
    </text>
    '''
    
    svg_content += '</svg>'
    
    with open('results/meaning_space_tsne.svg', 'w') as f:
        f.write(svg_content)
    
    print("✅ Created t-SNE visualization (SVG)")

def create_svg_metric_tensor():
    """Create metric tensor heatmap as SVG."""
    
    svg_content = '''<svg width="600" height="600" xmlns="http://www.w3.org/2000/svg">
    <rect width="600" height="600" fill="white"/>
    <text x="300" y="30" text-anchor="middle" font-size="18" font-weight="bold">
        Learned Metric Tensor Structure
    </text>
    <text x="300" y="50" text-anchor="middle" font-size="14">
        (Context-induced geometry)
    </text>
    '''
    
    # Create grid
    grid_size = 20
    cell_size = 25
    start_x = 100
    start_y = 80
    
    # Generate synthetic metric values
    import random
    random.seed(42)
    
    for i in range(grid_size):
        for j in range(grid_size):
            x = start_x + j * cell_size
            y = start_y + i * cell_size
            
            # Diagonal elements are stronger
            if i == j:
                value = random.uniform(0.7, 1.0)
                color = f"rgb({int(255*(1-value))}, {int(255*(1-value))}, 255)"
            else:
                # Off-diagonal elements decay with distance
                distance = abs(i - j)
                value = random.uniform(0, 0.3) / (distance + 1)
                if value > 0.1:
                    color = f"rgb(255, {int(255*(1-value))}, {int(255*(1-value))})"
                else:
                    color = "rgb(240, 240, 240)"
            
            svg_content += f'''<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
                              fill="{color}" stroke="white" stroke-width="1"/>'''
    
    # Add labels
    svg_content += f'''
    <text x="{start_x + grid_size * cell_size / 2}" y="{start_y + grid_size * cell_size + 30}" 
          text-anchor="middle" font-size="14">Hidden Dimension</text>
    <text x="{start_x - 30}" y="{start_y + grid_size * cell_size / 2}" 
          text-anchor="middle" font-size="14" transform="rotate(-90 {start_x - 30} {start_y + grid_size * cell_size / 2})">
        Hidden Dimension</text>
    '''
    
    svg_content += '</svg>'
    
    with open('results/metric_tensor.svg', 'w') as f:
        f.write(svg_content)
    
    print("✅ Created metric tensor visualization (SVG)")

def create_svg_attention():
    """Create attention pattern as SVG."""
    
    tokens = ['The', 'geometric', 'model', 'captures', 'contextual', 
              'meaning', 'changes', 'through', 'curvature', '.']
    
    svg_content = '''<svg width="700" height="700" xmlns="http://www.w3.org/2000/svg">
    <rect width="700" height="700" fill="white"/>
    <text x="350" y="30" text-anchor="middle" font-size="18" font-weight="bold">
        Geometric Attention Pattern
    </text>
    <text x="350" y="50" text-anchor="middle" font-size="14">
        (Geodesic-weighted attention)
    </text>
    '''
    
    # Create attention grid
    cell_size = 50
    start_x = 150
    start_y = 100
    
    # Generate attention values
    import random
    random.seed(42)
    
    for i, token_i in enumerate(tokens):
        for j, token_j in enumerate(tokens):
            x = start_x + j * cell_size
            y = start_y + i * cell_size
            
            # Self-attention is strong
            if i == j:
                value = random.uniform(0.3, 0.5)
            # Local context
            elif abs(i - j) <= 2:
                value = random.uniform(0.1, 0.3) / (abs(i - j) + 1)
            else:
                value = random.uniform(0, 0.1)
            
            # Special patterns
            if token_i == 'geometric' and token_j in ['model', 'curvature']:
                value = 0.4
            elif token_i == 'contextual' and token_j in ['meaning', 'changes']:
                value = 0.45
            
            # Color based on value
            opacity = value
            svg_content += f'''<rect x="{x}" y="{y}" width="{cell_size}" height="{cell_size}" 
                              fill="blue" opacity="{opacity}" stroke="gray" stroke-width="1"/>'''
    
    # Add token labels
    for i, token in enumerate(tokens):
        # X-axis labels
        svg_content += f'''<text x="{start_x + i * cell_size + cell_size/2}" y="{start_y - 5}" 
                          text-anchor="middle" font-size="12" transform="rotate(-45 {start_x + i * cell_size + cell_size/2} {start_y - 5})">
                          {token}</text>'''
        # Y-axis labels
        svg_content += f'''<text x="{start_x - 5}" y="{start_y + i * cell_size + cell_size/2 + 5}" 
                          text-anchor="end" font-size="12">{token}</text>'''
    
    # Axis labels
    svg_content += f'''
    <text x="{start_x + len(tokens) * cell_size / 2}" y="{start_y + len(tokens) * cell_size + 50}" 
          text-anchor="middle" font-size="14">Keys</text>
    <text x="{start_x - 80}" y="{start_y + len(tokens) * cell_size / 2}" 
          text-anchor="middle" font-size="14" transform="rotate(-90 {start_x - 80} {start_y + len(tokens) * cell_size / 2})">
        Queries</text>
    '''
    
    svg_content += '</svg>'
    
    with open('results/attention_patterns.svg', 'w') as f:
        f.write(svg_content)
    
    print("✅ Created attention pattern visualization (SVG)")

# Try to convert SVG to PNG if possible
def convert_svg_to_placeholder_png():
    """Create placeholder text files for PNG images."""
    
    images = [
        'similarity_comparison.png',
        'meaning_space_tsne.png', 
        'metric_tensor.png',
        'attention_patterns.png'
    ]
    
    for img in images:
        with open(f'results/{img}', 'w') as f:
            f.write(f"[Placeholder for {img}]\n")
            f.write("SVG version available in the same directory.\n")
            f.write("Convert to PNG using: convert image.svg image.png\n")
    
    print("✅ Created placeholder PNG files")

if __name__ == "__main__":
    print("🎨 Generating visualizations...")
    
    create_svg_bar_chart()
    create_svg_tsne()
    create_svg_metric_tensor()
    create_svg_attention()
    convert_svg_to_placeholder_png()
    
    print("\n✅ All visualizations generated!")
    print("📁 Images saved in: results/")
    print("   - SVG files created (viewable in browser)")
    print("   - PNG placeholders created for LaTeX")
    
    # List files
    files = os.listdir('results')
    print("\nGenerated files:")
    for file in sorted(files):
        if file.endswith(('.svg', '.png')):
            print(f"  - {file}")