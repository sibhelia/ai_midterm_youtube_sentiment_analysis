import matplotlib.pyplot as plt
import os

def draw_neural_net(ax, left, right, bottom, top, layer_sizes):
    v_spacing = (top - bottom)/float(max(layer_sizes))
    h_spacing = (right - left)/float(len(layer_sizes) - 1)
    
    for n, layer_size in enumerate(layer_sizes):
        layer_top = v_spacing*(layer_size - 1)/2. + (top + bottom)/2.
        for m in range(layer_size):
            circle = plt.Circle((n*h_spacing + left, layer_top - m*v_spacing), v_spacing/4.,
                                color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            
            if m == 0:
                if n == 0: 
                    label = "Girdi Katmanı\n(300 Özellik)"
                    y_pos = layer_top + v_spacing*1.5
                elif n == len(layer_sizes)-1: 
                    label = "Çıktı Katmanı\n(3 Sınıf)"
                    y_pos = top + 0.05 
                else: 
                    label = f"Gizli Katman\n(500 Nöron)"
                    y_pos = top + 0.02  
                plt.text(n*h_spacing + left, y_pos, label, ha='center', fontsize=12)

    for n, (layer_size_a, layer_size_b) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        layer_top_a = v_spacing*(layer_size_a - 1)/2. + (top + bottom)/2.
        layer_top_b = v_spacing*(layer_size_b - 1)/2. + (top + bottom)/2.
        for m in range(layer_size_a):
            for o in range(layer_size_b):
                line = plt.Line2D([n*h_spacing + left, (n + 1)*h_spacing + left],
                                  [layer_top_a - m*v_spacing, layer_top_b - o*v_spacing], c='k', alpha=0.2)
                ax.add_artist(line)

fig = plt.figure(figsize=(12, 8))
ax = fig.gca()
ax.axis('off')


draw_neural_net(ax, .1, .9, .15, .85, [8, 12, 3]) 

plt.title("Model 1 (Geniş Mimari) Ağ Topolojisi", fontsize=15, pad=20)
plt.tight_layout()

output_dir = 'reports'
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'network_topology.png')

plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"Ağ topolojisi resmi '{output_path}' olarak kaydedildi.")
plt.show()
