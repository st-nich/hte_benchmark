#!/usr/bin/env python3
"""
HTE Benchmark Visualizations
Run this on your local machine with matplotlib installed
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.style.use('seaborn-v0_8-whitegrid')

# ============================================================================
# FIGURE 1: Model Rankings
# ============================================================================
def fig1_model_rankings():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    models = ['T-Learner\n(Linear)', 'T-Learner\n(LGBM)', 'T-Learner\n(RF)', 
               'S-Learner\n(Linear)', 'Naive\nATE', 'X-Learner\n(LGBM)', 
               'X-Learner\n(Linear)', 'X-Learner\n(RF)']
    pehe = [0.329, 0.376, 0.434, 0.505, 0.511, 0.585, 0.601, 0.601]
    colors = ['#27ae60', '#2ecc71', '#1abc9c', '#3498db', '#3498db', '#e74c3c', '#e74c3c', '#e74c3c']
    
    bars = ax.barh(models, pehe, color=colors)
    ax.set_xlabel('PEHE (lower = better)', fontsize=12)
    ax.set_title('Model Rankings by PEHE', fontsize=14, fontweight='bold')
    ax.axvline(x=0.4, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 0.8)
    
    for bar, v in zip(bars, pehe):
        ax.text(v + 0.02, bar.get_y() + bar.get_height()/2, f'{v:.3f}', va='center')
    
    plt.tight_layout()
    plt.savefig('fig1_model_rankings.png', dpi=150, bbox_inches='tight')
    print("Saved fig1_model_rankings.png")

# ============================================================================
# FIGURE 2: Confounding Effect
# ============================================================================
def fig2_confounding():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    conf_levels = [0.0, 0.3, 0.7, 0.9]
    naive_pehe = [0.56, 0.60, 0.69, 0.81]
    t_learner_pehe = [0.24, 0.19, 0.23, 0.22]
    
    x = np.arange(len(conf_levels))
    width = 0.35
    
    ax.bar(x - width/2, naive_pehe, width, label='Naive ATE', color='#e74c3c')
    ax.bar(x + width/2, t_learner_pehe, width, label='T-Learner', color='#27ae60')
    
    ax.set_xlabel('Confounding Level', fontsize=12)
    ax.set_ylabel('PEHE', fontsize=12)
    ax.set_title('Effect of Treatment Confounding', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(['0.0\n(Random)', '0.3', '0.7', '0.9\n(High)'])
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('fig2_confounding.png', dpi=150, bbox_inches='tight')
    print("Saved fig2_confounding.png")

# ============================================================================
# FIGURE 3: Power Heatmap
# ============================================================================
def fig3_power():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data = np.array([
        [0.73, 0.65, 0.58, 0.58],
        [0.93, 0.81, 0.70, 0.59],
        [0.97, 0.92, 0.81, 0.67],
        [0.99, 0.97, 0.91, 0.76]
    ])
    effect_sizes = [0.2, 0.5, 1.0, 2.0]
    noise_levels = [0.5, 1.0, 2.0, 5.0]
    
    im = ax.imshow(data, cmap='RdYlGn', vmin=0.5, vmax=1.0)
    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels(noise_levels)
    ax.set_yticks(range(len(effect_sizes)))
    ax.set_yticklabels(effect_sizes)
    ax.set_xlabel('Noise Std Dev', fontsize=12)
    ax.set_ylabel('Effect Size', fontsize=12)
    ax.set_title('Detection Power (AUC) by Signal-to-Noise', fontsize=14, fontweight='bold')
    
    for i in range(len(effect_sizes)):
        for j in range(len(noise_levels)):
            ax.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', fontsize=11)
    
    plt.colorbar(im, ax=ax, label='AUC')
    plt.tight_layout()
    plt.savefig('fig3_power.png', dpi=150, bbox_inches='tight')
    print("Saved fig3_power.png")

# ============================================================================
# FIGURE 4: Sparse Effects
# ============================================================================
def fig4_sparse():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    responder_pcts = [5, 10, 20, 50]
    t_learner = [16, 56, 90, 100]
    s_learner = [0, 33, 44, 100]
    dml = [14, 57, 90, 100]
    
    x = np.arange(len(responder_pcts))
    width = 0.25
    
    ax.bar(x - width, t_learner, width, label='T-Learner', color='#27ae60')
    ax.bar(x, s_learner, width, label='S-Learner', color='#3498db')
    ax.bar(x + width, dml, width, label='LinearDML', color='#9b59b6')
    
    ax.set_xlabel('% of Population That Responds', fontsize=12)
    ax.set_ylabel('Precision @ 10%', fontsize=12)
    ax.set_title('Sparse Effects: Can We Find Who Responds?', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{p}%' for p in responder_pcts])
    ax.legend()
    ax.set_ylim(0, 110)
    
    plt.tight_layout()
    plt.savefig('fig4_sparse.png', dpi=150, bbox_inches='tight')
    print("Saved fig4_sparse.png")

# ============================================================================
# FIGURE 5: Bidirectional Effects
# ============================================================================
def fig5_bidirectional():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    splits = [10, 30, 50, 70, 90]
    naive = [0.60, 0.80, 1.00, 0.80, 0.60]
    t_linear = [0.50, 0.60, 0.61, 0.60, 0.50]
    t_rf = [0.30, 0.28, 0.25, 0.28, 0.30]
    
    ax.plot(splits, naive, 'o-', label='Naive ATE', color='#e74c3c', linewidth=2, markersize=8)
    ax.plot(splits, t_linear, 's-', label='T-Learner (Linear)', color='#3498db', linewidth=2, markersize=8)
    ax.plot(splits, t_rf, '^-', label='T-Learner (RF)', color='#27ae60', linewidth=2, markersize=8)
    
    ax.set_xlabel('% with Positive Effect', fontsize=12)
    ax.set_ylabel('PEHE', fontsize=12)
    ax.set_title('Bidirectional Effects: Positive vs Negative Treatment', fontsize=14, fontweight='bold')
    ax.legend()
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 1.2)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig('fig5_bidirectional.png', dpi=150, bbox_inches='tight')
    print("Saved fig5_bidirectional.png")

# ============================================================================
# RUN ALL
# ============================================================================
if __name__ == '__main__':
    print("Generating HTE Benchmark Visualizations...")
    fig1_model_rankings()
    fig2_confounding()
    fig3_power()
    fig4_sparse()
    fig5_bidirectional()
    print("\nAll figures saved!")
