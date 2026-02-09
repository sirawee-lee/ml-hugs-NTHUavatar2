#!/usr/bin/env python3
"""
Quick visualization of converted SMPL parameters to check orientation.
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def visualize_smpl_npz(npz_path):
    """Visualize SMPL parameters from npz file."""
    data = np.load(npz_path)
    
    print(f"\nLoaded: {npz_path}")
    print(f"Keys: {list(data.keys())}")
    
    global_orient = data['global_orient']
    body_pose = data['body_pose']
    transl = data['transl']
    betas = data['betas']
    
    print(f"\nShapes:")
    print(f"  global_orient: {global_orient.shape}")
    print(f"  body_pose: {body_pose.shape}")
    print(f"  transl: {transl.shape}")
    print(f"  betas: {betas.shape}")
    
    # Show translation trajectory
    print(f"\nTranslation stats:")
    print(f"  X range: [{transl[:, 0].min():.3f}, {transl[:, 0].max():.3f}]")
    print(f"  Y range: [{transl[:, 1].min():.3f}, {transl[:, 1].max():.3f}]")
    print(f"  Z range: [{transl[:, 2].min():.3f}, {transl[:, 2].max():.3f}]")
    
    # Show global orientation (root rotation)
    print(f"\nGlobal orient (axis-angle) stats:")
    print(f"  X range: [{global_orient[:, 0].min():.3f}, {global_orient[:, 0].max():.3f}]")
    print(f"  Y range: [{global_orient[:, 1].min():.3f}, {global_orient[:, 1].max():.3f}]")
    print(f"  Z range: [{global_orient[:, 2].min():.3f}, {global_orient[:, 2].max():.3f}]")
    
    # Plot translation trajectory
    fig = plt.figure(figsize=(15, 5))
    
    # 3D trajectory
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.plot(transl[:, 0], transl[:, 1], transl[:, 2], 'b-', linewidth=2)
    ax1.scatter(transl[0, 0], transl[0, 1], transl[0, 2], c='g', s=100, marker='o', label='Start')
    ax1.scatter(transl[-1, 0], transl[-1, 1], transl[-1, 2], c='r', s=100, marker='x', label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_zlabel('Z')
    ax1.set_title('3D Trajectory')
    ax1.legend()
    ax1.grid(True)
    
    # XY view
    ax2 = fig.add_subplot(132)
    ax2.plot(transl[:, 0], transl[:, 1], 'b-', linewidth=2)
    ax2.scatter(transl[0, 0], transl[0, 1], c='g', s=100, marker='o', label='Start')
    ax2.scatter(transl[-1, 0], transl[-1, 1], c='r', s=100, marker='x', label='End')
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y (up)')
    ax2.set_title('Top View (XY)')
    ax2.grid(True)
    ax2.axis('equal')
    ax2.legend()
    
    # XZ view
    ax3 = fig.add_subplot(133)
    ax3.plot(transl[:, 0], transl[:, 2], 'b-', linewidth=2)
    ax3.scatter(transl[0, 0], transl[0, 2], c='g', s=100, marker='o', label='Start')
    ax3.scatter(transl[-1, 0], transl[-1, 2], c='r', s=100, marker='x', label='End')
    ax3.set_xlabel('X')
    ax3.set_ylabel('Z (forward)')
    ax3.set_title('Side View (XZ)')
    ax3.grid(True)
    ax3.axis('equal')
    ax3.legend()
    
    plt.tight_layout()
    output_path = npz_path.replace('.npz', '_trajectory.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved plot to: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Visualize SMPL NPZ parameters')
    parser.add_argument('--input', '-i', required=True, help='Path to NPZ file')
    args = parser.parse_args()
    
    visualize_smpl_npz(args.input)


if __name__ == '__main__':
    main()
