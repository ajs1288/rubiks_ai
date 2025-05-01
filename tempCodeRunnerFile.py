def draw_cube(ax, cube: Cube, move: str = None):
    ax.clear()
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 9)
    ax.set_aspect('equal')
    ax.axis('off')

    for face in face_order:
        x0, y0 = face_positions[face]
        face_grid = cube.get_face(face)
        for i, row in enumerate(face_grid):
            for j, sq in enumerate(row):
                color = color_map.get(sq.colour.lower(), 'gray')
                rect = plt.Rectangle((x0 + j, y0 + 2 - i), 1, 1,
                                     facecolor=color, edgecolor='black')
                ax.add_patch(rect)
        ax.text(x0 + 1.5, y0 + 3.1, face, ha='center', va='bottom',
                fontsize=12, fontweight='bold')

    if move:
        ax.set_title(f"Move: {move}", fontsize=14)

    plt.pause(0.5)  # delay between frames
