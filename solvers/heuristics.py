from cube.cube import RubiksCube

# Map of face labels to their center color
FACE_CENTERS = {
    'U': 'y',
    'D': 'w',
    'F': 'g',
    'B': 'b',
    'L': 'o',
    'R': 'r'
}

def misplaced_heuristic(cube: RubiksCube) -> int:
    """
    Counts the number of facelets that do not match their face's center color.
    A simple approximation of distance from solved.

    Args:
        cube (RubiksCube): the cube to evaluate

    Returns:
        int: heuristic cost
    """
    count = 0
    face = cube.cube
    for face_name, center_color in FACE_CENTERS.items():
        stickers = face.get_face(face_name)
        count += sum(1 for row in stickers for sticker in row if sticker != center_color)
    return count

def weighted_heuristic(cube: RubiksCube) -> int:
    # Assume corners = 3x weight, edges = 1x
    face = cube.cube
    score = 0
    for face_name, center_color in FACE_CENTERS.items():
        stickers = face.get_face(face_name)
        # Apply weights: corners are [0][0], [0][2], [2][0], [2][2]
        for i, row in enumerate(stickers):
            for j, sticker in enumerate(row):
                if sticker != center_color:
                    weight = 3 if (i, j) in [(0,0), (0,2), (2,0), (2,2)] else 1
                    score += weight
    return score
