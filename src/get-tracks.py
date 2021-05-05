
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json

def draw_border(border, img_shape):
    img = np.zeros(img_shape) + 1
    for c in border:
        img[int(c[0]), int(c[1])] = 0
    return img

def find_upper_left_from(img, x=0, y=0):
    i=1
    while i:
        i+=1
        for j in range(i):
            if img[x+i-j-1, y+j] == 0:
                return (x+i-j-1, y+j)
    return None

def boundary_following(border, upper_left_corner=(0,0), stay_out=True, allow_diag=True):
    
    x_n = [0, -1, -1, -1,  0,  1, 1, 1] if allow_diag else [0, -1,  0, 1]
    y_n = [1,  1,  0, -1, -1, -1, 0, 1] if allow_diag else [1,  0, -1, 0]
    nbr_neighbore = len(x_n)
    angle = nbr_neighbore//2
    
    def get_neighbor(p,a):
        x, y = p[0] + x_n[a], p[1] + y_n[a]
        if (0 <= x < border.shape[0]) and (0 <= y < border.shape[1]):
            return border[x, y] == 0
        else: return None

    b = find_upper_left_from(border, upper_left_corner[0], upper_left_corner[1])
    b_init = False
    coord_border = [b]
    chain_code = []

    while True:
        # Revolve around b until hit border
        while not get_neighbor(b, angle):
            angle = (angle - 1) if angle else (nbr_neighbore - 1)
        # Prefer direct neighbore
        if (not stay_out) and allow_diag and (angle%2 == 1) \
            and get_neighbor(b, (angle - 1) if angle else 7):
            angle = (angle - 1) if angle else (nbr_neighbore - 1)
        # Update b <- n(k)
        b = (b[0] + x_n[angle], b[1] + y_n[angle])
        # End condition: two successive boundary pixels already visited
        if b_init:
            if b == coord_border[1]: break
            else: b_init = False
        if b == coord_border[0]: b_init = True
        # Store new border pixel
        chain_code.append(angle)
        coord_border.append(b)
        # Reset angle, c <- n(k−1)
        angle = (angle+angle%2+2)%8 if allow_diag else (angle+1)%4
    return np.array(coord_border), chain_code

def capture_track_positions(img, track_names: []):
    
    track_positions = {}
    index = 0

    print(track_names[index])
    def onClick(event, x, y, flags, param):
        nonlocal index
        if event == cv2.EVENT_LBUTTONDOWN:
            if index>=len(track_names):
                print("No more tracks")
                return
            print(x, y)
            track_positions[track_names[index]] = (y,x)
            index += 1
            print("No more tracks" if index>=len(track_names) else track_names[index])

    cv2.imshow('image', img_raw)
    cv2.setMouseCallback('image', onClick)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return track_positions

# driver function
if __name__=="__main__":

    '''
    img_raw = cv2.imread('./dataset/test.jpg',cv2.IMREAD_GRAYSCALE)
    img_threshold = cv2.threshold(img_raw,20,1,cv2.THRESH_BINARY)[1]
    
    track_upper_left = capture_track_positions(img_threshold, ["track name"])["track name"]
    print(track_upper_left)

    #img_smoothed = cv2.blur(img_raw,(3,3))
    border_coord, _ = boundary_following(img_threshold, track_upper_left)
    print(border_coord.shape)

    # Visualization
    fig, axs = plt.subplots(2)
    for ax in axs:
        ax.set_xticks([])
        ax.set_yticks([])
    axs[0].set_title('Input image')
    axs[0].imshow(img_raw, cmap='gray')
    axs[1].set_title('Outer boundary')
    axs[1].imshow(draw_border(border_coord, img_raw.shape), cmap='gray')

    fig.tight_layout()
    plt.show()
    '''

    with open('./dataset/tracks.json') as f:
        data = json.load(f)
    print(json.dumps(data, indent=4))
