
import numpy as np
import cv2
from matplotlib import pyplot as plt
import json
import sys
from json_encoder import NoIndent, MyEncoder

class TracksExtraction:

    def __init__(self, img_path:str, input_data_path:str, display_factor:float=1.0):
        self.display_factor = display_factor
        with open(input_data_path) as f:
            self.data = json.load(f)
        self.img_raw = cv2.imread(img_path,cv2.IMREAD_GRAYSCALE)
        self.img_threshold = cv2.threshold(self.img_raw,20,1,cv2.THRESH_BINARY)[1]
        self.img_display = cv2.resize(self.img_raw, (int(self.img_raw.shape[1] * display_factor), int(self.img_raw.shape[0] * display_factor)))

    def save_data(self, path_save:str):
        with open(path_save, 'w') as outfile:
            json.dump(self.data, outfile, indent=4)
        print("Tracks data saved in", path_save)

    def __find_upper_left_from(self, x=0, y=0):
        i=1
        while i:
            i+=1
            for j in range(i):
                if self.img_threshold[x+i-j-1, y+j] == 0:
                    return (x+i-j-1, y+j)
        return None

    def __boundary_following(self, upper_left_corner=(0,0), stay_out=True, allow_diag=True):
        
        x_n = [0, -1, -1, -1,  0,  1, 1, 1] if allow_diag else [0, -1,  0, 1]
        y_n = [1,  1,  0, -1, -1, -1, 0, 1] if allow_diag else [1,  0, -1, 0]
        nbr_neighbore = len(x_n)
        angle = nbr_neighbore//2
        
        def get_neighbor(p,a):
            x, y = p[0] + x_n[a], p[1] + y_n[a]
            if (0 <= x < self.img_threshold.shape[0]) and (0 <= y < self.img_threshold.shape[1]):
                return self.img_threshold[x, y] == 0
            else: return None

        b = self.__find_upper_left_from(upper_left_corner[0], upper_left_corner[1])
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
        return coord_border, chain_code

    def capture_track_positions(self):
        def onClick(event, x, y, flags, param):
            nonlocal index_capture_track
            self.img_display
            if event == cv2.EVENT_RBUTTONDOWN:
                print("Pass track")
                index_capture_track += 1
                print("No more tracks" if index_capture_track>=len(self.data["tracks"]) else self.data["tracks"][index_capture_track]["name"])
            if event == cv2.EVENT_LBUTTONDOWN:
                if index_capture_track>=len(self.data["tracks"]):
                    print("No more tracks")
                    return
                x_scaled, y_scaled = int(x / self.display_factor), int(y / self.display_factor)
                print(x_scaled, y_scaled)
                self.data["tracks"][index_capture_track]['upper-left-corner'] = (y_scaled,x_scaled)
                index_capture_track += 1

                self.img_display = cv2.circle(self.img_display, (x,y), 2, color=(0, 0, 255), thickness=-1)
                cv2.imshow('image', self.img_display)
                print("No more tracks" if index_capture_track>=len(self.data["tracks"]) else self.data["tracks"][index_capture_track]["name"])
        
        index_capture_track = 0
        print(self.data["tracks"][index_capture_track]['name'])
        cv2.imshow('image', self.img_display)
        cv2.setMouseCallback('image', onClick)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def tracks_point_generation(self):
        max_coordinate = 0.0
        borders_centered = {}
        for track in self.data["tracks"]:
            if 'upper-left-corner' in track:
                border, _ = self.__boundary_following(track['upper-left-corner'])
                border_centered = np.array(border)
                border_centered = border_centered - border_centered.mean(0)
                borders_centered[track['name']] = border_centered
                max_coordinate = max(max_coordinate, np.absolute(border_centered).max())
        
        for i in range(len(self.data["tracks"])):
            if self.data["tracks"][i]['name'] in borders_centered:
                self.data["tracks"][i]['points'] = (borders_centered[self.data["tracks"][i]['name']] / max_coordinate).tolist()
    
    def display_track(self, index, save:bool=False, path:str="./"):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(*np.array(self.data["tracks"][index]['points']).transpose(), s=0.2)
        ax.set_xlim(-1, 1)
        ax.set_ylim(-1, 1)
        ax.set_aspect('equal')
        ax.set_title(self.data["tracks"][index]['name'])
        if save:
            fig.savefig(path+self.data["tracks"][index]['name'].replace(' ', '_')+'.png', dpi=250)
            plt.close('all')
        else:
            fig.show()
    
    def generate_track_plot(self, folder_path:str):
        for i in range(len(self.data["tracks"])):
            if 'points' in self.data["tracks"][i]:
                print(self.data["tracks"][i]['name'])
                self.display_track(i, True, folder_path)
    
    def save_data_normalized(self, path_save:str, points_order:int=0):
        clean_data = {}
        for track in self.data["tracks"]:
            if 'points' in track:
                border_complex = np.array([c[1] + 1j * c[0] for c in track['points']])
                
                if points_order>0:
                    border_complex = np.interp(np.linspace(0, len(border_complex), points_order), np.arange(0, len(border_complex)), border_complex)
                
                fourier_descriptors = np.fft.fft(border_complex, len(border_complex))
                clean_data[track['name']] = {
                    'points': NoIndent(track['points']),
                    'fourier-descriptors': {
                        'real' : NoIndent(fourier_descriptors.real.tolist()),
                        'imag' : NoIndent(fourier_descriptors.imag.tolist())
                    }
                }
        with open(path_save, 'w') as outfile:
            outfile.write(json.dumps(clean_data, cls=MyEncoder, sort_keys=True, indent=4))
            #json.dump(clean_data, outfile, indent=4)
        print("Tracks data saved in", path_save)
        

if __name__=="__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ['1','2','3','4']:
        print("Choose action as argument:")
        print("\t1 - Capture tracks upper-left corners")
        print("\t2 - Generate normalized tracks")
        print("\t3 - Generate tracks plots")
        print("\t4 - Only export tracks points and Fourier Descriptors")
    
    else:
        if sys.argv[1] == '1':
            tracks_extraction = TracksExtraction('./data/racetrackmap_raw.jpg', './data/tracks.json', .45)
            data_out = tracks_extraction.capture_track_positions()
            print("Tracks positionning completed.")
            tracks_extraction.save_data("./data/tracks_positions.json")

        elif sys.argv[1] == '2':
            tracks_extraction = TracksExtraction('./data/racetrackmap_raw.jpg', './data/tracks_positions.json')
            tracks_extraction.tracks_point_generation()
            print("Tracks points generation completed.")
            tracks_extraction.save_data("./data/tracks_extracted.json")
            tracks_extraction.display_track(0)

        elif sys.argv[1] == '3':
            tracks_extraction = TracksExtraction('./data/racetrackmap_raw.jpg', './data/tracks_extracted.json')
            tracks_extraction.generate_track_plot("./data/tracks_plots/")
        
        elif sys.argv[1] == '4':
            tracks_extraction = TracksExtraction('./data/racetrackmap_raw.jpg', './data/tracks_extracted.json')
            tracks_extraction.save_data_normalized("./data/tracks_fourier.json", 2**8)