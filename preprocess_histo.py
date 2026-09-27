import cv2
import numpy as np
import matplotlib.pyplot as plt
from config import GlobalConfig
configx = GlobalConfig()


night = True


#loop pada semua route
route_list = sorted(p.name for p in configx.datadir.iterdir() if p.is_dir())
if configx.select_route != "all":
    route_list = [configx.select_route]
    print(f"only route: {configx.select_route} is selected")
for route in route_list:
    route_path = configx.datadir / route
    img_dir = route_path / "camera" / "rgb"
    save_dir = route_path / "camera" / "histogram"
    ##save_dir_opt = route_path / "camera" / "optical_flow"
    save_dir.mkdir(parents=True, exist_ok=True)
    #save_dir_opt.mkdir(parents=True, exist_ok=True)

    rgb_images = sorted(p.name for p in img_dir.iterdir())

    first_image = cv2.imread(str(img_dir / rgb_images[0]))
    prvs = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(first_image)
    hsv[..., 1] = 255

    for img in rgb_images:
        print(img_dir / img)
        image = cv2.imread(str(img_dir / img))
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        """
        #OPTICAL FLOW
        flow = cv2.calcOpticalFlowFarneback(prvs, gray_image, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv[..., 0] = ang*180/np.pi/2
        hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
        bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        h, w, c = bgr.shape
        bgr2 = cv2.resize(bgr, (int(w/2), int(h/2)))
        cv2.imwrite(save_dir_opt+img, bgr2)

        prvs = gray_image
        """
        #HISTOGRAM
        #load img lalu convert ke gray

        #calculate histogramnya
        hist = cv2.calcHist([gray_image], [0], None, [256], [0, 256])

        #normalize ke 0-100%
        total_pixels = gray_image.shape[0] * gray_image.shape[1]
        hist_normalized = 100 * hist / total_pixels

        #plot
        """
        plt.figure()
        plt.grid(True)
        # plt.title('Histogram')
        # plt.xlabel('Bins')
        # plt.ylabel('Frequency')
        if night:
            plt.gca().set_facecolor('black')
            plt.plot(hist_normalized, color='yellow')
        else:
            plt.plot(hist_normalized)
        plt.xlim([0, 256])
        
        plt.margins(0)
        plt.xticks(np.arange(0, 256, 50))
        plt.yticks(np.arange(0, 21, 5))
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_dir / img)
        """

        # Create a figure and axes
        fig, ax = plt.subplots()

        # Set the background color
        fig.set_facecolor('black')
        ax.set_facecolor('black')

        # Plot the data
        ax.plot(hist_normalized, color='yellow')

        # Set the x and y limits
        ax.set_xlim([0, 256])

        # Turn on the grid
        ax.grid(True, color='white')
        # Set the margins and adjust the layout
        ax.margins(0)
        
        # Set the tick positions
        ax.set_xticks(np.arange(0, 256, 50))
        ax.set_yticks(np.arange(0, 13, 4))
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')


        fig.tight_layout()


        # Set the figure size in inches
        fig.set_size_inches(6, 3)  # HxW = 256x512 pixels (at 100 DPI)



        fig.savefig(save_dir / img, dpi=100)

