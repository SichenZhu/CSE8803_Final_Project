import os
from glob import glob
from PIL import Image
import numpy as np
from tqdm import tqdm
from dgl.data.utils import save_graphs

from histocartography.utils import download_example_data
from histocartography.preprocessing import (
    NucleiExtractor,
    DeepFeatureExtractor,
    KNNGraphBuilder
)
from histocartography.visualization import OverlayGraphVisualization

def generate_cell_graph(image_path, output_path):
    """
    Generate a cell graph for all the images in image path dir.
    """

    # 1. get image path
    image_fnames = glob(os.path.join(image_path, '*.png'))

    # 2. define nuclei extractor
    nuclei_detector = NucleiExtractor()

    # 3. define feature extractor: Extract patches of 72x72 pixels around each
    # nucleus centroid, then resize to 224 to match ResNet input size.
    feature_extractor = DeepFeatureExtractor(
        architecture='resnet34',
        patch_size=72
        # resize_size=224
    )

    # 4. define k-NN graph builder with k=5 and thresholding edges longer
    # than 50 pixels. Add image size-normalized centroids to the node features.
    # For e.g., resulting node features are 512 features from ResNet34 + 2
    # normalized centroid features.
    knn_graph_builder = KNNGraphBuilder(k=5, thresh=50, add_loc_feats=True)

    # 5. define graph visualizer
    visualizer = OverlayGraphVisualization()

    # 6. process all the images
    for image_path in tqdm(image_fnames):

        # a. load image
        _, image_name = os.path.split(image_path)
        image = np.array(Image.open(image_path))

        # b. extract nuclei
        nuclei_map, _ = nuclei_detector.process(image)

        # c. extract deep features
        features = feature_extractor.process(image, nuclei_map)

        # d. build a kNN graph
        graph = knn_graph_builder.process(nuclei_map, features)

        # e. save the graph
        fname = image_name.replace('.png', '.bin')
        save_graphs(os.path.join(output_path, 'cell_graphs', fname), [graph])

        # f. visualize and save the graph
        canvas = visualizer.process(image, graph, instance_map=nuclei_map)
        canvas.save(os.path.join(output_path, 'cell_graphs_viz', image_name))


image_path="/storage/home/hcocice1/gye31/cell_image_dataset_2_6/train/"
output_path = "/storage/home/hcocice1/gye31/cell_graph_dataset_2_6/train/"

os.makedirs(os.path.join(output_path, 'cell_graphs'), exist_ok=True)
os.makedirs(os.path.join(output_path, 'cell_graphs_viz'), exist_ok=True)

generate_cell_graph(image_path=image_path, output_path=output_path)