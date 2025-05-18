import pickle

# Load the pickle file
with open(r'C:\Users\lakshay\Desktop\detection\Co-DETR\results\bboxes2.pkl', 'rb') as f:
    data = pickle.load(f)

# # Write to a text file
with open(r'C:\Users\lakshay\Desktop\detection\Co-DETR\results\bboxes2.txt', 'w') as out_file:
    for entry in data:
        out_file.write(str(entry) + '\n')


# import os
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from PIL import Image
# import numpy as np

# def draw_and_save_bboxes_from_results(results, image_dir, output_dir, class_names=None, score_thr=0.3):
#     os.makedirs(output_dir, exist_ok=True)
#     num_classes = len(results[0]) - 1  # last entry is filename

#     # Define color palette
#     colors = ['r', 'g', 'b', 'y', 'c', 'm', 'orange', 'lime', 'cyan', 'pink', 'purple', 'gold']

#     for result in results:
#         bboxes_per_class = result[:num_classes]  # list of 12 arrays
#         filename = result[-1].split('/')[1]
#         image_path = os.path.join(image_dir, filename)

#         if not os.path.exists(image_path):
#             print(f"Image not found: {image_path}")
#             continue

#         # Load image
#         img = Image.open(image_path).convert('RGB')
#         fig, ax = plt.subplots(1)
#         ax.imshow(img)

#         for class_idx, bboxes in enumerate(bboxes_per_class):
#             color = colors[class_idx % len(colors)]
#             label = class_names[class_idx] if class_names else f'class_{class_idx}'

#             if isinstance(bboxes, np.ndarray) and len(bboxes) > 0:
#                 for bbox in bboxes:
#                     if bbox[-1] < score_thr:
#                         continue  # Skip low-confidence boxes
#                     x1, y1, x2, y2, score = bbox
#                     rect = patches.Rectangle(
#                         (x1, y1), x2 - x1, y2 - y1,
#                         linewidth=2, edgecolor=color, facecolor='none'
#                     )
#                     ax.add_patch(rect)
#                     ax.text(x1, y1 - 5, f'{label} {score:.2f}', color='white',
#                             fontsize=8, bbox=dict(facecolor=color, edgecolor='none', alpha=0.7))

#         ax.axis('off')

#         output_path = os.path.join(output_dir, os.path.basename(filename))
#         plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
#         plt.close(fig)

#         print(f"Saved: {output_path}")

# print(os.curdir)
# draw_and_save_bboxes_from_results(
#     data,                 # your list of 13-element lists
#     image_dir='Data\Det\\test\images',     # folder where raw images are stored
#     output_dir='results\images_with_bbox_14',    # where to save the drawn images
#     class_names=['0','1','2','3','4','5','6','7','8','9','10','11'
#     ],
#     score_thr=0.3
# )
