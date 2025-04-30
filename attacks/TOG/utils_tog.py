from dataset_utils.preprocessing import letterbox_image_padded
from PIL import Image
from tog.attacks import *
import matplotlib.pyplot as plt

eps = 8 / 255.       # Hyperparameter: epsilon in L-inf norm
eps_iter = 2 / 255.  # Hyperparameter: attack learning rate
n_iter = 10          # Hyperparameter: number of attack iterations

def plot_and_save_image(x_query, output_path):
    plt.clf()
    plt.figure()
    input_img = x_query
    if len(input_img.shape) == 4:
        input_img = input_img[0]
    plt.imshow(input_img)
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()

def tog_untargeted_attack(detector, image_input_path, output_path):
    print('Running untargeted attack...')
    input_img = Image.open(image_input_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_adv_untargeted = tog_untargeted(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    plot_and_save_image(x_adv_untargeted, output_path)

def tog_vanishing_attack(detector, image_input_path, output_path):
    print('Running vanishing attack...')
    input_img = Image.open(image_input_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_adv_vanishing = tog_vanishing(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    plot_and_save_image(x_adv_vanishing, output_path)

def tog_fabrication_attack(detector, image_input_path, output_path):
    print('Running fabrication attack...')
    input_img = Image.open(image_input_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_adv_fabricated = tog_fabrication(victim=detector, x_query=x_query, n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    plot_and_save_image(x_adv_fabricated, output_path)

def tog_mislabeling_attack_most_likely(detector, image_input_path, output_path):
    '''
    This attack consistently causes the victim detector to misclassify the objects detected on the input image by replacing 
    their source class label with the maliciously chosen target class label, while maintaining the same set of correct 
    bounding boxes. Such an attack can cause fatal collisions in many scenarios, e.g., misclassifying the stop sign as an 
    umbrella. We can conduct the most-likely (ML) class attack by setting the attack targets to the incorrect class label 
    with the highest probability predicted by the victim or the least-likely (LL) class attack with the lowest probability.
    '''
    print('Running mislabeling attack (most likely)...')
    input_img = Image.open(image_input_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_adv_mislabeled = tog_mislabeling(victim=detector, x_query=x_query, target='ml', n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    plot_and_save_image(x_adv_mislabeled, output_path)

def tog_mislabeling_attack_least_likely(detector, image_input_path, output_path):
    print('Running mislabeling attack (least likely)...')
    input_img = Image.open(image_input_path)
    x_query, x_meta = letterbox_image_padded(input_img, size=detector.model_img_size)
    x_adv_mislabeled = tog_mislabeling(victim=detector, x_query=x_query, target='ll', n_iter=n_iter, eps=eps, eps_iter=eps_iter)
    plot_and_save_image(x_adv_mislabeled, output_path)