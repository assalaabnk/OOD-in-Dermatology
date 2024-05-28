
from odin import norm_perturbations
from scipy.special import softmax
import os
import numpy as np
import keras.backend as K
import cv2
from PIL import Image
from odin import norm_perturbations
from odin import get_tpr_and_fpr, auroc
from scipy.special import softmax
from tqdm import trange
from sklearn.metrics import roc_curve, auc
import tensorflow as tf
import tensorflow
from tensorflow import keras
from keras.layers import Dense






def compute_baseline_softmax_scores(model, generator_in, generator_out):
    """
    Calculate the base confidence of the output, no perturbation added here, no temperature scaling used.
    Directly copy the original prediction results.
    """
    print("Begin to compute baseline softmax scores")
    distributions = ["In", "Out"]

    for dist in distributions:
        if dist == "In":
            generator = generator_in
        elif dist == "Out":
            generator = generator_out

        with open("./densenet121_Base_{}.txt".format(dist), "w") as f:
            for i in range(len(generator.fids)):
                # print('thisFid: '+str(generator.fids[i]))
                thisImg = Image.open(generator.fids[i] + ".jpg").convert("RGB")
                thisImg = np.expand_dims(
                    np.asarray(thisImg.resize((224, 224))) / 255, axis=0
                )
                softmax_probs = model.predict(thisImg)
                softmax_score = np.max(softmax_probs)
                f.write("{}\n".format(softmax_score))
        f.close()


def get_perturbation_helper_func(model, temperature, num_classes):
    """Return Keras functions for calculating perturbations."""
    # Compute loss based on the second last layer's output and temperature scaling
    dense_pred_layer_output = model.get_layer("dense_4").output
    scaled_dense_pred_output = dense_pred_layer_output / temperature

    print(dense_pred_layer_output)
    print(scaled_dense_pred_output)

    print(K.argmax(model.outputs))

    label_tensor = K.one_hot(K.argmax(model.outputs), 1)
    tf.print(label_tensor)
    loss = K.sparse_categorical_crossentropy(
        label_tensor, scaled_dense_pred_output, from_logits=True
    )
    grad_loss = K.gradients(loss, model.inputs)

    compute_perturbations = K.function(model.inputs + [K.learning_phase()], grad_loss)
    get_scaled_dense_pred_output = K.function(
        model.inputs + [K.learning_phase()], [scaled_dense_pred_output]
    )

    return compute_perturbations, get_scaled_dense_pred_output



def compute_odin_softmax_scores(model, generator_in, generator_out, num_classes=6, batch_size=1):
    distributions = ['In', 'Out']

    # This file is used for recording what parameter combinations were already computed.
    progress_file = os.path.join('Done.txt')
    done_set = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            done_set = set(line.rstrip('\n') for line in f)

    # ODIN parameters
    model_name = 'DenseNet201'
    temperatures = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]
    #temperatures 200

    magnitudes = np.round(np.arange(0.0002, 0.0041, 0.0002), 4)
    #magnitudes = np.round([0.0002, 0.0041], 4)
    need_norm_perturbations = 1
    image_data_format = K.image_data_format()
    learning_phase = 0  # 0 = test, 1 = train

    for temperature in temperatures:
        compute_perturbations, get_scaled_dense_pred_output = get_perturbation_helper_func(model, temperature, num_classes)

        for magnitude in magnitudes:
            for dist in distributions:
                # Skip if the parameter combination has done
                param_comb_id = "{}, {}, {}".format(dist, temperature, magnitude)
                if param_comb_id in done_set:
                    print('Skip ', param_comb_id)
                    continue

                if dist == 'In':
                    generator = generator_in
                elif dist == 'Out':
                    generator = generator_out

                print("\n===== Temperature: {}, Magnitude: {}, {}-Distribution =====".format(temperature, magnitude, dist))

                # Check if the output file already exists, and if it does, skip processing
                output_file_path = "./densenet121_ODIN_{}_{}_{}.txt".format(temperature, magnitude, dist)
                if os.path.exists(output_file_path):
                    print(f"Output file {output_file_path} already exists. Skipping...")
                    continue

                try:
                    f = open(output_file_path, 'w')
                    for i in trange(len(generator.fids)):
                        thisFid = generator.fids[i]
                        images = np.expand_dims(cv2.resize(np.array(Image.open(thisFid + '.jpg').convert("RGB")) / 255,
                                                           (224, 224), interpolation=cv2.INTER_LINEAR), axis=0)

                        perturbations = compute_perturbations([images, learning_phase])[0]
                        perturbations = np.sign(perturbations)

                        if need_norm_perturbations:
                            perturbations = norm_perturbations(perturbations, image_data_format)

                        perturbative_images = images - magnitude * perturbations

                        dense_pred_outputs = get_scaled_dense_pred_output([perturbative_images, learning_phase])[0]
                        softmax_probs = softmax(dense_pred_outputs)
                        softmax_scores = np.max(softmax_probs, axis=-1)

                        for s in softmax_scores:
                            f.write("{}\n".format(s))
                    f.close()

                    with open(progress_file, 'a') as f_done:
                        f_done.write("{}\n".format(param_comb_id))

                except Exception as e:
                    # Handle any exceptions that might occur during file processing
                    print(f"An error occurred for {param_comb_id}: {str(e)}")
                    continue

                last_processed_combination = param_comb_id


def compute_odin_softmax_scores_fixed(model, generator_in, generator_out, num_classes=6, batch_size=1):
    distributions = ['Out']
    compute_perturbations, get_scaled_dense_pred_output = get_perturbation_helper_func(model, optimal_temperature, num_classes)

    # This file is used for recording what parameter combinations were already computed.
    progress_file = os.path.join('Done.txt')
    done_set = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            done_set = set(line.rstrip('\n') for line in f)

    image_data_format = K.image_data_format()
    learning_phase = 0  # 0 = test, 1 = train
    need_norm_perturbations = 1

    for dist in distributions:
        param_comb_id = "{}, {}, {}".format(dist, optimal_temperature, optimal_magnitude)
        if param_comb_id in done_set:
            print('Skip ', param_comb_id)
            continue

        if dist == 'In':
            generator = generator_in
        elif dist == 'Out':
            generator = generator_out

        print("\n===== Temperature: {}, Magnitude: {}, {}-Distribution =====".format(optimal_temperature, optimal_magnitude, dist))

        f = open("./densenet121_ODIN_{}_{}_{}.txt".format(optimal_temperature, optimal_magnitude, dist), 'w')

        for i in trange(len(generator.fids)):
            try:
                thisFid = generator.fids[i]
                images = np.expand_dims(
                    cv2.resize(np.array(Image.open(thisFid + '.jpg').convert("RGB")) / 255, (224, 224),
                               interpolation=cv2.INTER_LINEAR), axis=0)

                perturbations = compute_perturbations([images, learning_phase])[0]
                # Get sign of perturbations
                perturbations = np.sign(perturbations)

                # Normalize the perturbations to the same space of image
                if need_norm_perturbations:
                    perturbations = norm_perturbations(perturbations, image_data_format)

                # Add perturbations to images
                perturbative_images = images - optimal_magnitude * perturbations

                # Calculate the confidence after adding perturbations
                dense_pred_outputs = get_scaled_dense_pred_output([perturbative_images, learning_phase])[0]
                softmax_probs = softmax(dense_pred_outputs)
                softmax_scores = np.max(softmax_probs, axis=-1)
                for s in softmax_scores:
                    f.write("{}\n".format(s))
            except FileNotFoundError:
                print(f"File not found for index {i}. Continuing...")
                continue

        f.close()

        with open(progress_file, 'a') as f_done:
            f_done.write("{}\n".format(param_comb_id))







def compute_odin_parameters_and_metrics(model, generator_in, generator_out, num_classes=6, batch_size=1):
    distributions = ['In', 'Out']
    compute_perturbations, get_scaled_dense_pred_output = get_perturbation_helper_func(model, optimal_temperature, num_classes)

    # This file is used for recording what parameter combinations were already computed.
    progress_file = os.path.join('Done.txt')
    done_set = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            done_set = set(line.rstrip('\n') for line in f)

    image_data_format = K.image_data_format()
    learning_phase = 0  # 0 = test, 1 = train

    need_norm_perturbations = 1

    # Initialize variables to store metrics
    auroc_max = -1  # Initialize AUROC to a very low value
    fpr_min = 1     # Initialize FPR to a very high value
    odinparam_auroc_max = None
    odinparam_fpr_min = None

    for dist in distributions:
        for magnitude in [optimal_magnitude]:
            # Skip if the parameter combination has been done
            param_comb_id = "{}, {}, {}".format(dist, optimal_temperature, magnitude)
            if param_comb_id in done_set:
                print('Skip ', param_comb_id)
                continue

            if dist == 'In':
                generator = generator_in
            elif dist == 'Out':
                generator = generator_out

            print("\n===== Temperature: {}, Magnitude: {}, {}-Distribution =====".format(optimal_temperature, magnitude, dist))

            f = open("./densenet121_ODIN_{}_{}_{}.txt".format(optimal_temperature, magnitude, dist), 'w')

            # Start from the beginning
            for i in trange(len(generator.fids)):
                try:
                    thisFid = generator.fids[i]
                    images = np.expand_dims(
                        cv2.resize(np.array(Image.open(thisFid + '.jpg').convert("RGB")) / 255, (224, 224),
                                   interpolation=cv2.INTER_LINEAR), axis=0)

                    perturbations = compute_perturbations([images, learning_phase])[0]
                    # Get the sign of perturbations
                    perturbations = np.sign(perturbations)

                    # Normalize the perturbations to the same space of the image
                    if need_norm_perturbations:
                        perturbations = norm_perturbations(perturbations, image_data_format)

                    # Add perturbations to images
                    perturbative_images = images - magnitude * perturbations

                    # Calculate the confidence after adding perturbations
                    dense_pred_outputs = get_scaled_dense_pred_output([perturbative_images, learning_phase])[0]
                    softmax_probs = softmax(dense_pred_outputs)
                    softmax_scores = np.max(softmax_probs, axis=-1)
                    for s in softmax_scores:
                        f.write("{}\n".format(s))
                except FileNotFoundError:
                    print(f"File not found for index {i}. Continuing...")
                    continue

            f.close()

            with open(progress_file, 'a') as f_done:
                f_done.write("{}\n".format(param_comb_id))






'''
Commented function ...


def compute_odin_softmax_scores(model, generator_in, generator_out, num_classes = 6, batch_size = 1):
    distributions = ['In', 'Out']

    # This file is used for recording what parameter combinations were already computed.
    progress_file = os.path.join('Done.txt')
    done_set = set()
    if os.path.exists(progress_file):
        with open(progress_file, 'r') as f:
            done_set = set(line.rstrip('\n') for line in f)

    # ODIN parameters

    model_name = 'DenseNet201'
    #Grid Search
    temperatures = [1000, 500, 200, 100, 50, 20, 10, 5, 2, 1]

    magnitudes = np.round(np.arange(0, 0.0041, 0.0002), 4)

    need_norm_perturbations = 1



#     model_param_map = get_transfer_model_param_map()
    image_data_format = K.image_data_format()
    learning_phase = 0 # 0 = test, 1 = train



    for temperature in temperatures:
        compute_perturbations, get_scaled_dense_pred_output = get_perturbation_helper_func(model, temperature, num_classes)

        for magnitude in magnitudes:
            for dist in distributions:
                # Skip if the parameter combination has done
                param_comb_id = "{}, {}, {}".format( dist, temperature, magnitude)
                if param_comb_id in done_set:
                    print('Skip ', param_comb_id)
                    continue

                if dist == 'In':
                    generator = generator_in
                elif dist == 'Out':
                    generator = generator_out

                print("\n===== Temperature: {}, Magnitude: {}, {}-Distribution =====".format(temperature, magnitude, dist))

                try:
                    f = open("./densenet121_ODIN_{}_{}_{}.txt".format(temperature, magnitude,dist), 'w')
                    for i in trange(len(generator.fids)):
                        thisFid = generator.fids[i]
                        images = np.expand_dims(cv2.resize(np.array(Image.open(thisFid+'.jpg').convert("RGB"))/255, (224,224), interpolation=cv2.INTER_LINEAR), axis = 0)

                        perturbations = compute_perturbations([images, learning_phase])[0]
                    # Get sign of perturbations
                        perturbations = np.sign(perturbations)

                    # Normalize the perturbations to the same space of image
                    # https://github.com/facebookresearch/odin/issues/5
                    # Perturbations divided by ISIC Training Set STD
                        if need_norm_perturbations:
                            perturbations = norm_perturbations(perturbations, image_data_format)

                    # Add perturbations to images
                        perturbative_images = images - magnitude * perturbations

                    # Calculate the confidence after adding perturbations
                        dense_pred_outputs = get_scaled_dense_pred_output([perturbative_images, learning_phase])[0]
                        softmax_probs = softmax(dense_pred_outputs)
                        softmax_scores = np.max(softmax_probs, axis=-1)

                        for s in softmax_scores:
                            f.write("{}\n".format(s))
                    f.close()

                    with open(progress_file, 'a') as f_done:
                        f_done.write("{}\n".format(param_comb_id))

                except FileNotFoundError:
                    # Handle the "File not found" error
                    print(f"File not found for {param_comb_id}. Skipping...")
                    continue
                last_processed_combination = param_comb_id'''