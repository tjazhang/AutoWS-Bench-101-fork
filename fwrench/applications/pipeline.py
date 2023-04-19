import logging
import random
import copy

import argparse
import contextlib
import gc
import json
import logging
import multiprocessing
import os
import signal
import sys
from pprint import pprint

import warnings

import fire
import fwrench.embeddings as feats
import fwrench.utils.autows as autows
import fwrench.utils.data_settings as settings
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.metrics import top_k_accuracy_score
from wrench.logging import LoggingHandler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="An pipeline manager for the AutoWS-Bench-101")

    parser.add_argument("-d", "--dataset", help="Dataset to run experiments on (default: youtube)", default="youtube")
    parser.add_argument("-e", "--embedding", help="raw | pca | resnet18 | vae", default="raw")
    parser.add_argument("-r", "--root", help="Directory root storing all the datasets (default: ./)",
                        default="/gscratch/efml/tzhang26/limits-of-ws/weak_datasets")
    parser.add_argument("-ef", "--extract_fn", help="Extraction function used for dataset (text only)", default="bert")

    parser.add_argument("-lfs", "--lf_selector", default="snuba")
    parser.add_argument("-hl", "--hard_label", help="use hard label for label model (default: False)",
                        action='store_true')
    parser.add_argument("-nlp", "--n_labeled_points", type = int, default=100)
    parser.add_argument("-scs", "--snuba_combo_samples", type = int, default=-1)
    parser.add_argument("-sc", "--snuba_cardinality", type = int, default=1)
    parser.add_argument("-si", "--snuba_iterations", type = int, default=23)
    parser.add_argument("-lco", "--lf_class_options", default="default")


    args = parser.parse_args()

    dataset= args.dataset
    dataset_home= args.root
    embedding=args.embedding  # raw | pca | resnet18 | vae
    # text dataset only
    extract_fn = args.extract_fn # bow | bert | tfidf | sentence_transformer
    #
    # Goggles options
    goggles_method="SemiGMM" # SemiGMM | KMeans | Spectral
    #
    lf_selector=args.lf_selector # snuba | interactive | goggles

    em_hard_labels=False
    if args.hard_label:
        em_hard_labels=True  # Use hard or soft labels for end model training

    n_labeled_points=args.n_labeled_points  # Number of points used to train lf_selector
    #
    # Snuba options
    snuba_combo_samples=args.snuba_combo_samples # -1 uses all feat. combos
    # TODO this needs to work for Snuba and IWS
    snuba_cardinality=args.snuba_cardinality # Only used if lf_selector='snuba'
    
    iws_cardinality=1
    snuba_iterations=args.snuba_iterations
    lf_class_options=args.lf_class_options # default | comma separated list of lf classes to use in the selection procedure. Example: 'DecisionTreeClassifier,LogisticRegression'
    #
    # Interactive Weak Supervision options
    iws_iterations=25
    iws_auto = True
    iws_usefulness = 0.6


    seed=123
    prompt=None

    ################ HOUSEKEEPING/SELF-CARE 😊 ################################
    random.seed(seed)
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[LoggingHandler()],
    )
    logger = logging.getLogger(__name__)
    device = torch.device("cuda")

    ################ LOAD DATASET #############################################

    if dataset == "mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "fashion_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_fashion_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "kmnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_kmnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "cifar10":
        train_data, valid_data, test_data, k_cls, model = settings.get_cifar10(
            n_labeled_points, dataset_home
        )
    elif dataset == "spherical_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_spherical_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "permuted_mnist":
        train_data, valid_data, test_data, k_cls, model = settings.get_permuted_mnist(
            n_labeled_points, dataset_home
        )
    elif dataset == "ecg":
        train_data, valid_data, test_data, k_cls, model = settings.get_ecg(
            n_labeled_points, dataset_home
        )
    elif dataset == "ember":
        train_data, valid_data, test_data, k_cls, model = settings.get_ember_2017(
            n_labeled_points, dataset_home
        )
    elif dataset == "navier_stokes":
        train_data, valid_data, test_data, k_cls, model = settings.get_navier_stokes(
            n_labeled_points, dataset_home
        )
    elif dataset == "imdb":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_imdb(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_imdb(
                n_labeled_points, dataset_home, extract_fn
            )
    elif dataset == "yelp":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_yelp(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_yelp(
                n_labeled_points, dataset_home, extract_fn
            )
    #small dataset, only for testing 
    elif dataset == "youtube":
        if embedding == 'openai' or embedding == 'clip' or embedding == 'clip_zeroshot':
            train_data, valid_data, test_data, k_cls, model = settings.get_youtube(
                n_labeled_points, dataset_home, extract_fn=None
            )
        else:
            train_data, valid_data, test_data, k_cls, model = settings.get_youtube(
                n_labeled_points, dataset_home, extract_fn
            )
    elif dataset == "amazon-high-card":
        train_data, valid_data, test_data, k_cls, model = settings.get_amazon_high_card(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "banking-high-card":
        train_data, valid_data, test_data, k_cls, model = settings.get_banking_high_card(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "news-category":
        train_data, valid_data, test_data, k_cls, model = settings.get_news_category(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "dbpedia-219":
        train_data, valid_data, test_data, k_cls, model = settings.get_dbpedia_219(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "dbpedia":
        train_data, valid_data, test_data, k_cls, model = settings.get_dbpedia(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "dbpedia-111":
        train_data, valid_data, test_data, k_cls, model = settings.get_dbpedia_111(
                n_labeled_points, dataset_home, extract_fn
    )
    elif dataset == "massive-lowcard":
        train_data, valid_data, test_data, k_cls, model = settings.get_massive_lowcard(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "banking77":
        train_data, valid_data, test_data, k_cls, model = settings.get_banking77(
                n_labeled_points, dataset_home, extract_fn
        )
    elif dataset == "amazon31":
        train_data, valid_data, test_data, k_cls, model = settings.get_amazon31(
                n_labeled_points, dataset_home, extract_fn
        )
    else:
        raise NotImplementedError

    ################ FEATURE REPRESENTATIONS ##################################
    if embedding == "raw":
        embedder = feats.FlattenEmbedding()
    elif embedding == "pca":
        emb = PCA(n_components=100)
        embedder = feats.SklearnEmbedding(emb)
    elif embedding == "resnet18":
        embedder = feats.ResNet18Embedding(dataset)
    elif embedding == "vae":
        embedder = feats.VAE2DEmbedding()
    elif embedding == "clip":
        embedder = feats.CLIPEmbedding()
    elif embedding == "clip_zeroshot":
        embedder = feats.ZeroShotCLIPEmbedding(dataset=dataset, prompt=prompt)
    elif embedding == "oracle":
        embedder = feats.OracleEmbedding(k_cls)
    elif embedding == "openai":
        embedder = feats.OpenAICLIPEmbedding(dataset=dataset, prompt=prompt)
    else:
        raise NotImplementedError

    if ((embedding == "resnet18") and (dataset == "ecg")) or ((embedding == "resnet18") and (dataset == "ember")):
        embedder.fit(valid_data, test_data)
        valid_data_embed = embedder.transform(valid_data)
        test_data_embed = embedder.transform(test_data)
        train_data_embed = copy.deepcopy(valid_data_embed)
        train_data = copy.deepcopy(valid_data)
    else:
        embedder.fit(train_data, valid_data, test_data)
        train_data_embed = embedder.transform(train_data)
        valid_data_embed = embedder.transform(valid_data)
        test_data_embed = embedder.transform(test_data)

    ################ AUTOMATED WEAK SUPERVISION ###############################
    if lf_selector == "snuba":
        test_covered, hard_labels, soft_labels = autows.run_snuba(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            snuba_cardinality,
            snuba_combo_samples,
            snuba_iterations,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "snuba_multiclass":
        test_covered, hard_labels, soft_labels = autows.run_snuba_multiclass(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            snuba_cardinality,
            snuba_combo_samples,
            snuba_iterations,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "iws":
        test_covered, hard_labels, soft_labels = autows.run_iws(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            iws_cardinality,
            iws_iterations,
            iws_auto,
            iws_usefulness,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "iws_multiclass":
        test_covered, hard_labels, soft_labels = autows.run_iws_multiclass(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            iws_cardinality,
            iws_iterations,
            iws_auto,
            lf_class_options,
            k_cls,
            logger,
        )
    elif lf_selector == "goggles":
        test_covered, hard_labels, soft_labels = autows.run_goggles(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            goggles_method,
            logger,
        )
    elif lf_selector == "supervised":
        test_covered, hard_labels, soft_labels = autows.run_supervised(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    elif lf_selector == "label_prop":
        test_covered, hard_labels, soft_labels = autows.run_label_propagation(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    elif lf_selector == "clip_zero_shot" and (
        embedding == "clip_zeroshot" or embedding == "oracle" or embedding == "openai"
    ):
        test_covered, hard_labels, soft_labels = autows.run_zero_shot_clip(
            valid_data,
            train_data,
            test_data,
            valid_data_embed,
            train_data_embed,
            test_data_embed,
            logger,
        )
    else:
        raise NotImplementedError

    acc = accuracy_score(test_covered.labels, hard_labels)
    acc_top5 = top_k_accuracy_score(test_covered.labels, soft_labels, k = 5, labels = np.arange(k_cls))
    cov = float(len(test_covered.labels)) / float(len(test_data.labels))
    print( dataset, embedding, extract_fn, lf_selector, n_labeled_points, snuba_cardinality)
    logger.info(f"label model test acc:    {acc}")
    logger.info(f"top 5 label model test acc:   {acc_top5}")
    logger.info(f"label model coverage:    {cov}")

    with open("./results/{}.txt".format(args.dataset), "a+") as outfile:
        outfile.write(f"embedding: {args.embedding}, extract_fn: {args.extract_fn}, lf_selector: {args.lf_selector}, \nn_labeled_points: {args.n_labeled_points}, snuba_cardinality: {args.snuba_cardinality}, snuba_iterations: {args.snuba_iterations}, snuba_combo_samples: {args.snuba_combo_samples} \n")
        outfile.write(f"label model test acc:    {acc}\nlabel model coverage:     {cov} \ntop 5 label model test acc:   {acc_top5}\n")
        outfile.write("val-data label counts: " + np.array_str(np.bincount(valid_data.labels, minlength = k_cls))+ "\n\n")

    ################ TRAIN END MODEL ##########################################
    # model.fit(
    #     dataset_train=train_covered,
    #     y_train=hard_labels if em_hard_labels else soft_labels,
    #     dataset_valid=valid_data,
    #     evaluation_step=50,
    #     metric="acc",
    #     patience=1000,
    #     device=device,
    # )
    # logger.info(f"---LeNet eval---")
    # acc = model.test(test_data, "acc")
    # logger.info(f"end model (LeNet) test acc:    {acc}")
    ################ PROFIT 🤑 #################################################

