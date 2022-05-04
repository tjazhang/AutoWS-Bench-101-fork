import logging
import torch
import numpy as np
import fire
import sklearn
from functools import partial
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.metrics import jaccard_score, accuracy_score
from autosklearn.experimental.askl2 import AutoSklearn2Classifier
from fwrench.embeddings.vae_embedding import VAE2DEmbedding
from fwrench.embeddings.resnet_embedding import ResNetEmbedding

from wrench.dataset import load_dataset
from wrench.logging import LoggingHandler
from wrench.evaluation import f1_score_
from wrench.labelmodel import MajorityVoting, FlyingSquid, Snorkel
from wrench.endmodel import EndClassifierModel, MLPModel
from fwrench.lf_selectors import SnubaSelector, AutoSklearnSelector
from fwrench.embeddings import *
from fwrench.datasets import MNISTDataset

from utils import get_accuracy_coverage

def main(original_lfs=False, dataset_home='./datasets'):
    #### Just some code to print debug information to stdout
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%Y-%m-%d %H:%M:%S',
                        level=logging.INFO,
                        handlers=[LoggingHandler()])
    logger = logging.getLogger(__name__)
    device = torch.device('cuda')
    seed = 123 # TODO do something with this.

    train_data = MNISTDataset('train', name='MNIST')
    valid_data = MNISTDataset('valid', name='MNIST')
    test_data = MNISTDataset('test', name='MNIST')

    data = 'MNIST_3000'
    train_data, valid_data, test_data = load_dataset(
        dataset_home, data, 
        extract_feature=True,
        dataset_type='NumericDataset')

    # TODO hacky... Convert to binary problem
    def convert_to_binary(dset):
        dset.n_class = 2
        dset.id2label = {0: 0, 1: 1}
        for i in range(len(dset.labels)):
            dset.labels[i] = int(dset.labels[i] % 2 == 0)
        return dset
    binary_mode = False
    #train_data = convert_to_binary(train_data)
    #valid_data = convert_to_binary(valid_data)
    #test_data = convert_to_binary(test_data)
    
    # TODO also hacky... normalize data
    def normalize01(dset):
        # NOTE preprocessing... MNIST should be in [0, 1]
        for i in range(len(dset.examples)):
            dset.examples[i]['feature'] = np.array(
                dset.examples[i]['feature']).astype(float)
            dset.examples[i]['feature'] /= float(
                np.max(dset.examples[i]['feature']))
        return dset
    train_data = normalize01(train_data)
    valid_data = normalize01(valid_data)
    test_data = normalize01(test_data)

    # Dimensionality reduction...
    # Try Fred's dim. reduction? -- pretrained ResNet 
    # (not applicable everywhere)
    #emb = PCA(n_components=100)
    #embedder = SklearnEmbedding(emb) # Use PCA or any other sklearn embedding
    #embedder = FlattenEmbedding() # i.e., just use raw features
    #embedder = VAE2DEmbedding()
    embedder = ResNetEmbedding()

    embedder.fit(train_data, valid_data, test_data)
    train_data_embed = embedder.transform(train_data)
    valid_data_embed = embedder.transform(valid_data)
    test_data_embed = embedder.transform(test_data)

    # Fit Snuba with multiple LF function classes and a custom scoring function
    lf_classes = [
        #partial(AutoSklearn2Classifier, 
        #    time_left_for_this_task=30,
        #    per_run_time_limit=30,
        #    memory_limit=50000, 
        #    n_jobs=100),]
        partial(DecisionTreeClassifier, max_depth=1),
        #LogisticRegression
        ]
    scoring_fn = None #accuracy_score
    snuba = SnubaSelector(lf_classes, scoring_fn=scoring_fn)
    # Use Snuba convention of assuming only validation set labels...
    snuba.fit(valid_data_embed, train_data_embed, 
        b=0.1 if not binary_mode else 0.5, # TODO
        cardinality=1, iters=23)
    print(snuba.hg.heuristic_stats())
    # NOTE that snuba uses different F1 score implementations in 
    # different places... 
    # In it uses average='weighted' for computing abstain thresholds
    # and average='micro' for pruning... 
    # Maybe we should try different choices in different places as well?

    train_weak_labels = snuba.predict(train_data_embed)
    train_data.weak_labels = train_weak_labels.tolist()
    valid_weak_labels = snuba.predict(valid_data_embed)
    valid_data.weak_labels = valid_weak_labels.tolist()
    test_weak_labels = snuba.predict(test_data_embed)
    test_data.weak_labels = test_weak_labels.tolist()

    # Get score from majority vote
    #label_model = MajorityVoting()
    #label_model.fit(
    #    dataset_train=train_data,
    #    dataset_valid=valid_data
    #)
    #logger.info(f'---Majority Vote eval---')
    #acc = label_model.test(train_data, 'acc')
    #logger.info(f'label model (MV) train acc:    {acc}')
    #acc = label_model.test(valid_data, 'acc')
    #logger.info(f'label model (MV) valid acc:    {acc}')
    #acc = label_model.test(test_data, 'acc')
    #logger.info(f'label model (MV) test acc:     {acc}')

    # Get score from Snorkel (afaik, this is the default Snuba LM)
    label_model = Snorkel()
    label_model.fit(
        dataset_train=train_data,
        dataset_valid=valid_data
    )
    logger.info(f'---Snorkel eval---')
    acc = label_model.test(train_data, 'acc')
    logger.info(f'label model (Snorkel) train acc:    {acc}')
    acc = label_model.test(valid_data, 'acc')
    logger.info(f'label model (Snorkel) valid acc:    {acc}')
    acc = label_model.test(test_data, 'acc')
    logger.info(f'label model (Snorkel) test acc:     {acc}')
    # NOTE these values are misleading;
    # WRENCH reports accuracy using soft labels and often 
    # just collapses to majority class

    # Train end model
    #### Filter out uncovered training data
    train_data_covered = train_data.get_covered_subset()
    aggregated_hard_labels = label_model.predict(train_data_covered)
    aggregated_soft_labels = label_model.predict_proba(train_data_covered)


    # Get actual label model accuracy using hard labels
    get_accuracy_coverage(train_data, label_model, split='train')
    get_accuracy_coverage(valid_data, label_model, split='valid')
    get_accuracy_coverage(test_data, label_model, split='test')


    coverage = aggregated_soft_labels.shape[0] / len(train_data_embed.labels)
    print(f'coverage = {coverage:.4f}')

    # TODO train end model on combination of real valid labels and the estimated train labels. 
    # Does it do better than just training on the valid labels?
    model = EndClassifierModel(
        batch_size=512,
        test_batch_size=512,
        n_steps=1_000, # Increase this to 100_000 if needed
        backbone='LENET',
        optimizer='SGD',
        optimizer_lr=1e-3,
        optimizer_weight_decay=0.0,
        binary_mode=binary_mode,
    )
    model.fit(
        dataset_train=train_data_covered,
        y_train=aggregated_soft_labels,
        #y_train=aggregated_hard_labels,
        dataset_valid=valid_data,
        evaluation_step=50,
        metric='acc',
        patience=1000,
        device=device
    )
    logger.info(f'---LeNet eval---')
    acc = model.test(test_data, 'acc')
    logger.info(f'end model (LeNet) test acc:    {acc}')

if __name__ == '__main__':
    fire.Fire(main)
