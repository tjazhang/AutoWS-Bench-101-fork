#running experiment on amazon high card with different labelling points

source activate AutoWS-Bench-101
export PYTHONPATH="/gscratch/efml/tzhang26/AutoWS-Bench-101-fork"

# python fwrench/applications/pipeline.py --dataset amazon-high-card --lf_selector snuba_multiclass -nlp 100 --snuba_iterations 1
# python fwrench/applications/pipeline.py --dataset amazon-high-card --lf_selector snuba_multiclass -nlp 100 -scs=1000 -sc=10 --snuba_iterations 10
# python fwrench/applications/pipeline.py --dataset amazon-high-card --lf_selector snuba_multiclass -nlp 300 -scs=1000 -sc=10 --snuba_iterations 10


# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 300 --snuba_iterations 10
# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 100 -scs=1000 -sc=10 --snuba_iterations 10


# python fwrench/applications/pipeline.py --dataset youtube
# python fwrench/applications/pipeline.py --dataset amazon-high-card --lf_selector snuba_multiclass -nlp 300 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset amazon-high-card --lf_selector snuba_multiclass -nlp 1500 -scs=1000 -sc=10

# python fwrench/applications/pipeline.py --dataset yelp --lf_selector snuba_multiclass

# python fwrench/applications/pipeline.py --dataset imdb --lf_selector snuba_multiclass

# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 100 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 300 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 500 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset banking-high-card --lf_selector snuba_multiclass -nlp 1000 -scs=1000 -sc=10

# python fwrench/applications/pipeline.py --dataset news-category --lf_selector snuba_multiclass -nlp 100 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset news-category --lf_selector snuba_multiclass -nlp 300 -scs=1000 -sc=10
# python fwrench/applications/pipeline.py --dataset news-category --lf_selector snuba_multiclass -nlp 500 -scs=1000 -sc=10
python fwrench/applications/pipeline.py --dataset news-category --lf_selector snuba_multiclass -nlp 2000 -scs=1000 -sc=10



